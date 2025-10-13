import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from models.clip import clip
import torch.nn.functional as F

from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.classification.base_learner import VLBaseLearner
from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .losses import LossRegistry # ABHISHEK

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_name = cfg.MODEL.NAME if hasattr(cfg.MODEL, 'NAME') else 'clip'
    
    if model_name == 'clip':
        # Original CLIP loading
        print("\n\n\nUsing CLIP\n\n\n")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    elif model_name == 'plip':
        # PLIP path
        print("\n\n\nUsing PLIP\n\n\n")
        model_path = osp.join(cfg.MODEL_ROOT, "plip", 'plip_vit_b32.pt')
    elif model_name == 'quiltnet':
        # QuiltNet path
        print("\n\n\nUsing QuiltNet\n\n\n")
        model_path = osp.join(cfg.MODEL_ROOT, "quiltnet", 'quiltnet_b32.pt')
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose 'clip' or 'plip' or 'quiltnet")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'CoOp',
        "vision_depth": 0,
        "language_depth": 0, 
        "vision_ctx": 0,
        "language_ctx": 0
    }
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION if hasattr(cfg.TRAINER.COOP, "CLASS_TOKEN_POSITION") else "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

# Add CustomCLIP class for prompt learning
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # loss configuration
        self.enabled_losses = cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES
        # Store cfg for loss functions
        self.cfg = cfg
        # Initialize weights in LossRegistry
        LossRegistry.init_weights(cfg)

    def compute_losses(self, logits, label, features=None):
        """
        Compute all enabled losses and return their weighted sum
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            label: Ground truth labels (batch_size,)
            features: Optional feature embeddings for feature-based losses
        
        Returns:
            losses: Dictionary containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        for loss_name in self.enabled_losses:
            loss_fn = LossRegistry.get_loss(loss_name)
            if loss_fn is not None:
                # Handle different types of losses
                if loss_name == 'COSINE':
                    # Feature-based loss
                    if features is not None:
                        loss_value = loss_fn(features)
                    else:
                        continue
                elif loss_name == 'COSINE_MARGIN':
                    # Feature-based loss with margin
                    if features is not None:
                        loss_value = loss_fn(features, cfg=self.cfg)
                    else:
                        continue
                elif loss_name in ['FL', 'LS', 'SLMDCA']:
                    # Losses that need config parameters
                    loss_value = loss_fn(logits, label, cfg=self.cfg)
                else:
                    # Standard losses
                    loss_value = loss_fn(logits, label)
                    
                # Weight and accumulate loss
                weight = LossRegistry.get_weight(loss_name)
                losses[f'{loss_name}_loss'] = loss_value
                total_loss += weight * loss_value
                
        losses['loss'] = total_loss
        return losses

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return self.compute_losses(logits, label, text_features)

        return logits, image_features, text_features

# Hard Prompt model for full fine-tuning baseline
class HardPromptCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        
        # Define hard prompts based on what's in the config
        prompt_template = getattr(cfg.TRAINER.COOP, "PROMPT_TEMPLATE", "a photo of a {}")
            
        # Prepare text prompts
        classnames = [name.replace("_", " ") for name in classnames]
        self.prompts = [prompt_template.format(name) for name in classnames]
        
        # Tokenize prompts
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in self.prompts])
        
        # Store cfg for loss functions
        self.cfg = cfg
        # Initialize weights in LossRegistry
        LossRegistry.init_weights(cfg)
        # Get enabled losses
        self.enabled_losses = cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES
        
    def compute_losses(self, logits, label, features=None):
        """
        Compute all enabled losses and return their weighted sum
        """
        losses = {}
        total_loss = 0.0
        
        for loss_name in self.enabled_losses:
            loss_fn = LossRegistry.get_loss(loss_name)
            if loss_fn is not None:
                # Handle different types of losses
                if loss_name == 'COSINE':
                    # Feature-based loss
                    if features is not None:
                        loss_value = loss_fn(features)
                    else:
                        continue
                elif loss_name == 'COSINE_MARGIN':
                    # Feature-based loss with margin
                    if features is not None:
                        loss_value = loss_fn(features, cfg=self.cfg)
                    else:
                        continue
                elif loss_name in ['FL', 'LS', 'SLMDCA']:
                    # Losses that need config parameters
                    loss_value = loss_fn(logits, label, cfg=self.cfg)
                else:
                    # Standard losses
                    loss_value = loss_fn(logits, label)
                    
                # Weight and accumulate loss
                weight = LossRegistry.get_weight(loss_name)
                losses[f'{loss_name}_loss'] = loss_value
                total_loss += weight * loss_value
                
        losses['loss'] = total_loss
        return losses

    def forward(self, image, label=None):
        # Ensure consistent data type
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Process text features - ensure consistent dtype
        tokenized_prompts = self.tokenized_prompts.to(image.device)
        text_inputs = self.token_embedding(tokenized_prompts).type(self.dtype)
        text_features = self.text_encoder(text_inputs, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.training:
            return self.compute_losses(logits, label, text_features)
            
        return logits, image_features, text_features


@TRAINER_REGISTRY.register()
class CoOp(VLBaseLearner):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        
        # Set default for fine-tuning mode if not present
        if not hasattr(cfg.TRAINER.COOP, "FINETUNE_MODE"):
            cfg.TRAINER.COOP.FINETUNE_MODE = "prompt"  # Default: only prompt learning
            
        # Set default for prompt template if not present
        if not hasattr(cfg.TRAINER.COOP, "PROMPT_TEMPLATE"):
            cfg.TRAINER.COOP.PROMPT_TEMPLATE = "a photo of a {}"
            
        # Set default for class token position if not present
        if not hasattr(cfg.TRAINER.COOP, "CLASS_TOKEN_POSITION"):
            cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # Check fine-tuning mode
        finetune_mode = cfg.TRAINER.COOP.FINETUNE_MODE
        print(f"Building model in {finetune_mode} mode")
        
        if finetune_mode == "prompt" or finetune_mode not in ["prompt", "full_hard"]:
            # Original CoOp - only prompt learning
            print("Building CustomCLIP with learnable prompts (CoOp)")
            self.model = CustomCLIP(cfg, classnames, clip_model)
            
            print("Turning off gradients in both the image and the text encoder")
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
                    
            # Only optimize the prompt learner
            params_to_optimize = self.model.prompt_learner
            model_to_register = "prompt_learner"
            component_to_register = self.model.prompt_learner
            
        elif finetune_mode == "full_hard":
            # Full model fine-tuning with hard prompts
            print("Building HardPromptCLIP with fixed prompts and full model fine-tuning")
            self.model = HardPromptCLIP(cfg, classnames, clip_model)
            
            # All parameters are already trainable by default
            params_to_optimize = self.model
            model_to_register = "full_model"
            component_to_register = self.model

        if getattr(cfg.MODEL, 'INIT_WEIGHTS', None) and hasattr(self.model, 'prompt_learner'):
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # Build optimizer for the appropriate parameters
        self.optim = build_optimizer(params_to_optimize, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(model_to_register, component_to_register, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Multi-GPU handling
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            if finetune_mode == "prompt" or finetune_mode not in ["prompt", "full_hard"]:
                # For prompt learning, only parallelize the text encoder
                self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
            else:
                # For full fine-tuning, parallelize the whole model
                self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                losses = model(image, label)
            optim.zero_grad()
            scaler.scale(losses['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            losses = model(image, label)
            optim.zero_grad()
            losses['loss'].backward()
            optim.step()
        
        # Build loss summary to handle all enabled losses
        loss_summary = {'loss': losses['loss'].item()}
        
        # Get enabled_losses from the appropriate model type
        if hasattr(model, 'module'):
            # Handle DataParallel wrapper case
            enabled_losses = model.module.enabled_losses
        else:
            enabled_losses = model.enabled_losses
            
        for loss_name in enabled_losses:
            loss_key = f'{loss_name}_loss'
            if loss_key in losses:
                loss_summary[loss_key] = losses[loss_key].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)