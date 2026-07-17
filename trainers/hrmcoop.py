import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler

# Reuse CLIP loader and text encoder utilities from CoOp
from clip import clip
from trainers.regularizers import REGULARIZER_REGISTRY
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import convert_weights
from trainers.temp_scaling  import *
_tokenizer = _Tokenizer()
from trainers.coop import load_clip_to_cpu, TextEncoder #_tokenizer

class HRMRefiner(nn.Module):
    """
    Simple HRM-inspired hierarchical refiner:
      - z_L: per-class context tokens, shape [C, T, D]
      - z_H: per-class prototype tokens, shape [C, 1, D]

    It runs H_cycles × L_cycles updates:
      - L-level: updates z_L attending to z_H + z_L
      - H-level: updates z_H attending to z_L

    This mimics the structure of HierarchicalReasoningModel_ACTV1_Inner,
    but adapted for prompt-token refinement.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        H_cycles: int = 2,
        L_cycles: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Multi-head attention modules
        self.attn_L = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True
        )
        self.attn_H = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True
        )

        # Simple MLPs
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_L = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.mlp_H = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_L1 = nn.LayerNorm(dim)
        self.norm_L2 = nn.LayerNorm(dim)
        self.norm_H1 = nn.LayerNorm(dim)
        self.norm_H2 = nn.LayerNorm(dim)

    def forward(self, ctx: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx   : [C, T, D] initial context tokens per class
            proto : [C, 1, D] per-class visual prototype tokens

        Returns:
            refined_ctx: [C, T, D]
        """
        # Make sure inputs match the parameter dtype of the refiner
        param_dtype = self.attn_L.in_proj_weight.dtype
        ctx = ctx.to(param_dtype)
        proto = proto.to(param_dtype)

        # clone to avoid in-place messing with original params
        z_L = ctx
        z_H = proto

        C, T, D = z_L.shape

        for _ in range(self.H_cycles):
            # L-level: each class' tokens attend to its (H + L) tokens
            for _ in range(self.L_cycles):
                # query: L, key & value: H+L
                kv = torch.cat([z_H, z_L], dim=1)  # [C, 1+T, D]
                attn_out_L, _ = self.attn_L(
                    query=z_L,
                    key=kv,
                    value=kv,
                    need_weights=False,
                )
                z_L = z_L + attn_out_L
                z_L = self.norm_L1(z_L)
                z_L = z_L + self.mlp_L(z_L)
                z_L = self.norm_L2(z_L)

            # H-level: prototype attends to refined L tokens
            attn_out_H, _ = self.attn_H(
                query=z_H,
                key=z_L,
                value=z_L,
                need_weights=False,
            )
            z_H = z_H + attn_out_H
            z_H = self.norm_H1(z_H)
            z_H = z_H + self.mlp_H(z_H)
            z_H = self.norm_H2(z_H)

        return z_L
class HRMPromptLearner(nn.Module):
    """
    CoOp-style prompt learner with an HRMRefiner:
      - self.ctx is the initial generic or class-specific context
      - forward(class_prototypes) refines ctx via HRMRefiner
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.HRMCOOP.N_CTX
        ctx_init = cfg.TRAINER.HRMCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, \
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

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
            # random initialization (generic or class-specific)
            if cfg.TRAINER.HRMCOOP.CSC:
                print("Initializing class-specific contexts (HRMCoOp)")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context (HRMCoOp)")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'[HRMCoOp] Initial context: "{prompt_prefix}"')
        print(f"[HRMCoOp] Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # learnable initial context

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # SOS prefix and (class-name + EOS) suffix
        self.register_buffer("token_prefix", embedding[:, :1, :])      # (C, 1, D)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # (C, *, D)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.HRMCOOP.CLASS_TOKEN_POSITION

        # HRM-style refiner
        H_cycles = cfg.TRAINER.HRMCOOP.H_CYCLES
        L_cycles = cfg.TRAINER.HRMCOOP.L_CYCLES
        n_heads  = cfg.TRAINER.HRMCOOP.N_HEADS
        mlp_ratio = cfg.TRAINER.HRMCOOP.MLP_RATIO
        self.refiner = HRMRefiner(
                dim=ctx_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                H_cycles=H_cycles,
                L_cycles=L_cycles,
        )
        self.refiner.to(dtype=dtype)

    def forward(self, class_prototypes: torch.Tensor = None):
        """
        Args:
            class_prototypes: optional [C, D] or [C, 1, D] tensor
                              used as high-level state z_H. If None,
                              we skip HRM refinement and use raw ctx.
        """
        ctx = self.ctx

        # broadcast generic ctx to per-class if needed
        if ctx.dim() == 2:
            # [T, D] -> [C, T, D]
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        else:
            # already [C, T, D]
            pass

        # prepare prototype tokens for HRMRefiner
        if class_prototypes is not None:
            if class_prototypes.dim() == 2:
                class_prototypes = class_prototypes.unsqueeze(1)  # [C, 1, D]
            # run HRM refinement
            ctx = self.refiner(ctx, class_prototypes)

        prefix = self.token_prefix      # [C, 1, D]
        suffix = self.token_suffix      # [C, *, D]

        # Same 3 placement options as CoOp (end/middle/front)
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i  = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1,
                )
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)
        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i  = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i    = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i],
                    dim=1,
                )
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)
        else:
            raise ValueError(f"Unknown CLASS_TOKEN_POSITION={self.class_token_position}")

        return prompts
class HRMCustomCLIP(nn.Module):
    """
    CLIP wrapper that:
      - uses HRMPromptLearner for text side
      - optionally conditions prompts on per-class visual prototypes (HRM refinement)
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = HRMPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @torch.no_grad()
    def _compute_batch_prototypes(self, image_features, labels):
        """
        Build simple per-class prototypes from the current batch:
          proto[c] = mean of image_features[i] over i with label c.
        Classes not present in the batch get zeros; HRMRefiner sees them
        but gradients will mainly flow through classes that appear.
        """
        C = self.prompt_learner.n_cls
        B, D = image_features.shape
        device = image_features.device
        dtype = image_features.dtype

        protos = torch.zeros(C, D, device=device, dtype=dtype)
        counts = torch.zeros(C, device=device, dtype=torch.float32)

        for i in range(B):
            c = labels[i].item()
            protos[c] = protos[c] + image_features[i]
            counts[c] = counts[c] + 1.0

        counts = counts.clamp_min(1.0).unsqueeze(1)  # avoid divide-by-zero
        protos = protos / counts
        return protos  # [C, D]

    def forward(self, image, labels=None):
        # [B, D]
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Build HRM prototypes if labels are given (training)
        class_prototypes = None
        if labels is not None:
            class_prototypes = self._compute_batch_prototypes(image_features, labels)

        # Get refined prompts
        prompts = self.prompt_learner(class_prototypes=class_prototypes)
        tokenized_prompts = self.tokenized_prompts

        # Text encoding
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # Save some helpful fields if you want later regularizers
        self.image_features = image_features
        self.text_features = text_features
        self.logits_val = logits

        return logits
@TRAINER_REGISTRY.register()
class HRMCoOp(TrainerX):
    """
    HRM-style Context Optimization (HRMCoOp).

    - Keeps CLIP encoders frozen like CoOp.
    - Learns:
        (a) initial context tokens (prompt parameters)
        (b) HRMRefiner parameters that iteratively refine prompts
            using per-class visual prototypes in a few-shot regime.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.HRMCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"[HRMCoOp] Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # For HRMCoOp, handle precision like CoOp
        prec = cfg.TRAINER.HRMCOOP.PREC

        if prec in ["fp32", "amp"]:
            clip_model.float()          # all weights in float32
            #clip_model.dtype = torch.float32
        else:
            assert prec == "fp16"
            # convert_weights keeps LayerNorm in fp32, others in fp16
            convert_weights(clip_model)
            #clip_model.dtype = torch.float16

        # keep a consistent logical dtype
        #clip_model.dtype = clip_model.ln_final.weight.dtype
        #if cfg.TRAINER.HRMCOOP.PREC in ["fp32", "amp"]:
        #    clip_model.float()

        print("[HRMCoOp] Building HRMCustomCLIP")
        self.model = HRMCustomCLIP(cfg, classnames, clip_model)

        print("[HRMCoOp] Turning off gradients in image & text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Parameter statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        prompt_params = sum(p.numel() for p in self.model.prompt_learner.parameters())
        prompt_trainable_params = sum(
            p.numel() for p in self.model.prompt_learner.parameters()
            if p.requires_grad
        )

        print(f"[HRMCoOp][PARAM] Full model total params: {total_params/1e6:.2f} M")
        print(f"[HRMCoOp][PARAM] Trainable params (all modules): {trainable_params/1e6:.2f} M")
        print(f"[HRMCoOp][PARAM] Prompt+HRM params total: {prompt_params/1e6:.2f} M")
        print(f"[HRMCoOp][PARAM] Prompt+HRM TRAINABLE params: {prompt_trainable_params/1e6:.2f} M")

        if cfg.MODEL.INIT_WEIGHTS:
            # Only prompt_learner is trainable, so load there if needed
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # Optimizer only on prompt_learner (which includes HRMRefiner)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner",
            self.model.prompt_learner,
            self.optim,
            self.sched,
        )

        self.scaler = GradScaler() if cfg.TRAINER.HRMCOOP.PREC == "amp" else None

        # Multi-GPU
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"[HRMCoOp] Multiple GPUs detected (n_gpus={device_count}), using DataParallel")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.HRMCOOP.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, label)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        x = batch["img"]
        y = batch["label"]
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def load_model(self, directory, epoch=None):
        """
        Same logic as CoOp: load weights into prompt_learner.
        """
        if not directory:
            print("[HRMCoOp] load_model() skipped: no directory given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            epoch_ckpt = checkpoint["epoch"]

            # Ignore fixed buffers from old prompt learners if present
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f"[HRMCoOp] Loading weights to {name} from {model_path} (epoch={epoch_ckpt})")
            self._models[name].load_state_dict(state_dict, strict=False)
