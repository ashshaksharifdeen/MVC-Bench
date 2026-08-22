import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dists
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clip import clip
from clip.model import convert_weights
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.regularizers import REGULARIZER_REGISTRY
from trainers.temp_scaling  import *
_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "APTOS": "a photo of a {}.",
    "EYEPACS": "a photo of a {}.",
    "MESSIDOR": "a photo of a {}.",
    "MESSIDOR_2": "a photo of a {}.",
    "PanNuke": "a photo of a {}.",
    "KatherColon": "a photo of a {}.",
    "DigestPath": "a photo of a {}.",
    "RSNA18": "a photo of a {}.",
    "Covid": "a photo of a {}.",
}

def angular_distance_deg(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns angular distance (degrees) for the last dim.
    a, b: (..., D)
    """
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    cos = (a * b).sum(dim=-1).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos) * (180.0 / math.pi)

def load_clip_to_cpu_zs(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class ZeroshotCLIP():
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu_zs(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        
        self.positional_embedding = clip_model.positional_embedding #shape of positional_embedding: torch.Size([77, 512])
        
        self.ln_final = clip_model.ln_final
       
        self.text_projection = clip_model.text_projection #shape of text_projection: torch.Size([512, 512])
        
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
    
    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype) #shape of prompts + positional embedding: torch.Size([24, 77, 512])
        
        x = x.permute(1, 0, 2)  # NLD -> LND #shape after permute: torch.Size([77, 24, 512])
        
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        
        outputs = self.transformer(combined)
       
        x = outputs[0]  # extract the x back from here #shape of x ( outputs[0]): torch.Size([77, 24, 512])
        
        x = x.permute(1, 0, 2)  # LND -> NLD #shape of x ( x.permute(1, 0, 2)): torch.Size([24, 77, 512])
        
        x = self.ln_final(x).type(self.dtype) # torch.Size([24, 77, 512])
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection #shape of x at eot embedding: torch.Size([24, 512])
        
        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        #print("clip model ln_final.weight:",clip_model.ln_final.weight.shape) #clip model ln_final.weight: torch.Size([512])
        #print("clip model ctx_dim:",clip_model.ln_final.weight.shape[0]) #clip model ctx_dim: 512
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        #self.clip_model = clip_model
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) #prompt embedding dimension: torch.Size([1, 77, 512])
                
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :] #ctx_vectors dimension: torch.Size([2, 512])
            
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        #learnable shallow text prompts parameter
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        # deep prompts injected into internal layers (which internal layer ?) of the text transformer.
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)]) ## 2,512 is learnable matrix with the depth of 8

        
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        #Projects each text prompt into vision prompt space.
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1) #512,768 is learnable matrix with the depth of 8

        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn) #complete tokenized prompt shape: torch.Size([24, 77])
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) #complete prompt embedding shape: torch.Size([24, 77, 512])
             
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS token prefix shape: torch.Size([24, 1, 512]
        
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS token suffix shape: torch.Size([24, 74, 512]) other ([24, 2, 512]) will be learnable vector and it can be incoperate in prompt depth 
        

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        #Text deep prompts are transformed by F_k to produce vision deep prompts.
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        #print("projected dimension:", self.proj(self.ctx).shape)
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        """
        image_features shape: torch.Size([4, 512])
        text_features shape: torch.Size([51, 512])
        logits shape: torch.Size([4, 51])
        labels shape: torch.Size([4])
        """
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        self.sematic_val = image_features @ text_features.t()
        self.imfeatures = image_features
        self.textfeatures = text_features
        self.output_ = torch.softmax(logits, dim=1)
        #temperaturescaling
        #logits= logits/1.16
        
        self.logits_val = logits


        if self.prompt_learner.training:
                #Apply temperature scalliong  >>> temperature_value = {'ViT': 1.16, 'RN': 1.15} learned temperature value
                """mask = (
                torch.arange(logits.size(1), device=logits.device)
                .unsqueeze(0)
                .expand(logits.size(0), -1)
                != label.unsqueeze(1)
                )
                # divide only those positions by T
                logits = torch.where(mask, logits / 1.16, logits)"""
            
                return F.cross_entropy(logits, label)

        #temp_calibrator =TempScaling(bias=False)
        #temp_calibrator.fit(logits, label)
        #logits = temp_calibrator.calibrate(logits)
        return logits
    
    @torch.no_grad()
    def infer_intermediates(self, image):
        """
        Returns:
          effective_depths: list[int]
          logits_per_depth: list[Tensor[B, num_classes]]
        """
        prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()

        # text features once (full depth)
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # image features per depth
        feats_per_block, effective_depths = self.image_encoder.forward_intermediates(
            image.type(self.dtype), shared_ctx, deep_vision
        )

        logit_scale = self.logit_scale.exp()
        logits_per_depth = []
        for f in feats_per_block:
            f = f / f.norm(dim=-1, keepdim=True)
            logits = logit_scale * (f @ text_features.t())
            logits_per_depth.append(logits)

        return effective_depths, logits_per_depth

    
    def forward_features(self, image):
        """
        Returns the intermediate, unpooled features from the vision encoder.
        This method calls forward_features on the image encoder (if available) to obtain a 
        tensor of shape [batch, tokens, hidden_dim] that contains spatial information.
        """
        tokenized_prompts = self.tokenized_prompts
        _, shared_ctx, _, deep_compound_prompts_vision = self.prompt_learner()
        # Assume the image encoder has a method forward_features.
        if hasattr(self.image_encoder, "forward_features"):
            features = self.image_encoder.forward_features(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        else:
            # Fallback: use the standard forward and then unsqueeze to mimic spatial dims.
            features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
            features = features.unsqueeze(1)  # [B, hidden_dim] -> [B, hidden_dim, 1]
        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _append_csv_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _grad_norm(loss: torch.Tensor, params, retain_graph: bool = True) -> torch.Tensor:
    """
    Compute ||∇_params loss||_2 as a scalar tensor.
    Use allow_unused=True because some params may not affect a given term.
    """
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, create_graph=False, allow_unused=True
    )
    total = 0.0
    for g in grads:
        if g is None:
            continue
        total += (g.detach().float() ** 2).sum()
    return total.sqrt()

def _write_csv(path, header, rows):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        # --- Parameter counts for the CLIP backbone ---
        clip_total_params = sum(p.numel() for p in clip_model.parameters())
        clip_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)

        print(f"[PARAM] CLIP backbone total params: {clip_total_params/1e6:.2f} M")
        print(f"[PARAM] CLIP backbone trainable params (before freezing): "
              f"{clip_trainable_params/1e6:.2f} M")

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        # --- Parameter counts for the full MaPLe model ---
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)

        # Prompt-learner params (this is what you want for CVPR: "trainable params")
        prompt_params = sum(p.numel() for p in self.model.prompt_learner.parameters())
        prompt_trainable_params = sum(
            p.numel() for p in self.model.prompt_learner.parameters()
            if p.requires_grad
        )

        print(f"[PARAM] Full model (CLIP + prompts) total params: {total_params/1e6:.2f} M")
        print(f"[PARAM] Trainable params (all modules): {trainable_params/1e6:.2f} M")
        print(f"[PARAM] Prompt-learner params total: {prompt_params/1e6:.2f} M")
        print(f"[PARAM] Prompt-learner TRAINABLE params: {prompt_trainable_params/1e6:.2f} M")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # --- zero-shot CLIP hook ---
        self.zeroshot = ZeroshotCLIP()
        self.zeroshot.cfg = cfg
        self.zeroshot.dm = self.dm
        self.zeroshot.device = self.device
        self.zeroshot.build_model()

        # ---- scale logging state ----
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self._scale_step = 0
        self._scale_log_path = osp.join(cfg.OUTPUT_DIR, cfg.TRAINER.MAPLE.SCALE_LOG_FILE)
                
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel

        device_count = torch.cuda.device_count()
        """if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)"""

    def _get_model_unwrapped(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _reset_prompt_gradnorm_epoch(self):
        # sums for: [shallow_ctx] + [deep_prompt_0 ... deep_prompt_{D-2}]
        m = self._get_model_unwrapped()
        pl = m.prompt_learner
        n_deep = len(pl.compound_prompts_text)
        self._grad_sum = np.zeros(1 + n_deep, dtype=np.float64)
        self._grad_count = 0

    def _accumulate_prompt_gradnorm(self):
        """
        Accumulate grad L2 norms *per mini-batch*.
        Index meaning:
          i=0   -> shallow ctx (pl.ctx)
          i>=1  -> deep prompt (pl.compound_prompts_text[i-1])
        """
        m = self._get_model_unwrapped()
        pl = m.prompt_learner

        vals = []

        # i=0: shallow ctx
        if pl.ctx.grad is None:
            vals.append(0.0)
        else:
            vals.append(pl.ctx.grad.detach().float().norm(p=2).item())

        # i>=1: deep prompts (text-side params; vision prompts are derived from these)
        for p in pl.compound_prompts_text:
            if p.grad is None:
                vals.append(0.0)
            else:
                vals.append(p.grad.detach().float().norm(p=2).item())

        self._grad_sum += np.array(vals, dtype=np.float64)
        self._grad_count += 1

    def _finalize_prompt_gradnorm_epoch(self):
        if getattr(self, "_grad_count", 0) == 0:
            return None
        return (self._grad_sum / self._grad_count).tolist()

    def plot_effective_depth_vs_accuracy(self, loader, out_path, max_batches=50):
        """
        max_batches = number of mini-batches from loader used for the plot (speed control),
        NOT your batch size.
        """
        m = self._get_model_unwrapped()
        was_train = m.training
        m.eval()

        correct = None
        total = 0
        eff_depths = None

        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break

                if hasattr(self, "parse_batch_test"):
                    image, label = self.parse_batch_test(batch)
                else:
                    image, label = self.parse_batch_train(batch)

                eff_depths, logits_list = m.infer_intermediates(image)

                if correct is None:
                    correct = np.zeros(len(logits_list), dtype=np.int64)

                for k, logits in enumerate(logits_list):
                    pred = logits.argmax(dim=1)
                    correct[k] += (pred == label).sum().item()

                total += label.size(0)

        acc = (correct / max(total, 1)).tolist()

        plt.figure(figsize=(10, 7), dpi=200)
        plt.plot(eff_depths, acc, marker="o")
        plt.title("Effective depth vs Accuracy")
        plt.xlabel("Effective layers computed (ViT blocks)")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        if was_train:
            m.train()

        return eff_depths, acc

    def plot_effective_depth_vs_gradnorm(self, effective_depths, avg_grad_norms, out_path):
        """
        Map prompt index -> depth:
          i=0 (shallow) at x=0
          i=1 (deep0) uses block2 => map to effective_depths[1]
          i=2 (deep1) uses block3 => map to effective_depths[2]
          ...
        """
        if avg_grad_norms is None:
            return

        # x positions
        xs = [0]
        for i in range(1, len(avg_grad_norms)):
            idx = min(i, len(effective_depths) - 1)  # clamp
            xs.append(effective_depths[idx])

        plt.figure(figsize=(10, 7), dpi=200)
        plt.plot(xs, avg_grad_norms, marker="o")
        plt.title("Effective depth vs Prompt gradient norm")
        plt.xlabel("Effective layers computed (mapped to prompt injection depth)")
        plt.ylabel("Avg gradient L2 norm (epoch)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    @torch.no_grad()
    def model_inference_sals(self, image: torch.Tensor) -> torch.Tensor:
        """
        SaLS inference-time post-processing:
        logits_sals = eccv_sals(zs_logits, adapted_logits)
        """
        self.model.eval()

        adapted_logits = self.model(image)  # eval() => returns logits :contentReference[oaicite:4]{index=4}
        zs_logits = self.zeroshot.model_inference(image)  # :contentReference[oaicite:5]{index=5}

        sals_fn = REGULARIZER_REGISTRY.get("eccv_sals")  # :contentReference[oaicite:6]{index=6}
        logits_sals = sals_fn(zs_pred=zs_logits, output=adapted_logits)  # :contentReference[oaicite:7]{index=7}

        return logits_sals

    
    def forward_backward(self, batch):
        if getattr(self, "batch_idx", 0) == 0:
            self._reset_prompt_gradnorm_epoch()
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC

        #below you can intergrate loss or regularization function
        if prec == "amp":
            with autocast():
                loss = model(image, label)
                 

            optim.zero_grad()
            scaler.scale(loss).backward()
            # unscale so grad norms are meaningful
            scaler.unscale_(optim)
            self._accumulate_prompt_gradnorm()
            scaler.step(optim)
            scaler.update()
           
        else:

            loss = model(image, label)
            
            #CAP loss
            image_features = model.imfeatures
            text_features = model.textfeatures
            C = text_features.shape[0]
            #cross model zs alignment-----------
            # grab MaPLe features   
            #end---------------------------
            ##Inter class equvarience loss -begin------------------------------------------------
            logits = model.logits_val  #model.sematic_val  # shape (B, C)
            labels = label               # shape (B,)
            loss_ce = loss
            # fetch our helper from the registry
            #MARGIN_MEAN_VAR_ALLCLASS_EXPLICIT
            explicit_all = REGULARIZER_REGISTRY.get("margin_mean_var_allclass_loss_explicit")
            explicit_all_loss =explicit_all(logits,label,variance_mode="all_pairs")
            loss+=explicit_all_loss 

            optim.zero_grad()
            #loss_total.backward()
            #eccv_zs_loss.backward()
            loss.backward()
            self._accumulate_prompt_gradnorm()
            optim.step()
            #loss = loss_total
            self._scale_step += 1

        # ---- plot once per epoch (end of last batch) ----
        num_batches = getattr(self, "num_batches", None)
        if num_batches is None:
                # fallback
                num_batches = len(getattr(self, "train_loader_x", getattr(self, "train_loader", [])))

        is_last = (getattr(self, "batch_idx", 0) + 1) == num_batches
        if is_last:
                avg_g = self._finalize_prompt_gradnorm_epoch()

                # choose a loader for evaluation
                loader = getattr(self, "val_loader", None)
                if loader is None:
                    loader = getattr(self, "test_loader", None)

                if loader is not None:
                    outdir = self.cfg.OUTPUT_DIR
                    dname = getattr(self.cfg.DATASET, "NAME", "dataset")
                    seed = getattr(self.cfg, "SEED", 0)

                    acc_path = osp.join(outdir, f"{dname}_MaPLe_effdepth_vs_accuracy_seed{seed}.png")
                    grad_path = osp.join(outdir, f"{dname}_MaPLe_effdepth_vs_gradnorm_seed{seed}.png")

                    effective_depths, _ = self.plot_effective_depth_vs_accuracy(loader, acc_path, max_batches=50)
                    self.plot_effective_depth_vs_gradnorm(effective_depths, avg_g, grad_path)

        #loss_summary = {"loss": eccv_zs_loss.item()}
        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def _safe_forward_intermediates(self, image_encoder, image, shared_ctx, deep_vision, dtype):
        """
        Some backbones return (feats, eff_depths),
        others return (feats, halt_logits, eff_depths).
        We support both to avoid unpacking crashes.
        """
        out = image_encoder.forward_intermediates(image.type(dtype), shared_ctx, deep_vision)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            feats_per_block, _, eff_depths = out
        else:
            feats_per_block, eff_depths = out
        return feats_per_block, eff_depths

    @torch.no_grad()
    def _collect_vision_cross_dynamics(self, loader, max_batches=50):
        """
        Collects, over the loader:
          - vision angdist between consecutive vision features
          - cross angdist between consecutive prediction distributions (softmax probs)
          - layerwise accuracy, confidence curves
          - transition stats (prediction-change rate, flip-to-correct, flip-to-incorrect)
        """
        m = self._get_model_unwrapped()
        was_train = m.training
        m.eval()

        total = 0
        eff_depths_ref = None

        # Will be allocated after first batch (once we know K)
        K = None
        vis_ang_sum = None
        cross_ang_sum = None
        trans_cnt = None

        correct_sum = None
        top1_conf_sum = None
        gt_conf_sum = None
        top1_conf_correct_sum = None
        top1_conf_incorrect_sum = None
        n_correct = None
        n_incorrect = None

        pred_change_sum = None
        flip_to_correct_sum = None
        flip_to_incorrect_sum = None

        for bi, batch in enumerate(loader):
            if max_batches is not None and max_batches > 0 and bi >= max_batches:
                break

            if hasattr(self, "parse_batch_test"):
                image, label = self.parse_batch_test(batch)
            else:
                image, label = self.parse_batch_train(batch)

            # Build prompts once per batch (MaPLe learned ctx)
            prompts, shared_ctx, deep_text, deep_vision = m.prompt_learner()

            # Text features (final)
            text_features = m.text_encoder(prompts, m.tokenized_prompts, deep_text)
            text_features = F.normalize(text_features.float(), dim=-1)

            # Vision features per block
            feats_per_block, eff_depths = self._safe_forward_intermediates(
                m.image_encoder, image, shared_ctx, deep_vision, m.dtype
            )

            # logits/probs per depth
            logit_scale = m.logit_scale.exp()
            logits_list = []
            probs_list = []
            for f in feats_per_block:
                f = F.normalize(f.float(), dim=-1)
                logits = logit_scale * (f @ text_features.t())
                probs = logits.softmax(dim=1)
                logits_list.append(logits)
                probs_list.append(probs)

            if eff_depths_ref is None:
                eff_depths_ref = list(eff_depths)
                K = len(logits_list)

                vis_ang_sum = np.zeros(K - 1, dtype=np.float64)
                cross_ang_sum = np.zeros(K - 1, dtype=np.float64)
                trans_cnt = np.zeros(K - 1, dtype=np.float64)

                correct_sum = np.zeros(K, dtype=np.float64)
                top1_conf_sum = np.zeros(K, dtype=np.float64)
                gt_conf_sum = np.zeros(K, dtype=np.float64)
                top1_conf_correct_sum = np.zeros(K, dtype=np.float64)
                top1_conf_incorrect_sum = np.zeros(K, dtype=np.float64)
                n_correct = np.zeros(K, dtype=np.float64)
                n_incorrect = np.zeros(K, dtype=np.float64)

                pred_change_sum = np.zeros(K - 1, dtype=np.float64)
                flip_to_correct_sum = np.zeros(K - 1, dtype=np.float64)
                flip_to_incorrect_sum = np.zeros(K - 1, dtype=np.float64)

            B = label.size(0)
            total += B

            preds = []
            correct_masks = []

            # ---- depthwise stats ----
            for k in range(K):
                probs = probs_list[k]
                pred = probs.argmax(dim=1)
                top1_conf = probs.max(dim=1).values
                gt_conf = probs[torch.arange(B, device=label.device), label]

                correct = (pred == label)
                preds.append(pred)
                correct_masks.append(correct)

                correct_sum[k] += float(correct.sum().item())
                top1_conf_sum[k] += float(top1_conf.sum().item())
                gt_conf_sum[k] += float(gt_conf.sum().item())

                if correct.any():
                    top1_conf_correct_sum[k] += float(top1_conf[correct].sum().item())
                    n_correct[k] += float(correct.sum().item())
                if (~correct).any():
                    top1_conf_incorrect_sum[k] += float(top1_conf[~correct].sum().item())
                    n_incorrect[k] += float((~correct).sum().item())

            # ---- transition stats + angdist ----
            # vision angdist: between consecutive vision features
            # cross angdist : between consecutive prob vectors
            for k in range(K - 1):
                # vision ang
                a_v = feats_per_block[k].float()
                b_v = feats_per_block[k + 1].float()
                ang_v = angular_distance_deg(a_v, b_v)  # (B,)
                vis_ang_sum[k] += float(ang_v.sum().item())

                # cross ang (distribution refinement)
                a_p = probs_list[k].float()
                b_p = probs_list[k + 1].float()
                ang_p = angular_distance_deg(a_p, b_p)  # (B,)
                cross_ang_sum[k] += float(ang_p.sum().item())

                trans_cnt[k] += float(B)

                # prediction changes
                p0 = preds[k]
                p1 = preds[k + 1]
                c0 = correct_masks[k]
                c1 = correct_masks[k + 1]

                pred_change_sum[k] += float((p0 != p1).sum().item())
                flip_to_correct_sum[k] += float(((~c0) & (c1)).sum().item())
                flip_to_incorrect_sum[k] += float(((c0) & (~c1)).sum().item())

        # ---- finalize ----
        if total == 0:
            return None

        eff_depths_ref = np.asarray(eff_depths_ref, dtype=float)

        vis_ang = vis_ang_sum / np.maximum(trans_cnt, 1.0)
        cross_ang = cross_ang_sum / np.maximum(trans_cnt, 1.0)

        acc = correct_sum / max(total, 1)
        top1_conf = top1_conf_sum / max(total, 1)
        gt_conf = gt_conf_sum / max(total, 1)

        top1_conf_correct = top1_conf_correct_sum / np.maximum(n_correct, 1.0)
        top1_conf_incorrect = top1_conf_incorrect_sum / np.maximum(n_incorrect, 1.0)

        pred_change_rate = pred_change_sum / np.maximum(trans_cnt, 1.0)
        flip_to_correct_rate = flip_to_correct_sum / np.maximum(trans_cnt, 1.0)
        flip_to_incorrect_rate = flip_to_incorrect_sum / np.maximum(trans_cnt, 1.0)

        if was_train:
            m.train()

        return dict(
            eff_depths=eff_depths_ref,
            vis_ang=vis_ang,
            cross_ang=cross_ang,
            acc=acc,
            top1_conf=top1_conf,
            gt_conf=gt_conf,
            top1_conf_correct=top1_conf_correct,
            top1_conf_incorrect=top1_conf_incorrect,
            pred_change_rate=pred_change_rate,
            flip_to_correct_rate=flip_to_correct_rate,
            flip_to_incorrect_rate=flip_to_incorrect_rate,
        )

    @torch.no_grad()
    def _collect_text_dynamics(self, loader, max_batches=50):
        """
        Text-only dynamics:
          - text angdist between consecutive text blocks (class prototypes)
          - prediction/confidence evolution across text blocks
            (using FINAL vision features; swapping text prototypes per layer)
        """
        m = self._get_model_unwrapped()
        was_train = m.training
        m.eval()

        # Use current dataset classnames (base/new already handled by your subsampling)
        ds = self.dm.dataset
        classnames = getattr(ds, "classnames", None)
        if classnames is None:
            raise AttributeError("Dataset has no 'classnames' attribute")

        # Build prompts + tokenized + deep_text + shared_ctx + deep_vision from learned ctx
        prompts, tokenized, deep_text, shared_ctx, deep_vision = self._build_prompts_from_classnames(classnames)

        # Collect per-layer text EOT features using your hook-based utility
        # (list length L; each is [C, Dproj]) :contentReference[oaicite:4]{index=4}
        text_feats_per_layer_cpu = self._collect_text_eot_per_layer(prompts, tokenized, deep_text)

        # Move to device and normalize
        text_feats_per_layer = [F.normalize(t.to(self.device).float(), dim=-1) for t in text_feats_per_layer_cpu]
        L = len(text_feats_per_layer)

        # text angdist transitions
        text_ang = np.zeros(L - 1, dtype=np.float64)
        for i in range(L - 1):
            ang = angular_distance_deg(text_feats_per_layer[i], text_feats_per_layer[i + 1])  # (C,)
            text_ang[i] = float(ang.mean().item())

        # stats arrays
        total = 0
        correct_sum = np.zeros(L, dtype=np.float64)
        top1_conf_sum = np.zeros(L, dtype=np.float64)
        gt_conf_sum = np.zeros(L, dtype=np.float64)
        top1_conf_correct_sum = np.zeros(L, dtype=np.float64)
        top1_conf_incorrect_sum = np.zeros(L, dtype=np.float64)
        n_correct = np.zeros(L, dtype=np.float64)
        n_incorrect = np.zeros(L, dtype=np.float64)

        pred_change_sum = np.zeros(L - 1, dtype=np.float64)
        flip_to_correct_sum = np.zeros(L - 1, dtype=np.float64)
        flip_to_incorrect_sum = np.zeros(L - 1, dtype=np.float64)
        trans_cnt = np.zeros(L - 1, dtype=np.float64)

        logit_scale = m.logit_scale.exp()

        for bi, batch in enumerate(loader):
            if max_batches is not None and max_batches > 0 and bi >= max_batches:
                break

            if hasattr(self, "parse_batch_test"):
                image, label = self.parse_batch_test(batch)
            else:
                image, label = self.parse_batch_train(batch)

            # FINAL vision features (single pass)
            img_feat = m.image_encoder(image.type(m.dtype), shared_ctx, deep_vision)
            img_feat = F.normalize(img_feat.float(), dim=-1)

            B = label.size(0)
            total += B

            preds = []
            correct_masks = []

            for l in range(L):
                tf = text_feats_per_layer[l]  # [C, D]
                logits = logit_scale * (img_feat @ tf.t())
                probs = logits.softmax(dim=1)

                pred = probs.argmax(dim=1)
                top1_conf = probs.max(dim=1).values
                gt_conf = probs[torch.arange(B, device=label.device), label]

                correct = (pred == label)
                preds.append(pred)
                correct_masks.append(correct)

                correct_sum[l] += float(correct.sum().item())
                top1_conf_sum[l] += float(top1_conf.sum().item())
                gt_conf_sum[l] += float(gt_conf.sum().item())

                if correct.any():
                    top1_conf_correct_sum[l] += float(top1_conf[correct].sum().item())
                    n_correct[l] += float(correct.sum().item())
                if (~correct).any():
                    top1_conf_incorrect_sum[l] += float(top1_conf[~correct].sum().item())
                    n_incorrect[l] += float((~correct).sum().item())

            for l in range(L - 1):
                p0 = preds[l]
                p1 = preds[l + 1]
                c0 = correct_masks[l]
                c1 = correct_masks[l + 1]

                pred_change_sum[l] += float((p0 != p1).sum().item())
                flip_to_correct_sum[l] += float(((~c0) & (c1)).sum().item())
                flip_to_incorrect_sum[l] += float(((c0) & (~c1)).sum().item())
                trans_cnt[l] += float(B)

        if total == 0:
            return None

        acc = correct_sum / max(total, 1)
        top1_conf = top1_conf_sum / max(total, 1)
        gt_conf = gt_conf_sum / max(total, 1)
        top1_conf_correct = top1_conf_correct_sum / np.maximum(n_correct, 1.0)
        top1_conf_incorrect = top1_conf_incorrect_sum / np.maximum(n_incorrect, 1.0)

        pred_change_rate = pred_change_sum / np.maximum(trans_cnt, 1.0)
        flip_to_correct_rate = flip_to_correct_sum / np.maximum(trans_cnt, 1.0)
        flip_to_incorrect_rate = flip_to_incorrect_sum / np.maximum(trans_cnt, 1.0)

        if was_train:
            m.train()

        # x-axis as text-block index: 1..L (paper friendly)
        text_depths = np.arange(1, L + 1, dtype=float)

        return dict(
            text_depths=text_depths,
            text_ang=text_ang,
            acc=acc,
            top1_conf=top1_conf,
            gt_conf=gt_conf,
            top1_conf_correct=top1_conf_correct,
            top1_conf_incorrect=top1_conf_incorrect,
            pred_change_rate=pred_change_rate,
            flip_to_correct_rate=flip_to_correct_rate,
            flip_to_incorrect_rate=flip_to_incorrect_rate,
        )

    def _plot_dynamics_figure(self, x_depth, x_trans, y_ang, acc, pred_change_rate,
                             flip_to_correct_rate, flip_to_incorrect_rate,
                             top1_conf, gt_conf, top1_conf_correct, top1_conf_incorrect,
                             title, save_path):
        """
        3-panel figure:
          (1) angular distance between consecutive layers
          (2) accuracy + prediction flip dynamics
          (3) confidence refinement (top1 + gt + correct/incorrect split)
        """
        plt.rcParams["font.family"] = "serif"
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), dpi=300, sharex=True)

        # (1) Angular distance
        axes[0].plot(x_trans, y_ang, marker="o", linewidth=2)
        axes[0].set_ylabel("Avg angular dist (deg)")
        axes[0].set_title(title)
        axes[0].grid(alpha=0.25, linestyle="--")

        # annotate final-layer impact on angdist
        if len(x_trans) >= 1:
            axes[0].annotate(
                f"Final jump: {y_ang[-1]:.2f}°",
                xy=(x_trans[-1], y_ang[-1]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
            )

        # (2) Accuracy + flips
        axes[1].plot(x_depth, acc, marker="o", linewidth=2, label="Acc (GT correct)")
        axes[1].plot(x_trans, pred_change_rate, marker="x", linewidth=1.5, label="Pred change rate")
        axes[1].plot(x_trans, flip_to_correct_rate, marker="^", linewidth=1.5, label="Flip → correct")
        axes[1].plot(x_trans, flip_to_incorrect_rate, marker="v", linewidth=1.5, label="Flip → incorrect")
        axes[1].set_ylabel("Rate")
        axes[1].grid(alpha=0.25, linestyle="--")
        axes[1].legend(framealpha=0.9, fontsize=9, loc="best")

        # annotate final-layer delta acc
        if len(acc) >= 2:
            dacc = float(acc[-1] - acc[-2])
            axes[1].annotate(
                f"ΔAcc (final): {dacc:+.3f}",
                xy=(x_depth[-1], acc[-1]),
                xytext=(10, -15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
            )

        # (3) Confidence refinement
        axes[2].plot(x_depth, top1_conf, marker="o", linewidth=2, label="Top-1 conf (all)")
        axes[2].plot(x_depth, gt_conf, marker="s", linewidth=2, label="GT-class conf")
        axes[2].plot(x_depth, top1_conf_correct, linestyle="--", linewidth=2, label="Top-1 conf | correct")
        axes[2].plot(x_depth, top1_conf_incorrect, linestyle="--", linewidth=2, label="Top-1 conf | incorrect")
        axes[2].set_ylabel("Confidence")
        axes[2].set_xlabel("Layers passed (effective depth / block index)")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].grid(alpha=0.25, linestyle="--")
        axes[2].legend(framealpha=0.9, fontsize=9, loc="best")

        # annotate final-layer delta confidence (GT)
        if len(gt_conf) >= 2:
            dgt = float(gt_conf[-1] - gt_conf[-2])
            axes[2].annotate(
                f"ΔGT-conf (final): {dgt:+.3f}",
                xy=(x_depth[-1], gt_conf[-1]),
                xytext=(10, -15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
            )

        plt.tight_layout()
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def _plot_layer_dynamics_all_three(self, loader, outdir, split_name="all"):
        """
        Saves (in outdir/layer_dynamics):
          - {split}_vision_dynamics.png (+csvs)
          - {split}_cross_dynamics.png  (+csvs)
          - {split}_text_dynamics.png   (+csvs)
        """
        save_dir = osp.join(outdir, "layer_dynamics")
        os.makedirs(save_dir, exist_ok=True)

        max_batches = int(getattr(self.cfg.TRAINER.MAPLE, "ANGDIST_MAX_BATCHES", 50))

        # ---- vision + cross (single collection pass) ----
        vc = self._collect_vision_cross_dynamics(loader, max_batches=max_batches)
        if vc is not None:
            eff = vc["eff_depths"]
            x_depth = eff
            x_trans = eff[:-1]

            # Vision figure
            self._plot_dynamics_figure(
                x_depth=x_depth,
                x_trans=x_trans,
                y_ang=vc["vis_ang"],
                acc=vc["acc"],
                pred_change_rate=vc["pred_change_rate"],
                flip_to_correct_rate=vc["flip_to_correct_rate"],
                flip_to_incorrect_rate=vc["flip_to_incorrect_rate"],
                top1_conf=vc["top1_conf"],
                gt_conf=vc["gt_conf"],
                top1_conf_correct=vc["top1_conf_correct"],
                top1_conf_incorrect=vc["top1_conf_incorrect"],
                title=f"Vision encoder dynamics (avg angdist + prediction refinement) | split={split_name}",
                save_path=osp.join(save_dir, f"{split_name}_vision_dynamics.png"),
            )

            # Cross figure (distribution refinement)
            self._plot_dynamics_figure(
                x_depth=x_depth,
                x_trans=x_trans,
                y_ang=vc["cross_ang"],
                acc=vc["acc"],
                pred_change_rate=vc["pred_change_rate"],
                flip_to_correct_rate=vc["flip_to_correct_rate"],
                flip_to_incorrect_rate=vc["flip_to_incorrect_rate"],
                top1_conf=vc["top1_conf"],
                gt_conf=vc["gt_conf"],
                top1_conf_correct=vc["top1_conf_correct"],
                top1_conf_incorrect=vc["top1_conf_incorrect"],
                title=f"Vision×Text (cross) dynamics: angdist between successive prediction distributions | split={split_name}",
                save_path=osp.join(save_dir, f"{split_name}_cross_dynamics.png"),
            )

            # CSVs (vision/cross)
            depth_rows = []
            for i in range(len(eff)):
                depth_rows.append([
                    split_name, float(eff[i]),
                    float(vc["acc"][i]),
                    float(vc["top1_conf"][i]),
                    float(vc["gt_conf"][i]),
                    float(vc["top1_conf_correct"][i]),
                    float(vc["top1_conf_incorrect"][i]),
                ])
            _write_csv(
                osp.join(save_dir, f"{split_name}_vision_cross_depthwise.csv"),
                ["split", "eff_depth", "acc", "top1_conf", "gt_conf", "top1_conf_correct", "top1_conf_incorrect"],
                depth_rows
            )

            trans_rows = []
            for i in range(len(eff) - 1):
                trans_rows.append([
                    split_name, float(eff[i]), float(eff[i + 1]),
                    float(vc["vis_ang"][i]),
                    float(vc["cross_ang"][i]),
                    float(vc["pred_change_rate"][i]),
                    float(vc["flip_to_correct_rate"][i]),
                    float(vc["flip_to_incorrect_rate"][i]),
                ])
            _write_csv(
                osp.join(save_dir, f"{split_name}_vision_cross_transitions.csv"),
                ["split", "from_depth", "to_depth", "vision_ang_deg", "cross_ang_deg",
                 "pred_change_rate", "flip_to_correct_rate", "flip_to_incorrect_rate"],
                trans_rows
            )

        # ---- text (separate pass) ----
        td = self._collect_text_dynamics(loader, max_batches=max_batches)
        if td is not None:
            x_depth = td["text_depths"]
            x_trans = td["text_depths"][:-1]

            self._plot_dynamics_figure(
                x_depth=x_depth,
                x_trans=x_trans,
                y_ang=td["text_ang"],
                acc=td["acc"],
                pred_change_rate=td["pred_change_rate"],
                flip_to_correct_rate=td["flip_to_correct_rate"],
                flip_to_incorrect_rate=td["flip_to_incorrect_rate"],
                top1_conf=td["top1_conf"],
                gt_conf=td["gt_conf"],
                top1_conf_correct=td["top1_conf_correct"],
                top1_conf_incorrect=td["top1_conf_incorrect"],
                title=f"Text encoder dynamics (avg angdist + prediction refinement) | split={split_name}",
                save_path=osp.join(save_dir, f"{split_name}_text_dynamics.png"),
            )

            depth_rows = []
            for i in range(len(x_depth)):
                depth_rows.append([
                    split_name, float(x_depth[i]),
                    float(td["acc"][i]),
                    float(td["top1_conf"][i]),
                    float(td["gt_conf"][i]),
                    float(td["top1_conf_correct"][i]),
                    float(td["top1_conf_incorrect"][i]),
                ])
            _write_csv(
                osp.join(save_dir, f"{split_name}_text_depthwise.csv"),
                ["split", "text_block", "acc", "top1_conf", "gt_conf", "top1_conf_correct", "top1_conf_incorrect"],
                depth_rows
            )

            trans_rows = []
            for i in range(len(x_depth) - 1):
                trans_rows.append([
                    split_name, float(x_depth[i]), float(x_depth[i + 1]),
                    float(td["text_ang"][i]),
                    float(td["pred_change_rate"][i]),
                    float(td["flip_to_correct_rate"][i]),
                    float(td["flip_to_incorrect_rate"][i]),
                ])
            _write_csv(
                osp.join(save_dir, f"{split_name}_text_transitions.csv"),
                ["split", "from_block", "to_block", "text_ang_deg",
                 "pred_change_rate", "flip_to_correct_rate", "flip_to_incorrect_rate"],
                trans_rows
            )



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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def test(self, *args, **kwargs):
        out = super().test(*args, **kwargs)

        if not bool(getattr(self.cfg.TRAINER.MAPLE, "PLOT_ANGDIST", False)):
            return out

        outdir = self.cfg.OUTPUT_DIR
        split_name = str(getattr(self.cfg.DATASET, "SUBSAMPLE_CLASSES", "all"))

        # existing plots
        self._plot_angdist_text(outdir, split_name=split_name)

        loader = getattr(self, "test_loader", None) or getattr(self, "val_loader", None)
        if loader is not None:
            self._plot_angdist_vision(loader, outdir, split_name=split_name)

            # NEW: 3 NeurIPS-ready layer-dynamics figures
            self._plot_layer_dynamics_all_three(loader, outdir, split_name=split_name)

        return out

    def _get_model_unwrapped(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    @torch.no_grad()
    def _build_prompts_from_classnames(self, classnames):
        """
        Build (prompts, tokenized_prompts, deep_text_prompts, shared_ctx, deep_vision_prompts)
        using the *learned* ctx from prompt_learner, but tokenizing the provided classnames.
        """
        m = self._get_model_unwrapped()
        pl = m.prompt_learner
        cfg = self.cfg

        # optional cap for speed
        max_classes = int(getattr(cfg.TRAINER.MAPLE, "ANGDIST_MAX_CLASSES", 0))
        if max_classes > 0 and len(classnames) > max_classes:
            classnames = list(classnames)[:max_classes]

        n_ctx = pl.n_ctx
        ctx_init = getattr(cfg.TRAINER.MAPLE, "CTX_INIT", "")
        use_ctx_init = bool(ctx_init) and (n_ctx <= 4)
        if use_ctx_init:
            prompt_prefix = str(ctx_init).replace("_", " ")
        else:
            prompt_prefix = " ".join(["X"] * n_ctx)

        names = [c.replace("_", " ") for c in classnames]
        prompts_str = [prompt_prefix + " " + name + "." for name in names]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts_str]).to(self.device)

        # token embeddings from CLIP
        embedding = m.text_encoder.token_embedding(tokenized).type(m.dtype)
        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + n_ctx :, :]

        ctx = pl.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1)

        prompts = torch.cat([token_prefix, ctx, token_suffix], dim=1)

        deep_text = list(pl.compound_prompts_text)
        shared_ctx = pl.proj(pl.ctx)  # (n_ctx, vision_dim)

        deep_vision = []
        for i, proj in enumerate(pl.compound_prompt_projections):
            deep_vision.append(proj(pl.compound_prompts_text[i]))

        return prompts, tokenized, deep_text, shared_ctx, deep_vision

    @torch.no_grad()
    def _collect_text_eot_per_layer(self, prompts, tokenized, deep_text):
        """
        Returns list length L (num text blocks):
        feats[l] = (Nclasses, Dproj) EOT features after block l (projected).
        """
        m = self._get_model_unwrapped()
        transformer = m.text_encoder.transformer
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise AttributeError("Text transformer has no attribute 'resblocks'")

        eot_idx = tokenized.argmax(dim=-1)  # (N,)
        L = len(blocks)
        feats = [None] * L
        handles = []

        def make_hook(li):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, (list, tuple)) else output
                # out: (S, N, D)
                x = out.permute(1, 0, 2)  # (N, S, D)
                x = m.text_encoder.ln_final(x).type(m.dtype)
                eot = x[torch.arange(x.size(0), device=x.device), eot_idx]
                eot = eot @ m.text_encoder.text_projection  # (N, Dproj)
                feats[li] = eot.detach().float().cpu()
            return hook

        for li, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(make_hook(li)))

        _ = m.text_encoder(prompts, tokenized, deep_text)

        for h in handles:
            h.remove()

        if any(f is None for f in feats):
            raise RuntimeError("Failed to capture some text-layer outputs via hooks.")

        return feats

    @torch.no_grad()
    def _plot_angdist_text(self, outdir, split_name="all"):
        """
        Saves:
          - angdist_text.png
          - angdist_text.csv
        """
        m = self._get_model_unwrapped()
        was_train = m.training
        m.eval()

        # Use current dataset classnames for *this* split (base or new)
        ds = self.dm.dataset
        classnames = getattr(ds, "classnames", None)
        if classnames is None:
            raise AttributeError("Dataset has no 'classnames' attribute")

        prompts, tokenized, deep_text, _, _ = self._build_prompts_from_classnames(classnames)
        layer_feats = self._collect_text_eot_per_layer(prompts, tokenized, deep_text)

        xs = list(range(1, len(layer_feats)))  # 1..L-1
        ys = []
        for i in range(len(layer_feats) - 1):
            a = torch.from_numpy(layer_feats[i].numpy())
            b = torch.from_numpy(layer_feats[i + 1].numpy())
            ys.append(float(angular_distance_deg(a, b).mean().item()))

        os.makedirs(outdir, exist_ok=True)

        # CSV
        csv_path = osp.join(outdir, "angdist_text.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split", "layer_transition", "avg_angdist_deg"])
            w.writeheader()
            for i, x in enumerate(xs):
                w.writerow({"split": split_name, "layer_transition": f"{x}->{x+1}", "avg_angdist_deg": ys[i]})

        # PNG
        png_path = osp.join(outdir, "angdist_text.png")
        plt.figure(figsize=(10, 7), dpi=200)
        plt.plot(xs, ys, marker="o")
        plt.title(f"Text encoder ang. distance (avg) | split={split_name}")
        plt.xlabel("Text transformer block (i -> i+1)")
        plt.ylabel("Avg angular distance (degrees)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        if was_train:
            m.train()

    @torch.no_grad()
    def _plot_angdist_vision(self, loader, outdir, split_name="all"):
        """
        Saves:
          - angdist_vision.png
          - angdist_vision.csv
        """
        m = self._get_model_unwrapped()
        was_train = m.training
        m.eval()

        max_batches = int(getattr(self.cfg.TRAINER.MAPLE, "ANGDIST_MAX_BATCHES", 50))

        sums = None
        cnts = None
        xs = None

        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break

            if hasattr(self, "parse_batch_test"):
                image, _ = self.parse_batch_test(batch)
            else:
                image, _ = self.parse_batch_train(batch)

            # class-agnostic prompts for vision intermediates
            _, _, _, shared_ctx, deep_vision = self._build_prompts_from_classnames(self.dm.dataset.classnames)

            feats_per_block, eff_depths = m.image_encoder.forward_intermediates(
                image.type(m.dtype), shared_ctx, deep_vision
            )

            if xs is None:
                xs = eff_depths[:-1]
                sums = np.zeros(len(xs), dtype=np.float64)
                cnts = np.zeros(len(xs), dtype=np.float64)

            for i in range(len(feats_per_block) - 1):
                ang = angular_distance_deg(feats_per_block[i], feats_per_block[i + 1])  # (B,)
                sums[i] += float(ang.sum().item())
                cnts[i] += float(ang.numel())

        ys = (sums / np.maximum(cnts, 1.0)).tolist()

        os.makedirs(outdir, exist_ok=True)

        # CSV
        csv_path = osp.join(outdir, "angdist_vision.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split", "layer_transition", "avg_angdist_deg"])
            w.writeheader()
            for i, x in enumerate(xs):
                w.writerow({"split": split_name, "layer_transition": f"{x}->{x+1}", "avg_angdist_deg": ys[i]})

        # PNG
        png_path = osp.join(outdir, "angdist_vision.png")
        plt.figure(figsize=(10, 7), dpi=200)
        plt.plot(xs, ys, marker="o")
        plt.title(f"Vision encoder ang. distance (avg) | split={split_name}")
        plt.xlabel("Vision transformer effective depth (i -> i+1)")
        plt.ylabel("Avg angular distance (degrees)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        if was_train:
            m.train()
