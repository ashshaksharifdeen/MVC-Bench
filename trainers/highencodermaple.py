import os.path as osp
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import convert_weights
import csv
from pathlib import Path
from dassl.evaluation.evaluator import plot_learning_curve
import matplotlib
matplotlib.use("Agg")  # headless-safe (servers)
import matplotlib.pyplot as plt
_tokenizer = _Tokenizer()


       
# -----------------------------
# Helpers (MaPLe style)
# -----------------------------
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "HighEncoderMaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE.N_CTX,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def cast_clip_mha_to_dtype(model: nn.Module, dtype: torch.dtype):
    """
    Critical fix:
    CLIP text transformer uses nn.MultiheadAttention in this repo.
    Its weights can remain float32 while inputs are float16 -> crash.

    This forces ALL MultiheadAttention modules to the desired dtype.
    """
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            m.to(dtype=dtype)

def get_clip_dtype(clip_model: nn.Module) -> torch.dtype:
    # OpenAI CLIP exposes .dtype (read-only) in many forks; otherwise fall back.
    if hasattr(clip_model, "dtype"):
        return clip_model.dtype
    return clip_model.visual.conv1.weight.dtype

def get_clip_token_dtype(clip_model: nn.Module) -> torch.dtype:
    """
    Use token_embedding dtype as the "true" CLIP text dtype.
    Avoid ln_final.dtype because LayerNorms can be kept fp32 in some builds.
    """
    return clip_model.token_embedding.weight.dtype


# -----------------------------
# TextEncoder (MaPLe contract)
# -----------------------------
class TextEncoder(nn.Module):
    """
    Same contract as trainers/maple.py TextEncoder:
    forward(prompts, tokenized_prompts, compound_prompts_deeper_text) -> [C, D]
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        self.dtype = get_clip_dtype(clip_model)

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        # Ensure prompt stream is in the same dtype as CLIP text tokens
        x = prompts.to(self.dtype) + self.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# -----------------------------
# HRM reasoning block
# -----------------------------


class ACTQHead(nn.Module):
    """
    Predict halt/continue from compact statistics of logits.
    Output: q_logits [B,2] -> softmax => p_halt, p_continue
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, logits, step_idx: int, max_steps: int):
        # logits: [B,C]
        probs = logits.softmax(dim=-1)
        top2 = probs.topk(2, dim=-1).values
        maxp = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)

        step_frac = maxp.new_full((maxp.size(0),), float(step_idx) / max(1, (max_steps - 1)))
        feat = torch.stack([maxp, margin, entropy, step_frac], dim=-1)  # [B,4]
        return self.net(feat)  # [B,2]

# -----------------------------
# HRM + MaPLe prompt learner
# -----------------------------
class HRMMultiModalPromptLearner(nn.Module):
    """
    MaPLe outputs are preserved, but shallow per-class ctx is refined using HRMRefiner.
    Deep prompts stay MaPLe-style so CLIP vision/text injection APIs remain unchanged.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
    
        dtype = get_clip_dtype(clip_model) #get_clip_token_dtype(clip_model)
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512 for ViT-B/16 text width
        #self.act_qhead = ACTQHead(hidden=64).to(dtype)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH

        # ---- Shallow ctx init (MaPLe style) ----
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).to(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("HRMMaPLe design: MaPLe + HRM prompt refinement")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # shallow projection to vision prompt space
        self.proj = nn.Linear(ctx_dim, 768).to(dtype)

        self.ctx = nn.Parameter(ctx_vectors)  # [n_ctx, 512]

        # ---- Deep prompts (MaPLe style) ----
        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype))
             for _ in range(self.compound_prompts_depth - 1)]
        )
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        single_layer = nn.Linear(ctx_dim, 768).to(dtype)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        # ---- Token prefix/suffix (class-specific) ----
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])         # [C, 1, D]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :]) # [C, *, D]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        # ---- HRM config ----
        #self.use_hrm = cfg.TRAINER.HRMMAPLE.USE_HRM
        """self.refiner = HRMRefiner(
            dim=ctx_dim,
            n_heads=cfg.TRAINER.HRMMAPLE.N_HEADS,
            mlp_ratio=cfg.TRAINER.HRMMAPLE.MLP_RATIO,
            H_cycles=cfg.TRAINER.HRMMAPLE.H_CYCLES,
            L_cycles=cfg.TRAINER.HRMMAPLE.L_CYCLES,
        ).to(dtype)"""
        """self.use_vhrm = cfg.TRAINER.HRMMAPLE.USE_VHRM   
        # image feat dim is 512 for ViT-B/16 CLIP; vision prompt dim is 768
        img_feat_dim = clip_model.text_projection.shape[1]  # often 512; safer than hardcode
        vis_prompt_dim = 768

        self.imgfeat_to_vproto = nn.Linear(img_feat_dim, vis_prompt_dim).to(dtype)

        self.vis_refiner = HRMRefinerVision(
            dim=vis_prompt_dim,
            n_heads=cfg.TRAINER.HRMMAPLE.V_N_HEADS,
            mlp_ratio=cfg.TRAINER.HRMMAPLE.V_MLP_RATIO,
            H_cycles=cfg.TRAINER.HRMMAPLE.V_H_CYCLES,
            L_cycles=cfg.TRAINER.HRMMAPLE.V_L_CYCLES,
        ).to(dtype)


        self.null_proto = nn.Parameter(torch.zeros(1, 1, ctx_dim, dtype=dtype))
        self.null_vproto = nn.Parameter(torch.zeros(1, 1, vis_prompt_dim, dtype=dtype))"""

    def construct_prompts(self, ctx_class, prefix, suffix):
        return torch.cat([prefix, ctx_class, suffix], dim=1)

    def forward(self, class_prototypes=None, visual_proto=None):
        # Expand shared ctx -> per-class ctx
        ctx_class = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [C, n_ctx, 512]

        """if self.use_hrm and class_prototypes is not None:
            if class_prototypes.dim() == 2:
                class_prototypes = class_prototypes.unsqueeze(1)  # [C, 1, 512]

            # Fill missing classes (zeros) with null prototype
            mask_missing = (class_prototypes.abs().sum(dim=-1, keepdim=True) == 0)  # [C,1,1]
            class_prototypes = torch.where(
                mask_missing,
                self.null_proto.expand_as(class_prototypes),
                class_prototypes
            )

            ctx_class = self.refiner(ctx_class, class_prototypes)"""

        prompts = self.construct_prompts(ctx_class, self.token_prefix, self.token_suffix)

        visual_deep_prompts = []
        for i, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[i]))
        
        # Base (learnable) vision prompts
        #print("projected dimension:", self.proj(self.ctx).shape) torch.Size([2, 768])
        shared_vis = self.proj(self.ctx)  # [n_ctx, 768]
        #deep_vis = [layer(self.compound_prompts_text[i]) for i, layer in enumerate(self.compound_prompt_projections)]
        # each deep_vis[i]: [n_ctx, 768]

        """if self.use_vhrm and (visual_proto is not None):
                # visual_proto: [512] or [1,512] or [1,1,512] -> project to [1,1,768]
                if visual_proto.dim() == 1:
                    visual_proto = visual_proto.unsqueeze(0)
                if visual_proto.dim() == 2:
                    visual_proto = visual_proto.unsqueeze(1)  # [1,1,512]

                vproto_768 = self.imgfeat_to_vproto(visual_proto.squeeze(1)).unsqueeze(1)  # [1,1,768]

                # handle empty proto
                if vproto_768.abs().sum() == 0:
                   vproto_768 = self.null_vproto

                # Option 1 (simple): refine shallow vision tokens only
                shared_vis = self.vis_refiner(shared_vis, vproto_768)"""  # [n_ctx,768]

                # Option 2 (stronger): refine shallow+deep as one sequence, then split back
                # tokens = torch.cat([shared_vis] + deep_vis, dim=0)  # [(K*n_ctx),768]
                # tokens = self.vis_refiner(tokens, vproto_768)
                # shared_vis = tokens[:self.n_ctx]
                # deep_vis = [tokens[(i+1)*self.n_ctx:(i+2)*self.n_ctx] for i in range(len(deep_vis))]

        #idea to recursive refine >> self.compound_prompts_text and visual_deep_prompts
        return prompts, shared_vis, self.compound_prompts_text, visual_deep_prompts


# -----------------------------
# HRMCustomCLIP (2-pass)
# -----------------------------
class HRMCustomCLIP(nn.Module):
    """
    2-pass forward:
      pass-0: build prompts (no HRM), compute image feats, update EMA prototypes
      pass-1: refine prompts via HRM, compute final logits
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = HRMMultiModalPromptLearner(cfg, classnames, clip_model)#.to(torch.float16)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.model_clip_zs = load_clip_to_cpu_zs(cfg)
        #self.model_clip_zs.to(self.device)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        self.img_dtype = clip_model.visual.conv1.weight.dtype
        self.txt_dtype = get_clip_dtype(clip_model) #get_clip_token_dtype(clip_model)

        C = self.prompt_learner.n_cls
        D = clip_model.ln_final.weight.shape[0]
        self.register_buffer("proto_ema", torch.zeros(C, D, dtype=self.txt_dtype))

        self.proto_m = cfg.TRAINER.HRMMAPLE.PROTO_MOMENTUM
        self.use_ema = cfg.TRAINER.HRMMAPLE.USE_EMA_PROTO
        self.normalize_proto = cfg.TRAINER.HRMMAPLE.NORM_PROTO
    #creating a prototype >> consider modyfying it
    @torch.no_grad()
    def _batch_prototypes(self, image_features, labels):
        C = self.prompt_learner.n_cls
        B, D = image_features.shape
        device = image_features.device
        protos = torch.zeros(C, D, device=device, dtype=image_features.dtype)
        counts = torch.zeros(C, device=device, dtype=torch.float32)

        for i in range(B):
            c = int(labels[i].item())
            protos[c] += image_features[i]
            counts[c] += 1.0

        counts = counts.clamp_min(1.0).unsqueeze(1)
        protos = protos / counts
        return protos

    @torch.no_grad()
    def _update_proto_ema(self, batch_proto):
        if self.normalize_proto:
            batch_proto = batch_proto / (batch_proto.norm(dim=-1, keepdim=True) + 1e-6)
        self.proto_ema.mul_(self.proto_m).add_(
            batch_proto.to(self.proto_ema.dtype),
            alpha=(1.0 - self.proto_m)
        )

    def reset_deep_supervision_state(self):
        # text transformer blocks
        for blk in self.text_encoder.transformer.resblocks:
            if hasattr(blk, "reset_ds_state"):
                blk.reset_ds_state()

        # vision transformer blocks
        for blk in self.image_encoder.transformer.resblocks:
            if hasattr(blk, "reset_ds_state"):
                blk.reset_ds_state()    

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # ---- Pass-0 ----
        #prompts0, shared_ctx0, deep_text0, deep_vis0 = self.prompt_learner(class_prototypes=None, visual_proto=None)
        #text_features0 = self.text_encoder(prompts0, tokenized_prompts, deep_text0)
        #image_features0 = self.model_clip_zs.encode_image(image.to(self.img_dtype))
        """with torch.no_grad():
            # batch mean prototype in image feature space (512-dim)
            vproto = image_features0.detach().mean(dim=0)   # [512]
        image_features0 = image_features0 / image_features0.norm(dim=-1, keepdim=True)
        #text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)"""

        # ---- Build/update prototypes ----
        """class_prototypes = None
        if label is not None:
            batch_proto = self._batch_prototypes(image_features0.detach(), label)
            self._update_proto_ema(batch_proto)
            class_prototypes = self.proto_ema.to(self.txt_dtype)
        else:
            if self.use_ema and (self.proto_ema.abs().sum() > 0):
                class_prototypes = self.proto_ema.to(self.txt_dtype)"""

        # ---- Pass-1 (HRM refined) ----
        h_cycle = 2
        l_cycle = 2
        prompts, shared_ctx, deep_text, deep_vis = self.prompt_learner(class_prototypes=None, visual_proto=None)
        with torch.no_grad():
            for h in range(h_cycle):
                for l in range(l_cycle):   
                    text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
                    image_features,q_logits = self.image_encoder(image.to(self.img_dtype), shared_ctx, deep_vis)
                #hycycle
                text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
                image_features,q_logits = self.image_encoder(image.to(self.img_dtype), shared_ctx, deep_vis)
        for l in range(l_cycle):   
                text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
                image_features,q_logits = self.image_encoder(image.to(self.img_dtype), shared_ctx, deep_vis)
        #hycycle
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
        image_features,q_logits = self.image_encoder(image.to(self.img_dtype), shared_ctx, deep_vis)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training and (label is not None):
            #print("q_logits shape:", q_logits.shape)
            #q_prob= torch.softmax(q_logits,dim=-1)
            #q_hat=q_prob.max(dim=-1)[0]
            #p_halt=q_prob[:,0]
            #print("q-hat value:", q_hat)
            #print("q-hat shape:", q_hat.shape)
            # q_logits is [B,2] (halt vs continue)
            halt_logit = q_logits[:, 0].float()  # raw logit for HALT (no softmax)
            #print("halt_logit:", halt_logit)
            #print("q_logits[:, 1]:", q_logits[:, 1].half())
            #y=torch.softmax(logits,dim=-1)
            #y_hat=y.argmax(dim=-1)
            #print("y-hat ", y_hat)
            #print("y-hat shape:", y_hat.shape)
            #target=(y_hat==label).half()
            # target must be float (0/1)
            target = (logits.argmax(dim=-1) == label).float()
            BCE_loss=F.binary_cross_entropy_with_logits(halt_logit, target)
            loss=F.cross_entropy(logits, label)#+BCE_loss
            return loss, torch.sigmoid(halt_logit).detach()
            #return F.cross_entropy(logits, label)

        return logits
    
    """def forward_step(self, image, step_idx: int, max_steps: int, halt_mask=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # build prompts (your current pass-1)
        prompts, shared_ctx, deep_text, deep_vis = self.prompt_learner(class_prototypes=None, visual_proto=None)

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
        image_features = self.image_encoder(image.to(self.img_dtype), shared_ctx, deep_vis)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()  # [B,C]

        # Q-head (detach logits so Q learning doesn’t distort classifier unless you want it)
        q_logits = self.prompt_learner.act_qhead(logits.detach(), step_idx, max_steps)  # [B,2]
        return logits, q_logits"""


# -----------------------------
# TrainerX: HRMMaPLe
# -----------------------------
@TRAINER_REGISTRY.register()
class HighEncoderMaPLe(TrainerX):
    """
    HighEncoderMaPLe = MaPLe + HigherEncoder reasoning-based refinement on shallow text context tokens.
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.HRMMAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # Match MaPLe precision behavior
        if cfg.TRAINER.HRMMAPLE.PREC in ["fp32", "amp"]:
            clip_model.float()
        #if cfg.TRAINER.HRMMAPLE.PREC == "fp16":
        #    convert_weights(clip_model)
            #self.model.prompt_learner.to(torch.float16)
        # IMPORTANT FIX:
        # force transformer MHA weights to match CLIP token dtype
        #cast_clip_mha_to_dtype(clip_model, get_clip_token_dtype(clip_model))

        print("Building HighEncoderMaPLe + MaPLe CLIP")
        self.model = HRMCustomCLIP(cfg, classnames, clip_model)

        print("Freezing CLIP encoders; training only prompt_learner (incl. HRM)")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.HRMMAPLE.PREC == "amp" else None

    
    def _act_losses(self, logits, q_logits, label, active_mask, lam_q, lam_ponder):
        # classification loss only for active samples
        loss_cls = F.cross_entropy(logits[active_mask], label[active_mask])

        # correctness signal (supervision for Q)
        pred = logits.argmax(dim=-1)
        correct = (pred == label).float()  # [B]

        # Q losses (halt=correct, continue=incorrect)
        q_halt = q_logits[:, 0]
        q_cont = q_logits[:, 1]

        target_halt = correct
        target_cont = 1.0 - correct

        bce = nn.BCEWithLogitsLoss()
        loss_q = bce(q_halt[active_mask], target_halt[active_mask]) + bce(q_cont[active_mask], target_cont[active_mask])

        # ponder cost: penalize remaining active samples each step
        ponder = active_mask.float().mean()

        return loss_cls + lam_q * loss_q + lam_ponder * ponder
    
    def forward_backward(self, batch):
        
        # --- init learning-curve tracking once ---
        self._init_learning_curve_state_if_needed()

        # --- reset epoch accumulators when a new epoch starts ---
        cur_epoch = getattr(self, "epoch", 0)
        if self._lc_last_epoch != cur_epoch:
            self._lc_last_epoch = cur_epoch
            self._lc_epoch_train_sum = 0.0
            self._lc_epoch_train_count = 0

        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.HRMMAPLE.PREC
        nsup = 9 #int(self.cfg.TRAINER.HRMMAPLE.NSUP)
        """max_steps = self.cfg.TRAINER.HRMMAPLE.ACT_MAX_STEPS
        thr = self.cfg.TRAINER.HRMMAPLE.ACT_HALT_THRESHOLD
        lam_q = self.cfg.TRAINER.HRMMAPLE.ACT_LAMBDA_Q
        lam_ponder = self.cfg.TRAINER.HRMMAPLE.ACT_LAMBDA_PONDER
        eps = self.cfg.TRAINER.HRMMAPLE.ACT_EPSILON 
        self.model.reset_deep_supervision_state()
        B = label.size(0)
        halt_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        total_loss = 0.0
        last_logits = None
        steps_used = 0
        self.optim.zero_grad(set_to_none=True)
        final_logits = None"""
        # IMPORTANT: reset zH/zL for this mini-batch (so no leakage across batches)
        """for step in range(max_steps):
                steps_used += 1
                active = ~halt_mask
                if not active.any():
                    break

                if prec == "amp":
                    with autocast():
                        logits, q_logits = self.model.forward_step(image, step, max_steps, halt_mask=halt_mask)
                        loss_step = self._act_losses(logits, q_logits, label, active, lam_q, lam_ponder)
                    self.scaler.scale(loss_step).backward()
                else:
                        logits, q_logits = self.model.forward_step(image, step, max_steps, halt_mask=halt_mask)
                        loss_step = self._act_losses(logits, q_logits, label, active, lam_q, lam_ponder)
                        loss_step.backward()
                if final_logits is None:
                        final_logits = torch.empty_like(logits)
       
                total_loss = total_loss + float(loss_step.item())
                last_logits = logits

                # update halting decisions (TRM-like: softmax over [halt,continue])
                p = q_logits.softmax(dim=-1)      # [B,2]
                p_halt = p[:, 0]                  # [B]
                # optional exploration during training
                if self.model.training and eps > 0:
                        explore = (torch.rand_like(p_halt) < eps) & active
                        # random halt on explored samples
                        rand_halt = (torch.rand_like(p_halt) < 0.5)
                        halt_now = ((p_halt > thr) | (explore & rand_halt)) & active
                else:
                        halt_now = (p_halt > thr) & active

                # force halt at last step
                if step == max_steps - 1:
                        halt_now = active
                if halt_now.any():
                    final_logits[halt_now] = logits.detach()[halt_now]
                halt_mask = halt_mask | halt_now

        still_active = ~halt_mask
        if still_active.any():
                final_logits[still_active] = logits.detach()[still_active]
        # optimizer step ONCE
        if prec == "amp":
                self.scaler.step(self.optim)
                self.scaler.update()
        else:
                self.optim.step()"""
        if hasattr(self.model, "reset_deep_supervision_state"):
            self.model.reset_deep_supervision_state()

        loss_sum = 0.0
        steps_done = 0

        for step in range(nsup):
            steps_done += 1

            if prec == "amp":
                self.optim.zero_grad(set_to_none=True)
                with autocast():
                    loss, q_hat = self.model(image, label)  # returns CE loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.zero_grad(set_to_none=True)
                loss, q_hat = self.model(image, label)
                loss.backward()
                self.optim.step()
            #if (q_hat>0.5).all().item():
            #    break    

            loss_sum += float(loss.item())

            # Optional: you can later add ACT/Q early stopping here (paper), but skip for now.

        avg_loss = loss_sum / max(steps_done, 1)
        #avg_loss = total_loss / max(steps_used, 1)
        # logging + acc (use one forward; okay if it mutates state since next batch resets anyway)
        with torch.no_grad():
            logits = self.model(image, None)
            acc = compute_accuracy(logits, label)[0].item()

        bs = int(label.size(0))
        self._lc_epoch_train_sum += avg_loss * bs
        self._lc_epoch_train_count += bs


        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
            # --- finalize epoch learning curve ---
            avg_train_loss = self._lc_epoch_train_sum / max(self._lc_epoch_train_count, 1)

            val_loader = self._get_val_loader()
            avg_val_loss = self._compute_loss_on_loader(val_loader)

            self._lc_train_losses.append(avg_train_loss)
            self._lc_val_losses.append(avg_val_loss if avg_val_loss is not None else float("nan"))

            plot_learning_curve(
                self._lc_train_losses,
                self._lc_val_losses,
                self._lc_plot_path,
                title=f"{self.cfg.DATASET.NAME} | {self.cfg.TRAINER.NAME} | Learning Curve (Seed {self.cfg.SEED})",
            )

        return {"loss": loss.item(), "acc": acc}
    
    def _get_val_loader(self):
        # Try common places used in Dassl/TrainerX setups
        if hasattr(self, "val_loader") and self.val_loader is not None:
            return self.val_loader
        if hasattr(self, "dm") and hasattr(self.dm, "val_loader") and self.dm.val_loader is not None:
            return self.dm.val_loader
        # fallback (if no val split exists)
        return None
    @torch.no_grad()
    def forward_act(self, image):
        max_steps = self.cfg.TRAINER.HRMMAPLE.ACT_MAX_STEPS
        thr = self.cfg.TRAINER.HRMMAPLE.ACT_HALT_THRESHOLD

        self.model.reset_deep_supervision_state()
        B = image.size(0)
        halt_mask = torch.zeros(B, dtype=torch.bool, device=image.device)

        logits = None
        for step in range(max_steps):
            logits, q_logits = self.model.forward_step(image, step, max_steps, halt_mask=halt_mask)
            p_halt = q_logits.softmax(dim=-1)[:, 0]
            active = ~halt_mask
            halt_now = (p_halt > thr) & active
            if step == max_steps - 1:
                halt_now = active
            halt_mask |= halt_now
            if halt_mask.all():
                break
        return logits
    @torch.no_grad()
    def _compute_loss_on_loader(self, loader):
        if loader is None:
            return None

        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_n    = 0

        for batch in loader:
            x = batch["img"].to(self.device)
            y = batch["label"].to(self.device)

            # HRMCustomCLIP returns logits when label=None
            logits = self.model(x, None)
            loss   = F.cross_entropy(logits, y, reduction="sum")

            total_loss += float(loss.item())
            total_n    += int(y.size(0))

        if was_training:
            self.model.train()

        return total_loss / max(total_n, 1)

    def _init_learning_curve_state_if_needed(self):
        if hasattr(self, "_lc_initialized") and self._lc_initialized:
            return

        self._lc_initialized = True
        self._lc_train_losses = []
        self._lc_val_losses   = []

        self._lc_epoch_train_sum   = 0.0
        self._lc_epoch_train_count = 0
        self._lc_last_epoch        = -999999

        self._lc_plot_path = osp.join(
            self.cfg.OUTPUT_DIR,
            f"{self.cfg.DATASET.NAME}_{self.cfg.TRAINER.NAME}_overall_learning_curve{self.cfg.SEED}.png"
        )

    def parse_batch_train(self, batch):
        x = batch["img"].to(self.device)
        y = batch["label"].to(self.device)
        return x, y

    def load_model(self, directory, epoch=None):
        if not directory:
            print("load_model skipped: no directory given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]

            # Ignore computed token buffers if present
            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)

            print(f'Loading weights to {name} from "{model_path}"')
            self._models[name].load_state_dict(state_dict, strict=False)
