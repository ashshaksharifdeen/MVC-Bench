import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.model import convert_weights
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
from trainers.regularizers import REGULARIZER_REGISTRY
import math
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
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


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

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

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        #temperature scaling
        #logits= logits/1.16
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

            return F.cross_entropy(logits,
                                   label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits

def _l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def _angular_distance(a, b, eps=1e-6, degrees=True):
    """
    a, b: (..., D) already on same device
    returns: (...) angle
    """
    a = _l2norm(a, eps)
    b = _l2norm(b, eps)
    cos = (a * b).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    ang = torch.acos(cos)
    if degrees:
        ang = ang * (180.0 / math.pi)
    return ang

def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def _plot_curve(path, xs, ys, title, xlabel, ylabel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@TRAINER_REGISTRY.register()
class PromptSRC(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
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
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # ---- add THESE angdist flags at the end of build_model ----
        self.plot_angdist = cfg.TRAINER.PROMPTSRC.PLOT_ANGDIST
        self.angdist_max_batches = int(cfg.TRAINER.PROMPTSRC.ANGDIST_MAX_BATCHES)
        self.angdist_in_degrees = bool(cfg.TRAINER.PROMPTSRC.ANGDIST_IN_DEGREES)
        self.angdist_eps = float(cfg.TRAINER.PROMPTSRC.ANGDIST_EPS)
        self.angdist_save_csv = bool(cfg.TRAINER.PROMPTSRC.ANGDIST_SAVE_CSV)
        self.angdist_save_png = bool(cfg.TRAINER.PROMPTSRC.ANGDIST_SAVE_PNG)
        self.angdist_subdir = str(cfg.TRAINER.PROMPTSRC.ANGDIST_SUBDIR)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        
        # --- zero-shot CLIP hook ---
        self.zeroshot = ZeroshotCLIP()
        self.zeroshot.cfg = cfg
        self.zeroshot.dm = self.dm
        self.zeroshot.device = self.device
        self.zeroshot.build_model()
        
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # Keep model with GPA
        self.previous_model_gpa = None
        # ---- angular distance plot flags ----
        self.plot_angdist = cfg.TRAINER.PROMPTSRC.PLOT_ANGDIST
        self.angdist_max_batches = cfg.TRAINER.PROMPTSRC.ANGDIST_MAX_BATCHES
        self.angdist_in_degrees = cfg.TRAINER.PROMPTSRC.ANGDIST_IN_DEGREES
        self.angdist_subdir = cfg.TRAINER.PROMPTSRC.ANGDIST_SUBDIR

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            # Calculate the L_SCL_text loss
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                      reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()

            with torch.no_grad():
                zs_log = self.zeroshot.model_inference(image)  # [B, C]
                zs_img = self.zeroshot.clip_model.encode_image(image)
                #zs_img = zs_img / zs_img.norm(dim=-1, keepdim=True)  # [B, D]
                zs_txt = self.zeroshot.text_features 

            reg_fn = REGULARIZER_REGISTRY.get('margin_mean_var')
            margin_reg = reg_fn(logits, label, alpha=0.1, beta=0.01)

                        # fetch the regularizer
            mm_fn = REGULARIZER_REGISTRY.get("text_moment_matching")

            # compute it
            loss_mm_txt = mm_fn(normalized_text_features, zs_txt)

            
            #eccv_penalty
            eccv_penalty = REGULARIZER_REGISTRY.get("eccv_penalty")
            eccv_penalty_loss = eccv_penalty(zs_pred=zs_log, output=logits)
            #eccv_zeroshot
            eccv_zs = REGULARIZER_REGISTRY.get("eccv_zs")
            eccv_zs_loss = eccv_zs(zs_pred=zs_log, output=logits,label=label)
            #MDCA
            mdca = REGULARIZER_REGISTRY.get("MDCA")
            mdca_loss  = mdca(output=logits,label=label)
            #MBLS
            mbls = REGULARIZER_REGISTRY.get("MBLS")
            mbls_loss  = mbls(logits=logits,targets=label)
            #DCA
            dca = REGULARIZER_REGISTRY.get("DCA")
            dca_loss  = dca(logits=logits,label=label)
            #label smooth
            label_smooth = REGULARIZER_REGISTRY.get("label_smooth")
            label_smooth_loss  = label_smooth(output=logits,label=label)

            #mean var edit
            margin_var_all = REGULARIZER_REGISTRY.get("margin_mean_var_all")
            margin_var_all_loss  = margin_var_all(logits=logits,label=label,variance_mode='per_sample')
            explicit_all = REGULARIZER_REGISTRY.get("margin_mean_var_allclass_loss_explicit")
            explicit_all_loss =explicit_all(logits,label,variance_mode="all_pairs")
            #end-----    
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            loss = (loss_ce + L_SCL + explicit_all_loss ) #oxford_flowers
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

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

    def test(self, split=None):
        results = super().test(split)
        if getattr(self, "plot_angdist", False):
            print("[ANGDIST] Writing CSV/PNG angular-distance figures...")
            self.dump_angdist_figures()
        return results        

    def _get_model(self):
        # handle DataParallel
        m = self.model
        if hasattr(m, "module"):
           return m.module
        return m

    @torch.no_grad()
    def _compute_text_angdist(self):
        """
        Computes angular distance between consecutive text-transformer blocks,
        averaged across class prompts.
        """
        model = self._get_model()  # CustomCLIP
        text_enc = model.text_encoder  # TextEncoder
        prompts = model.prompt_learner()                 # (n_cls, n_tkn, dim) :contentReference[oaicite:4]{index=4}
        tokenized = model.tokenized_prompts              # (n_cls, n_tkn) :contentReference[oaicite:5]{index=5}

        transformer = text_enc.transformer
        ln_final = text_enc.ln_final
        text_proj = text_enc.text_projection
        pos_embed = text_enc.positional_embedding
        dtype = model.dtype

        # Find blocks
        if hasattr(transformer, "resblocks"):
            blocks = list(transformer.resblocks)
            run_transformer = transformer
        else:
            # fallback: treat transformer itself as iterable of blocks
            blocks = list(transformer)

        # capture each block output using hooks (safe even if transformer has special prompt logic)
        outs = [None] * len(blocks)
        handles = []

        for i, blk in enumerate(blocks):
            def _make_hook(ii):
                def _hook(_m, _inp, _out):
                    outs[ii] = _out.detach()
                return _hook
            handles.append(blk.register_forward_hook(_make_hook(i)))

        # Run one forward through transformer
        x = prompts + pos_embed.type(dtype)
        x = x.permute(1, 0, 2)  # (L, Ncls, C)
        _ = run_transformer(x)  # hooks fill outs

        for h in handles:
            h.remove()

        # Convert each layer output -> CLIP text embedding at that layer
        eot = tokenized.argmax(dim=-1)  # (n_cls,)
        layer_feats = []
        for out in outs:
            if out is None:
                continue
            y = out.permute(1, 0, 2)        # (n_cls, L, C)
            y = ln_final(y).type(dtype)
            feat = y[torch.arange(y.shape[0]), eot] @ text_proj
            feat = _l2norm(feat.float(), self.angdist_eps)
            layer_feats.append(feat)

        # consecutive angular distances
        ys = []
        for i in range(len(layer_feats) - 1):
            ang = _angular_distance(layer_feats[i], layer_feats[i+1],
                                    eps=self.angdist_eps,
                                    degrees=self.angdist_in_degrees)
            ys.append(float(ang.mean().item()))
        return ys

    @torch.no_grad()
    def _compute_vision_angdist(self):
        """
        Computes angular distance between consecutive vision-transformer blocks,
        averaged over images from current test loader.
        """
        model = self._get_model()
        vis = model.image_encoder  # clip_model.visual :contentReference[oaicite:6]{index=6}
        dtype = model.dtype

        if not hasattr(vis, "transformer") or not hasattr(vis.transformer, "resblocks"):
            print("[ANGDIST] Vision encoder is not ViT-style; skipping vision angdist.")
            return None

        blocks = list(vis.transformer.resblocks)
        outs = [None] * len(blocks)
        handles = []

        for i, blk in enumerate(blocks):
            def _make_hook(ii):
                def _hook(_m, _inp, _out):
                    outs[ii] = _out.detach()
                return _hook
            handles.append(blk.register_forward_hook(_make_hook(i)))

        # accumulate angle sums over dataset
        sum_angles = [0.0] * (len(blocks) - 1)
        count = 0

        max_batches = self.angdist_max_batches
        for bidx, batch in enumerate(self.test_loader):
            if bidx >= max_batches:
                break
            img = batch["img"].to(self.device)  # standard Dassl batch dict
            # Run normal forward once; hooks grab every block output
            _ = vis(img.type(dtype))

            # use class-token from each block output, then ln_post (+ proj if exists)
            layer_feats = []
            for out in outs:
                if out is None:
                    continue
                # CLIP ViT blocks typically output (L, B, C) where token 0 is CLS
                if out.dim() == 3:
                    cls = out[0]  # (B, C)
                else:
                    continue

                if hasattr(vis, "ln_post") and vis.ln_post is not None:
                    cls = vis.ln_post(cls)
                if hasattr(vis, "proj") and vis.proj is not None:
                    cls = cls @ vis.proj
                cls = _l2norm(cls.float(), self.angdist_eps)
                layer_feats.append(cls)

            # accumulate consecutive distances for this batch
            for i in range(len(layer_feats) - 1):
                ang = _angular_distance(layer_feats[i], layer_feats[i+1],
                                        eps=self.angdist_eps,
                                        degrees=self.angdist_in_degrees)
                sum_angles[i] += float(ang.sum().item())

            count += layer_feats[0].shape[0]

        for h in handles:
            h.remove()

        if count == 0:
            return None

        ys = [s / count for s in sum_angles]
        return ys

    def dump_angdist_figures(self):
        cfg = self.cfg
        split_tag = getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all")

        out_dir = osp.join(cfg.OUTPUT_DIR, "angdist", split_tag)
        os.makedirs(out_dir, exist_ok=True)

        # TEXT
        text_ys = self._compute_text_angdist()
        text_xs = list(range(1, len(text_ys) + 1))  # x=i means between layer i and i+1

        if self.angdist_save_csv:
            rows = [[split_tag, i, i+1, 1, float(y)] for i, y in enumerate(text_ys, start=1)]
            _write_csv(
                osp.join(out_dir, "angdist_text.csv"),
                header=["split", "src_layer", "tgt_layer", "layers_passed", "avg_angdist"],
                rows=rows
            )

        if self.angdist_save_png:
            _plot_curve(
                osp.join(out_dir, "angdist_text.png"),
                text_xs, text_ys,
                title=f"Text encoder angular distance (split={split_tag})",
                xlabel="Layer index i  (distance between i and i+1)",
                ylabel=f"Avg angular distance ({'deg' if self.angdist_in_degrees else 'rad'})"
            )

        # VISION
        vis_ys = self._compute_vision_angdist()
        if vis_ys is not None:
            vis_xs = list(range(1, len(vis_ys) + 1))

            if self.angdist_save_csv:
                rows = [[split_tag, i, i+1, 1, float(y)] for i, y in enumerate(vis_ys, start=1)]
                _write_csv(
                    osp.join(out_dir, "angdist_vision.csv"),
                    header=["split", "src_layer", "tgt_layer", "layers_passed", "avg_angdist"],
                    rows=rows
                )

            if self.angdist_save_png:
                _plot_curve(
                    osp.join(out_dir, "angdist_vision.png"),
                    vis_xs, vis_ys,
                    title=f"Vision encoder angular distance (split={split_tag})",
                    xlabel="Layer index i  (distance between i and i+1)",
                    ylabel=f"Avg angular distance ({'deg' if self.angdist_in_degrees else 'rad'})"
                )  