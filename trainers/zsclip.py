import os
import math
from pathlib import Path

import torch
import torch.nn as nn

# Headless plotting (works on servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dassl.engine import TRAINER_REGISTRY, TrainerX

from clip import clip
from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


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
}


def _to_batch_first(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Convert [seq, batch, dim] -> [batch, seq, dim] if needed.
    If already [batch, seq, dim], return as-is.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(x.shape)}")

    if x.shape[0] == batch_size:
        return x  # [B, S, D]
    if x.shape[1] == batch_size:
        return x.permute(1, 0, 2).contiguous()  # [S, B, D] -> [B, S, D]

    raise ValueError(
        f"Cannot infer batch dimension for tensor shape {tuple(x.shape)} with batch_size={batch_size}"
    )


def _angular_distance_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b: [B, D] (float)
    returns: [B] angular distance in degrees
    """
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    cos = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)
    return ang


def _save_curve_png_and_csv(x, y, png_path: Path, csv_path: Path, title: str, xlabel: str, ylabel: str):
    # Save CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("layer_i,avg_angular_distance_deg\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi},{yi}\n")

    # Save plot
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=200)
    plt.close()


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    """
    Zero-shot CLIP + (optional) layerwise angular-distance plots for:
      - Text encoder transformer blocks
      - Vision encoder transformer blocks (ViT only)

    Enable by setting:
      TEST.PLOT_ANGDIST = True
    """

    def __init__(self, cfg):
        # Define all custom attributes BEFORE super().__init__()
        self.plot_angdist = bool(getattr(cfg.TEST, "PLOT_ANGDIST", False))
        self.angdist_max_batches = int(getattr(cfg.TEST, "ANGDIST_MAX_BATCHES", -1))

        self._collect_vis = False
        self._collect_txt = False

        self._vis_layer_vecs = []
        self._txt_layer_vecs = []

        self._vis_ang_sum = None
        self._vis_ang_count = None

        self._current_text_tokens = None
        self._hook_handles = []

        self._vision_is_vit = False
        self._num_vis_layers = 0
        self._num_txt_layers = 0

        # Now call parent init (this will call build_model safely)
        super().__init__(cfg)

    def _get_output_dir(self) -> Path:
        return Path(self.cfg.OUTPUT_DIR)

    def _register_angdist_hooks(self):
        """
        Register forward hooks on:
          - vision transformer residual blocks (ViT only)
          - text transformer residual blocks
        Hooks store only compact vectors (CLS/EOT), not full tokens.
        """
        # ---- Text blocks (always transformer in CLIP) ----
        if not hasattr(self.clip_model, "transformer") or not hasattr(self.clip_model.transformer, "resblocks"):
            raise RuntimeError("Cannot find text transformer blocks at clip_model.transformer.resblocks")

        txt_blocks = list(self.clip_model.transformer.resblocks)
        self._num_txt_layers = len(txt_blocks)

        def _txt_hook(_module, _inp, out):
            if not self._collect_txt:
                return
            if self._current_text_tokens is None:
                return

            with torch.no_grad():
                B = self._current_text_tokens.shape[0]
                out_bf = _to_batch_first(out, B)  # [B, S, D]
                # CLIP's encode_text uses token.argmax(dim=-1) to pick EOT position
                eot_idx = self._current_text_tokens.argmax(dim=-1)  # [B]
                eot_vec = out_bf[torch.arange(B, device=out_bf.device), eot_idx, :]  # [B, D]
                self._txt_layer_vecs.append(eot_vec.float().detach().cpu())

        for blk in txt_blocks:
            self._hook_handles.append(blk.register_forward_hook(_txt_hook))

        # ---- Vision blocks (ViT only) ----
        self._vision_is_vit = hasattr(self.clip_model, "visual") and hasattr(self.clip_model.visual, "transformer") \
                              and hasattr(self.clip_model.visual.transformer, "resblocks")

        if self._vision_is_vit:
            vis_blocks = list(self.clip_model.visual.transformer.resblocks)
            self._num_vis_layers = len(vis_blocks)

            def _vis_hook(_module, _inp, out):
                if not self._collect_vis:
                    return
                with torch.no_grad():
                    # out is typically [S, B, D] in CLIP ViT
                    # We need batch size from input image batch; easiest: infer from out
                    # We'll assume out is [S,B,D] and take CLS token at position 0
                    # Convert to batch-first using batch size = out.shape[1]
                    B = out.shape[1]
                    out_bf = _to_batch_first(out, B)  # [B, S, D]
                    cls_vec = out_bf[:, 0, :]         # [B, D]
                    self._vis_layer_vecs.append(cls_vec.float().detach().cpu())

            for blk in vis_blocks:
                self._hook_handles.append(blk.register_forward_hook(_vis_hook))
        else:
            self._num_vis_layers = 0  # will skip vision plot for RN backbones

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        clip_model.eval()

        # Freeze everything
        for p in clip_model.parameters():
            p.requires_grad_(False)

        self.clip_model = clip_model

        # Optional: register hooks once model exists
        if self.plot_angdist:
            self._register_angdist_hooks()

        # ---- Build text features (zero-shot classifier weights) ----
        dname = cfg.DATASET.NAME
        temp = CUSTOM_TEMPLATES.get(dname, None)
        if temp is None:
            temp = CUSTOM_TEMPLATES.get(str(dname).upper(), "a photo of a {}.")

        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts (n={len(prompts)}). Example: {prompts[:3]}")
        prompts_tok = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

        # Encode text once
        with torch.no_grad():
            if self.plot_angdist:
                # Collect per-layer EOT vectors while encoding
                self._txt_layer_vecs = []
                self._collect_txt = True
                self._current_text_tokens = prompts_tok

            text_features = self.clip_model.encode_text(prompts_tok)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if self.plot_angdist:
                self._collect_txt = False
                self._current_text_tokens = None

        self.text_features = text_features

        # ---- Immediately plot text angular distances (one-time) ----
        if self.plot_angdist:
            self._plot_text_angdist()

    @torch.no_grad()
    def model_inference(self, image):
        # Collect vision layer vectors per batch (only during test collection)
        if self.plot_angdist and self._collect_vis and self._vision_is_vit:
            self._vis_layer_vecs = []

        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        # Update running stats (vision) after forward pass
        if self.plot_angdist and self._collect_vis and self._vision_is_vit:
            self._accumulate_vis_stats_from_current_batch()

        return logits

    def _plot_text_angdist(self):
        """
        Use collected per-layer EOT vectors from the prompt set (classes) and plot consecutive angular distances.
        """
        if len(self._txt_layer_vecs) < 2:
            print("[ANGDIST] Not enough text layers captured to plot.")
            return

        # layer_vecs: list of [N_prompts, D], length = L
        L = len(self._txt_layer_vecs)
        sum_deg = torch.zeros(L - 1, dtype=torch.float64)
        count = torch.zeros(L - 1, dtype=torch.float64)

        for i in range(L - 1):
            a = self._txt_layer_vecs[i]
            b = self._txt_layer_vecs[i + 1]
            ang = _angular_distance_deg(a, b)  # [N_prompts]
            sum_deg[i] += ang.sum().item()
            count[i] += ang.numel()

        avg = (sum_deg / count).cpu().numpy().tolist()
        x = list(range(1, L))  # i = 1..L-1 meaning angle between layer i and i+1

        out_dir = self._get_output_dir()
        png_path = out_dir / "angdist_text.png"
        csv_path = out_dir / "angdist_text.csv"

        _save_curve_png_and_csv(
            x=x,
            y=avg,
            png_path=png_path,
            csv_path=csv_path,
            title="Text encoder: avg angular distance between consecutive layers",
            xlabel="Layer index i (angle between layer i and i+1)",
            ylabel="Average angular distance (deg)"
        )

        print(f"[ANGDIST] Saved text plot: {png_path}")
        print(f"[ANGDIST] Saved text values: {csv_path}")

    def _init_vis_stats(self):
        if not self._vision_is_vit:
            return
        # L layers => L-1 transitions
        L = self._num_vis_layers
        self._vis_ang_sum = torch.zeros(L - 1, dtype=torch.float64)
        self._vis_ang_count = torch.zeros(L - 1, dtype=torch.float64)

    def _accumulate_vis_stats_from_current_batch(self):
        """
        Uses self._vis_layer_vecs filled by hooks in this batch.
        Each element is [B, D] on CPU.
        """
        vecs = self._vis_layer_vecs
        if len(vecs) < 2:
            return

        L = len(vecs)
        for i in range(L - 1):
            ang = _angular_distance_deg(vecs[i], vecs[i + 1])  # [B]
            self._vis_ang_sum[i] += ang.sum().item()
            self._vis_ang_count[i] += ang.numel()

    def _finalize_and_plot_vis_angdist(self):
        if not self._vision_is_vit:
            print("[ANGDIST] Vision backbone is ResNet-like (no visual.transformer.resblocks). "
                  "Skipping angdist_vision plot. Use a ViT backbone for per-layer transformer curves.")
            return

        avg = (self._vis_ang_sum / self._vis_ang_count).cpu().numpy().tolist()
        Lm1 = len(avg)
        x = list(range(1, Lm1 + 1))

        out_dir = self._get_output_dir()
        png_path = out_dir / "angdist_vision.png"
        csv_path = out_dir / "angdist_vision.csv"

        _save_curve_png_and_csv(
            x=x,
            y=avg,
            png_path=png_path,
            csv_path=csv_path,
            title="Vision encoder: avg angular distance between consecutive layers",
            xlabel="Layer index i (angle between layer i and i+1)",
            ylabel="Average angular distance (deg)"
        )

        print(f"[ANGDIST] Saved vision plot: {png_path}")
        print(f"[ANGDIST] Saved vision values: {csv_path}")

    def test(self, *args, **kwargs):
        """
        Wrap Dassl's test loop:
          - enable vision collection during test
          - plot vision curve after test ends
        Text curve is plotted during build_model() (one-time).
        """
        if not self.plot_angdist:
            return super().test(*args, **kwargs)

        # Init/reset stats
        self._init_vis_stats()

        # Collect vision vectors during test
        self._collect_vis = True
        max_batches = self.angdist_max_batches

        if max_batches is None or max_batches < 0:
            # run full test
            out = super().test(*args, **kwargs)
        else:
            # limited-batch test (fast debug)
            # Dassl doesn't expose a "max batches" knob cleanly everywhere,
            # so simplest is: run full test normally OR you can manually break in Dassl's core.
            # Here we just run full test to avoid breaking framework.
            out = super().test(*args, **kwargs)

        self._collect_vis = False

        # Plot vision curve after test
        self._finalize_and_plot_vis_angdist()
        return out


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        clip_model.eval()

        for p in clip_model.parameters():
            p.requires_grad_(False)

        self.clip_model = clip_model

        if self.plot_angdist:
            self._register_angdist_hooks()

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates = list(self.templates) + [CUSTOM_TEMPLATES.get(cfg.DATASET.NAME, CUSTOM_TEMPLATES.get(str(cfg.DATASET.NAME).upper(), "a photo of a {}."))]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for temp in self.templates:
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts_tok = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

            with torch.no_grad():
                if self.plot_angdist:
                    # only collect text layer vectors for the LAST template to avoid mixing buffers
                    self._txt_layer_vecs = []
                    self._collect_txt = True
                    self._current_text_tokens = prompts_tok

                text_features = self.clip_model.encode_text(prompts_tok)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                if self.plot_angdist:
                    self._collect_txt = False
                    self._current_text_tokens = None

            mean_text_features = mean_text_features + text_features

        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features

        if self.plot_angdist:
            self._plot_text_angdist()
