import json
import os.path as osp
import random
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import QuickGELU
from clip.model import convert_weights
from .imagenet_templates import IMAGENET_TEMPLATES
from collections import OrderedDict
import math
from trainers.regularizers import REGULARIZER_REGISTRY
_tokenizer = _Tokenizer()

CoPrompt_dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
    "APTOS": "aptos",
    "EYEPACS": "eyepacs",
    "MESSIDOR": "messidor",
    "MESSIDOR_2": "messidor_2",
    "PanNuke": "pannuke",
    "KatherColon": "kather",
    "DigestPath": "digestpath",
    "RSNA18": "rsna18",
    "Covid": "covid",
}

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.HICROPLReason.TEACHER_NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Return original CLIP model for generating frozen VL features
    design_details = {"trainer": 'IVLP',
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
        design_details = {"trainer": 'HiCroPLReason',
                          "vision_depth": cfg.TRAINER.HICROPL.PROMPT_DEPTH,
                          "language_depth": cfg.TRAINER.HICROPL.PROMPT_DEPTH,
                          "vision_ctx": cfg.TRAINER.HICROPL.N_CTX,
                          "language_ctx": cfg.TRAINER.HICROPL.N_CTX}
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def build_frozen_text_x(self, tokenized_prompts):
        with torch.no_grad():
            tokenized_prompts = tokenized_prompts.to(self.token_embedding.weight.device)
            raw = self.token_embedding(tokenized_prompts).type(self.dtype)
            raw = raw + self.positional_embedding.type(self.dtype)
            raw = raw.permute(1, 0, 2)   # NLD -> LND
        return  raw    #raw.detach()

    def _text_get_prompt_tokens(self, x, n_ctx):
        """
        x: [77, n_cls, 512]
        return prompt slots: [n_ctx, n_cls, 512]
        """
        return x[1:1 + n_ctx, :, :]


    def _text_replace_prompt_tokens(self, x, prompt_tokens, n_ctx):
        """
        Replace text prompt slots with refined tokens.
        x: [77, n_cls, 512]
        prompt_tokens: [n_ctx, n_cls, 512] or [n_ctx, 512]
        """
        if prompt_tokens.dim() == 2:
            prompt_tokens = prompt_tokens.unsqueeze(1).expand(-1, x.shape[1], -1)

        prompt_tokens = prompt_tokens.to(dtype=x.dtype, device=x.device)

        prefix = x[:1, :, :]
        suffix = x[1 + n_ctx:, :, :]
        return torch.cat([prefix, prompt_tokens, suffix], dim=0)


    def _run_block_payload(
        self,
        blk,
        payload,
        disable_prompt_injection=False,
        prompt_override=None,
    ):
        """
        Run one transformer block with optional recursive-control flags.
        """
        block_input = {
            "x": payload[0],
            "cross_prompts_deeper": payload[1],
            "frozen_x": payload[2],
            "frozen_contexts_list": payload[3],
            "init_payload": payload[4],
            "disable_prompt_injection": disable_prompt_injection,
            "prompt_override": prompt_override,
        }

        block_output = blk(block_input)

        return [
            block_output["x"],
            block_output["cross_prompts_deeper"],
            block_output["frozen_x"],
            block_output["frozen_contexts_list"],
            block_output["init_payload"],
        ]    

    """def forward(self, prompts, tokenized_prompts, cross_prompts_text_deeper,  init_cross_prompts_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # frozen branch: direct CLIP token embeddings, no learnable ctx inserted
        frozen_x = self.build_frozen_text_x(tokenized_prompts).to(device=x.device, dtype=x.dtype)
        #combined = [x, cross_prompts_text_deeper]
        #outputs = self.transformer(combined)
        outputs = self.transformer([x, cross_prompts_text_deeper, frozen_x,None,init_cross_prompts_text])
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x"""
    """def forward(
        self,
        prompts,
        tokenized_prompts,
        cross_prompts_text_deeper,
        init_cross_prompts_text,
        init_cross_prompts_visual,
        return_layer_states=False
    ):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        frozen_x = self.build_frozen_text_x(tokenized_prompts).to(
            device=x.device, dtype=x.dtype
        )

        init_payload = (init_cross_prompts_text, init_cross_prompts_visual)
        payload = [x, cross_prompts_text_deeper, frozen_x, None, init_payload]

        layer_states = []
        for blk in self.transformer.resblocks:
            payload = blk(payload)
            layer_states.append(payload[0])   # [L, Ncls, 512]

        x = payload[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        if return_layer_states:
            return x, layer_states

        return x"""
    
    """def forward(
        self,
        prompts,
        tokenized_prompts,
        cross_prompts_text_deeper,
        init_cross_prompts_text,
        init_cross_prompts_visual,
        return_layer_states=False
    ):"""
    def forward(
    self,
    prompts,
    tokenized_prompts,
    cross_prompts_text_deeper,
    init_cross_prompts_text,
    init_cross_prompts_visual,
    return_layer_states=False,
    ):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        frozen_x = self.build_frozen_text_x(tokenized_prompts).to(
            device=x.device, dtype=x.dtype
        )

        init_payload = (init_cross_prompts_text, init_cross_prompts_visual)
        payload = [x, cross_prompts_text_deeper, frozen_x, None, init_payload]

        layer_inputs = []
        layer_outputs = []

        for blk in self.transformer.resblocks:
            layer_inputs.append(payload[0])   # pre-block tokens
            payload = blk(payload)
            layer_outputs.append(payload[0])  # post-block tokens

        x = payload[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        if return_layer_states:
            return x, layer_inputs, layer_outputs, frozen_x

        return x

class MLPAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, zero_last=False):
        super().__init__()

        #if hidden_dim is None:
        #    hidden_dim = max(in_dim, out_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        if zero_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    @property
    def weight(self):
        return self.net[-1].weight

    @property
    def bias(self):
        return self.net[-1].bias

    def forward(self, x):
        return self.net(x)
    
class RecursiveMLPAdapter(nn.Module):
    """
    Recursive Token Mapper adapted for CommonPromptTransformer.

    This version does NOT use token-mixer + channel-mixer.
    Instead, it uses the existing MLPAdapter as the shared recursive block f.

    Input:
        x: [T, in_dim] or [T, N, in_dim]

    Output:
        delta: [T, out_dim] or [T, N, out_dim]

    RTM-style logic:
        z0 = input projection
        z_L = fast inner state
        z_H = slow outer state

        for H cycles:
            for L cycles:
                z_L = z_L + f(z_L + z_H + z0)
            z_H = z_H + f(z_H + z_L)

        output = readout(z_H)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        H_cycles: int = 4,
        L_cycles: int = 1,
        refinement_steps: int = 1,
        short_grad: bool = True,
        update_scale_init: float = -4.0,
        output_scale_init: float = -3.0,
        zero_last: bool = False,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.H_cycles = int(H_cycles)
        self.L_cycles = int(L_cycles)
        self.refinement_steps = int(refinement_steps)
        self.short_grad = bool(short_grad)

        # Project common prompt tokens into recursive state space.
        # For text:   common_dim 512 -> text_dim 512
        # For vision: common_dim 512 -> vision_dim 768
        self.input_proj = MLPAdapter(
            in_dim=in_dim,
            out_dim=out_dim,
            dropout=dropout,
            zero_last=False,
        )

        # Shared recursive block f.
        # This is reused at every H/L cycle.
        self.shared_f = MLPAdapter(
            in_dim=out_dim,
            out_dim=out_dim,
            dropout=dropout,
            zero_last=False,
        )

        # Final readout from refined z_H to output delta.
        self.readout = MLPAdapter(
            in_dim=out_dim,
            out_dim=out_dim,
            dropout=dropout,
            zero_last=zero_last,
        )

        # Small learnable scales for stable recursive updates.
        self.update_scale = nn.Parameter(torch.tensor(float(update_scale_init)))
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

    @property
    def weight(self):
        # Keeps compatibility with your _init_stable() logic.
        return self.readout.weight

    @property
    def bias(self):
        return self.readout.bias

    def _recursive_update(self, z_H, z_L, z0):
        """
        One RTM-style outer cycle:
            L inner updates for z_L
            1 outer update for z_H
        """
        scale = torch.sigmoid(self.update_scale).to(
            dtype=z0.dtype,
            device=z0.device,
        )

        for _ in range(self.L_cycles):
            z_L = z_L + self.shared_f(z_L + z_H + z0) #z_L +  self.shared_f(z_L + z_H + z0)

        z_H = z_H + self.shared_f(z_H + z_L) #z_H + self.shared_f(z_H + z_L)

        return z_H, z_L

    def forward(self, x: torch.Tensor):
        squeeze_back = False

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [T, D] -> [T, 1, D]
            squeeze_back = True

        if x.dim() != 3:
            raise ValueError(
                f"RecursiveMLPAdapter expected [T, D] or [T, N, D], got {x.shape}"
            )

        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"RecursiveMLPAdapter expected input dim={self.in_dim}, "
                f"got {x.shape[-1]}"
            )

        # z0 is the injected original signal.
        z0 = self.input_proj(x)

        # Prompt-safe initialization:
        # use the projected input itself as the initial state.
        # This is more stable than random fixed carry vectors for CLIP prompts.
        z_H = z0
        z_L = torch.zeros_like(z0)

        for _ in range(self.refinement_steps):

            if self.short_grad and self.H_cycles >=1:
                # Short-gradient RTM:
                # earlier cycles refine the state without storing full graph.
                with torch.no_grad():
                    for _ in range(self.H_cycles - 1):
                        z_H, z_L = self._recursive_update(z_H, z_L, z0)

                # Explicit RTM-style detach before the final cycle
                #z_H = z_H.detach()
                #z_L = z_L.detach()
                
                # Final cycle keeps gradient.
                z_H, z_L = self._recursive_update(z_H, z_L, z0)

            else:
                # Full-gradient version.
                for _ in range(self.H_cycles):
                    z_H, z_L = self._recursive_update(z_H, z_L, z0)

                    # Optional: if you want Algorithm-1 style detach even without no_grad
                    #if self.short_grad and h < self.H_cycles - 1:
                    #    z_H = z_H.detach()
                    #    z_L = z_L.detach()

        delta = z_H #self.readout(z_H)

        """out_scale = torch.sigmoid(self.output_scale).to(
            dtype=delta.dtype,
            device=delta.device,
        )"""
        #delta = out_scale * delta

        if squeeze_back:
            delta = delta.squeeze(1)

        return delta

class TokenMixMLP(nn.Module):
    """
    MLP over sequence length, like MLP-Mixer token mixing.

    Input:
        x: [L, N, D]

    Output:
        x: [L, N, D]

    L = prompt token length, e.g. n_ctx
    N = batch-like dimension, usually 1 here
    D = feature/channel dimension
    """
    def __init__(self, seq_len: int, hidden_mult: float = 2.0, dropout: float = 0.0):
        super().__init__()

        hidden = max(seq_len, int(seq_len * hidden_mult))

        self.fc1 = nn.Linear(seq_len, hidden)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden, seq_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [L, N, D] -> [N, D, L]
        y = x.permute(1, 2, 0).contiguous()

        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)

        # [N, D, L] -> [L, N, D]
        return y.permute(2, 0, 1).contiguous()

class TokenMixOnlyAdapter(nn.Module):
    """
    TokenMixMLP-only adapter.

    It does NOT change feature dimension.
    
    Text:
        [n_ctx, 512] -> [n_ctx, 512]

    Vision:
        [n_ctx, 768] -> [n_ctx, 768]
    """

    def __init__(
        self,
        seq_len: int,
        dim: int,
        token_hidden_mult: float = 2.0,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        self.dim = int(dim)

        self.ln = nn.LayerNorm(dim)

        self.token_mixer = TokenMixMLP(
            seq_len=seq_len,
            hidden_mult=token_hidden_mult,
            dropout=dropout,
        )

        # Small learnable scale for stability
        self.scale = nn.Parameter(torch.tensor(-4.0))

    @property
    def weight(self):
        # For compatibility with your work_device/work_dtype logic
        return self.ln.weight

    @property
    def bias(self):
        return self.ln.bias

    def forward(self, x: torch.Tensor):
        """
        Accepts:
            x: [T, D]
            x: [T, N, D]

        Returns:
            delta: same shape as x
        """

        squeeze_back = False

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [T, D] -> [T, 1, D]
            squeeze_back = True

        if x.dim() != 3:
            raise ValueError(f"Expected [T, D] or [T, N, D], got {x.shape}")

        if x.shape[0] != self.seq_len:
            raise ValueError(
                f"Token length mismatch in TokenMixOnlyAdapter: "
                f"expected seq_len={self.seq_len}, got {x.shape[0]}"
            )

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Feature dimension mismatch in TokenMixOnlyAdapter: "
                f"expected dim={self.dim}, got {x.shape[-1]}"
            )

        delta = self.token_mixer(self.ln(x))

        # Stable scaled delta
        #delta = torch.sigmoid(self.scale).to(dtype=delta.dtype, device=delta.device) * delta

        if squeeze_back:
            delta = delta.squeeze(1)

        return delta


class PromptMixerBlock(nn.Module):
    """
    Token-mixing MLP + channel MLP.

    Input:
        x: [L, N, D]

    Output:
        x: [L, N, D]
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        out_dim:int,
        token_hidden_mult: float = 2.0,
        channel_hidden_mult: float = 4.0,
        dropout: float = 0.0,
        
    ):
        super().__init__()

        self.ln_tok = nn.LayerNorm(d_model)
        self.token_mixer = TokenMixMLP(
            seq_len=seq_len,
            hidden_mult=token_hidden_mult,
            dropout=dropout,
        )

        self.ln_chn = nn.LayerNorm(d_model)

        channel_hidden = max(d_model, int(d_model * channel_hidden_mult))

        self.channel_mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d_model, channel_hidden)),
            ("act", QuickGELU()),
            ("drop1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(channel_hidden, out_dim)),
            ("drop2", nn.Dropout(dropout)),
        ]))

    def forward(self, x: torch.Tensor):
        x = self.token_mixer(self.ln_tok(x))
        #x = self.channel_mlp(self.ln_chn(x)) #x + self.channel_mlp(self.ln_chn(x))
        return x

class TokenMixToVisionAdapter(nn.Module):
    """
    Uses TokenMixMLP on t_refined_common, then projects to vision space.

    Input:
        [n_ctx, 512]

    Output:
        [n_ctx, 768]
    """

    def __init__(
        self,
        seq_len: int,
        in_dim: int = 512,
        out_dim: int = 768,
        token_hidden_mult: float = 2.0,
        dropout: float = 0.0,
        zero_last: bool = True,
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.ln = nn.LayerNorm(in_dim)

        self.token_mixer = TokenMixMLP(
            seq_len=seq_len,
            hidden_mult=token_hidden_mult,
            dropout=dropout,
        )

        # This is required to go from 512 -> 768
        self.out_proj = nn.Linear(in_dim, out_dim)

        if zero_last:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @property
    def weight(self):
        return self.out_proj.weight

    @property
    def bias(self):
        return self.out_proj.bias

    def forward(self, x):
        """
        x: [T, 512] or [T, N, 512]
        """

        squeeze_back = False

        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_back = True

        if x.shape[0] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {x.shape[0]}"
            )

        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected input dim={self.in_dim}, got {x.shape[-1]}"
            )

        # Token mixing only over prompt-token dimension
        y = self.token_mixer(self.ln(x))
        # [T, N, 512]

        # Required channel projection into vision space
        y = self.out_proj(y)
        # [T, N, 768]

        if squeeze_back:
            y = y.squeeze(1)

        return y


class CommonToModalityTokenMixer(nn.Module):
    """
    Token-mixer output adapter for CommonPromptTransformer.

    It replaces:
        self.common_to_text
        self.common_to_vision

    It performs:
        [T, common_dim] -> token mixing in common space -> [T, out_dim]

    Why not raw TokenMixMLP only?
        TokenMixMLP preserves D.
        But common_to_vision must convert 512 -> 768.
        Therefore we need token mixing + final channel projection.
    """
    def __init__(
        self,
        seq_len: int,
        common_dim: int,
        out_dim: int,
        mixer_depth: int = 1,
        token_hidden_mult: float = 2.0,
        channel_hidden_mult: float = 4.0,
        dropout: float = 0.0,
        zero_last: bool = True,
    ):
        super().__init__()

        self.seq_len = int(seq_len)

        self.blocks = nn.ModuleList([
            PromptMixerBlock(
                seq_len=seq_len,
                d_model=common_dim,
                out_dim=out_dim,
                token_hidden_mult=token_hidden_mult,
                channel_hidden_mult=channel_hidden_mult,
                dropout=dropout,
                
            )
            for _ in range(mixer_depth)
        ])

        self.ln_out = nn.LayerNorm(common_dim)
        #self.out_proj = nn.Linear(common_dim, out_dim)

        # Important for stability.
        # The output adapter starts near zero, so it does not destroy CLIP prompts
        # during the first iterations.
        if zero_last:
            #nn.init.zeros_(self.out_proj.weight)
            #nn.init.zeros_(self.out_proj.bias)
            pass

    @property
    def weight(self):
        return self.out_proj.weight

    @property
    def bias(self):
        return self.out_proj.bias

    def forward(self, x: torch.Tensor):
        """
        Accepts:
            x: [T, D]
            x: [T, N, D]

        Returns:
            y: [T, out_dim]
            y: [T, N, out_dim]
        """

        squeeze_back = False

        if x.dim() == 2:
            # [T, D] -> [T, 1, D]
            x = x.unsqueeze(1)
            squeeze_back = True

        if x.dim() != 3:
            raise ValueError(f"Expected [T, D] or [T, N, D], got {x.shape}")

        if x.shape[0] != self.seq_len:
            raise ValueError(
                f"Token length mismatch in CommonToModalityTokenMixer: "
                f"expected seq_len={self.seq_len}, got {x.shape[0]}"
            )

        y = x

        for blk in self.blocks:
            y = blk(y)

        #y = self.out_proj(self.ln_out(y))

        if squeeze_back:
            y = y.squeeze(1)

        return y



# LKP
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(AttentionPooling, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        #print("token_query", token_query.shape,
        #"sequence_key", sequence_key.shape,
        #"sequence_value", sequence_value.shape)
        #token_query = token_query + self.attn(self.ln_1(token_query), self.ln_1(sequence_key), self.ln_1(sequence_value), need_weights=False)[0]
        #token_query = self.ln_2(token_query)
        #return token_query
        # Make tensors 3D: (L, N, E) where N=1
        squeeze_back = False
        if token_query.dim() == 2:
            token_query = token_query.unsqueeze(1)      # (1, 1, E)
            sequence_key = sequence_key.unsqueeze(1)    # (n_ctx, 1, E)
            sequence_value = sequence_value.unsqueeze(1)
            squeeze_back = True
        attn_out = self.attn(
            self.ln_1(token_query),
            self.ln_1(sequence_key),
            self.ln_1(sequence_value),
            need_weights=False
        )[0]    
        token_query = token_query + attn_out
        token_query = self.ln_2(token_query)

        if squeeze_back:
            token_query = token_query.squeeze(1)        # back to (1, E)

        return token_query
# Multi-scale Knowledge Mapper
"""class CrossPromptAttention(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super(CrossPromptAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        # hidden_size is Q's dimension, encoder_hidden_size is K, V's dimension
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size)
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_size * 4, hidden_size))
        ]))
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        q_proj = q_proj + self.attn(self.ln_1(q_proj), self.ln_1(k_proj), self.ln_1(v_proj), need_weights=False)[0]
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj"""

class FP16SafeLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        out = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor):
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, weight, bias)

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        squeeze_back = False

        if x.dim() == 2:
            x = x.unsqueeze(1)   # [L, E] -> [L, 1, E]
            squeeze_back = True

        out = self.attn(x, x, x, need_weights=False)[0]

        if squeeze_back:
            out = out.squeeze(1)  # [L, 1, E] -> [L, E]

        return out

    def forward(self, inputs):
        x = inputs
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossPromptAttention(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size,
                                          num_heads=num_attention_heads)  # expects (L,N,E)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size)
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size)
        #self.linear_q = CastedLinear(hidden_size, hidden_size, bias=True)
        #self.linear_k = CastedLinear(encoder_hidden_size, hidden_size, bias=True)
        #self.linear_v = CastedLinear(encoder_hidden_size, hidden_size, bias=True)


        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_size * 4, hidden_size)),
        ]))
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        # q: (Lq,E_q) or (Lq,N,E_q)
        # k,v: (Lk,E_k) or (Lk,N,E_k)

        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)

        # ---- add batch dim if unbatched (2D) ----
        squeeze_back = False
        if q_proj.dim() == 2:
            # convert (L,E) -> (L,1,E)
            q_proj = q_proj.unsqueeze(1)
            k_proj = k_proj.unsqueeze(1)
            v_proj = v_proj.unsqueeze(1)
            squeeze_back = True

        attn_out = self.attn(
            self.ln_1(q_proj),
            self.ln_1(k_proj),
            self.ln_1(v_proj),
            need_weights=False
        )[0]

        q_proj = q_proj + attn_out
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))

        # ---- remove batch dim back to 2D ----
        if squeeze_back:
            q_proj = q_proj.squeeze(1)  # (L,E)

        return q_proj


"""class CrossPromptAttention(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)

        self.linear_q = CastedLinear(hidden_size, hidden_size, bias=True)
        self.linear_k = CastedLinear(encoder_hidden_size, hidden_size, bias=True)
        self.linear_v = CastedLinear(encoder_hidden_size, hidden_size, bias=True)

        self.ln_q = FP16SafeLayerNorm(hidden_size)
        self.ln_k = FP16SafeLayerNorm(hidden_size)
        self.ln_v = FP16SafeLayerNorm(hidden_size)
        self.ln_2 = FP16SafeLayerNorm(hidden_size)

        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", CastedLinear(hidden_size, hidden_size * 4, bias=True)),
            ("gelu", QuickGELU()),
            ("c_proj", CastedLinear(hidden_size * 4, hidden_size, bias=True)),
        ]))

    def forward(self, q, k, v):
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)

        squeeze_back = False
        if q_proj.dim() == 2:
            q_proj = q_proj.unsqueeze(1)
            k_proj = k_proj.unsqueeze(1)
            v_proj = v_proj.unsqueeze(1)
            squeeze_back = True

        qn = self.ln_q(q_proj)
        kn = self.ln_k(k_proj)
        vn = self.ln_v(v_proj)

        attn_dtype = self.attn.in_proj_weight.dtype
        attn_device = self.attn.in_proj_weight.device
        qn = qn.to(dtype=attn_dtype, device=attn_device)
        kn = kn.to(dtype=attn_dtype, device=attn_device)
        vn = vn.to(dtype=attn_dtype, device=attn_device)

        attn_out = self.attn(qn, kn, vn, need_weights=False)[0]
        attn_out = attn_out.to(dtype=q_proj.dtype, device=q_proj.device)

        q_proj = q_proj + attn_out
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))

        if squeeze_back:
            q_proj = q_proj.squeeze(1)

        return q_proj"""

class CrossModalTRM(nn.Module):
    """
    Cross-modal prompt-level TRM.
    target prompt shape: [T, Dt]
    source proxy shape:  [S, Ds]   (usually S=1 here)
    """

    def __init__(self, target_dim, source_dim, num_heads=8, steps=1, warmup=0):
        super().__init__()
        self.steps = steps
        self.warmup = warmup

        # z update uses [x, y, z] as query, source proxy as K/V
        self.z_updater = CrossPromptAttention(
            hidden_size=target_dim,
            encoder_hidden_size=source_dim,
            num_attention_heads=num_heads,
        )

        # y update uses [y, z] as query, source proxy as K/V
        self.y_updater = CrossPromptAttention(
            hidden_size=target_dim,
            encoder_hidden_size=source_dim,
            num_attention_heads=num_heads,
        )

        self._ds_prev_y = None
        self._ds_prev_z = None

    def reset_ds_state(self):
        self._ds_prev_y = None
        self._ds_prev_z = None

    def _update_z(self, x, y, z, proxy):
        # x, y, z: [T, Dt], proxy: [S, Ds]
        T = x.shape[0]
        q = torch.cat([x, y, z], dim=0)       # [3T, Dt]
        out = self.z_updater(q, proxy, proxy) # [3T, Dt]
        return out[-T:]                       # [T, Dt]

    def _update_y(self, y, z, proxy):
        T = y.shape[0]
        q = torch.cat([y, z], dim=0)          # [2T, Dt]
        out = self.y_updater(q, proxy, proxy) # [2T, Dt]
        return out[:T]                        # [T, Dt]

    def _rollout(self, x, y, z, proxy, n_steps, grad_enabled):
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            for _ in range(n_steps):
                z = self._update_z(x, y, z, proxy)
            y = self._update_y(y, z, proxy)
        return y, z

    def forward(self, base_prompt, proxy):
        """
        base_prompt: [T, Dt]
        proxy:       [S, Ds]
        return:      [T, Dt]
        """
        if (
            self._ds_prev_y is not None
            and self._ds_prev_z is not None
            and self._ds_prev_y.shape == base_prompt.shape
            and self._ds_prev_z.shape == base_prompt.shape
        ):
            y = self._ds_prev_y.to(dtype=base_prompt.dtype, device=base_prompt.device)
            z = self._ds_prev_z.to(dtype=base_prompt.dtype, device=base_prompt.device)
        else:
            y = base_prompt.clone()
            z = torch.zeros_like(base_prompt) #base_prompt.clone()

        x = base_prompt

        if self.warmup > 0:
            for _ in range(self.warmup):
                y, z = self._rollout(x, y, z, proxy, self.steps, grad_enabled=False)
                #y = y.detach()
                #z = z.detach()

        if self.steps > 0:
            y, z = self._rollout(x, y, z, proxy, self.steps, grad_enabled=True)

        self._ds_prev_y = y.detach()
        self._ds_prev_z = z.detach()

        return y

class FrozenTextLayerContextEncoder(nn.Module):
    """
    Extract per-layer text contexts from a frozen no-prompt CLIP text encoder.
    Returns one [n_ctx, 512] tensor per prompt depth.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, tokenized_prompts, n_ctx: int, prompt_depth: int):
        # tokenized_prompts: [n_cls, 77]
        x = self.token_embedding(tokenized_prompts).type(self.dtype)   # [n_cls, 77, 512]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # [77, n_cls, 512]

        ctxs = []
        # shallow prompt location = positions 1 : 1+n_ctx
        #ctxs.append(x[1:1 + n_ctx].mean(dim=1))  # [n_ctx, 512]
        #ctxs.append(x)

        # deeper prompt locations = same slot positions before each next block
        for blk in self.transformer.resblocks:
            #if len(ctxs) >= prompt_depth:
            #    break
            x = blk(x)
            #ctxs.append(x[1:1 + n_ctx].mean(dim=1))  # [n_ctx, 512]
            ctxs.append(x)

        #return ctxs[:prompt_depth]
        return ctxs


class FrozenVisionLayerContextEncoder(nn.Module):
    """
    Extract per-layer visual contexts from a frozen no-prompt CLIP ViT encoder.
    Returns one [n_ctx, 768] tensor per prompt depth.

    Since the frozen visual encoder has no appended prompt slots, we use the
    last n_ctx tokens as the position-aligned proxy for the prompt insertion tail.
    """
    def __init__(self, visual):
        super().__init__()
        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.positional_embedding = visual.positional_embedding
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer
        self.dtype = visual.conv1.weight.dtype

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, image, n_ctx: int, prompt_depth: int):
        x = self.conv1(image.type(self.dtype))                 # [B, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)             # [B, 768, 196]
        x = x.permute(0, 2, 1)                                # [B, 196, 768]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )                                                     # [B, 197, 768]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                                # [197, B, 768]

        ctxs = []
        # shallow visual prompt is appended at the tail in the prompted model,
        # so use the tail of the frozen token sequence as aligned proxy
        #ctxs.append(x[-n_ctx:].mean(dim=1))                   # [n_ctx, 768]

        for blk in self.transformer.resblocks:
            #if len(ctxs) >= prompt_depth:
            #    break
            x = blk(x)
            #ctxs.append(x[-n_ctx:].mean(dim=1))
            ctxs.append(x)                # [n_ctx, 768]

        #return ctxs[:prompt_depth]
        return ctxs

class CrossModalMLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    
class CommonPromptTransformerBlock(nn.Module):
    """
    Shared transformer block used only for learnable prompt tokens.

    It does not receive full CLIP image/text tokens.
    It only processes projected prompt tokens in a common latent space.
    """

    def __init__(self, common_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.ln_1 = nn.LayerNorm(common_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        self.ln_2 = nn.LayerNorm(common_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(common_dim, common_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(common_dim * 4, common_dim)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [L, B, common_dim]
        Here B is normally 1 because prompt tokens are class-independent.
        """
        x_norm = self.ln_1(x)
        attn_out = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )[0]

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class CommonPromptTransformer(nn.Module):
    """
    AlignCLIP-style shared parameter path, but applied only to learnable prompts.

    Text prompt:
        [n_ctx, 512] -> text_to_common -> shared transformer -> common_to_text -> [n_ctx, 512]

    Vision prompt:
        [n_ctx, 768] -> vision_to_common -> same shared transformer -> common_to_vision -> [n_ctx, 768]

    This avoids destroying pretrained CLIP full-token features.
    """

    def __init__(
        self,
        text_dim: int = 512,
        vision_dim: int = 768,
        common_dim: int = 512,
        commin_dim_vis: int = 768,
        num_heads: int = 8,
        depth: int = 1,
        prompt_depth: int = 12,
        prompt_len: int = 2,
        target_layers=(3, 6, 10),
        after_layer: bool = False,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        gate_init: float = -3.0,
        mixer_depth: int = 1,
        token_hidden_mult: float = 2.0,
        channel_hidden_mult: float = 4.0,
    ):
        super().__init__()

        self.prompt_depth = int(prompt_depth)
        self.prompt_len = int(prompt_len)
        self.target_layers = [int(x) for x in target_layers]
        self.after_layer = bool(after_layer)
        self.residual_scale = float(residual_scale)

        # Modality-specific adapters into common space
        adapter_hidden = max(text_dim, vision_dim, common_dim)
        #self.text_to_common = nn.Linear(text_dim, common_dim)
        #self.vision_to_common = nn.Linear(vision_dim, common_dim)
        # Modality-specific adapters into common space
        self.text_to_common = MLPAdapter(
            in_dim=text_dim,
            out_dim=common_dim,
            dropout=dropout,
            zero_last=True,
        )

        self.vision_to_common = MLPAdapter(
            in_dim=vision_dim,
            out_dim=common_dim,
            dropout=dropout,
            zero_last=True,
        )

        # Shared transformer encoder over prompt tokens
        """self.blocks = nn.ModuleList([
            CommonPromptTransformerBlock(
                common_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(depth)
        ])"""

        # Convert selected human layer numbers into prompt indices
        self.target_prompt_indices = []
        for layer in self.target_layers:
            idx = layer + 1 if self.after_layer else layer
            if 0 <= idx < self.prompt_depth:
                self.target_prompt_indices.append(idx)

        self.target_prompt_indices = sorted(set(self.target_prompt_indices))

        # One layer-specific transformer stack per selected prompt layer
        self.layer_blocks = nn.ModuleDict({
            str(idx): nn.ModuleList([
                CommonPromptTransformerBlock(
                    common_dim=common_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth-3)
            ])
            for idx in self.target_prompt_indices
        })
        # One layer-specific transformer stack per selected prompt layer
        self.layer_blocks_vis = nn.ModuleDict({
            str(idx): nn.ModuleList([
                CommonPromptTransformerBlock(
                    common_dim=vision_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth-6)
            ])
            for idx in self.target_prompt_indices
        })
        """self.shared_blocks = nn.ModuleList([
            CommonPromptTransformerBlock(
                common_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(depth)
        ])"""

        # Modality-specific adapters back to original prompt dimensions
        #self.common_to_text = nn.Linear(common_dim, text_dim)
        #self.common_to_vision = nn.Linear(common_dim, vision_dim)
        
        # Modality-specific adapters back to original prompt dimensions
        """self.common_to_text = RecursiveMLPAdapter(
            in_dim=common_dim,
            out_dim=text_dim,
            dropout=dropout,
            H_cycles=1,
            L_cycles=1,
            refinement_steps=1,
            short_grad=True,
            update_scale_init=-4.0,
            output_scale_init=-3.0,
            zero_last=False,
        )"""

        """self.common_to_vision = RecursiveMLPAdapter(
            in_dim=common_dim,
            out_dim=vision_dim,
            dropout=dropout,
            H_cycles=1,
            L_cycles=1,
            refinement_steps=1,
            short_grad=True,
            update_scale_init=-4.0,
            output_scale_init=-3.0,
            zero_last=False,
        )"""
        
        self.common_to_text = MLPAdapter(
            in_dim=common_dim, #commin_dim_vis,
            out_dim=text_dim,
            dropout=dropout,
            zero_last=False,
        )
        self.common_to_txt_proj = nn.Linear(common_dim, text_dim)

        self.common_to_vision = MLPAdapter(
            in_dim=common_dim, #commin_dim_vis,
            out_dim=vision_dim,
            dropout=dropout,
            zero_last=False,
        )
        self.common_to_vis_proj = nn.Linear(common_dim, vision_dim)

        """self.common_to_text = CommonToModalityTokenMixer(
            seq_len=self.prompt_len,
            common_dim=common_dim,
            out_dim=text_dim,
            mixer_depth=1,
            token_hidden_mult=2.0,
            channel_hidden_mult=4.0,
            dropout=dropout,
            zero_last=True,
        )"""

        """self.common_to_vision = CommonToModalityTokenMixer(
            seq_len=self.prompt_len,
            common_dim=common_dim,
            out_dim=vision_dim,
            mixer_depth=1,
            token_hidden_mult=2.0,
            channel_hidden_mult=4.0,
            dropout=dropout,
            zero_last=True,
        )"""
        """self.common_to_text = TokenMixOnlyAdapter(
            seq_len=self.prompt_len,
            dim=text_dim,  # 512
            token_hidden_mult=token_hidden_mult,
            dropout=dropout,
        )

        self.common_to_vision = TokenMixToVisionAdapter(
            seq_len=self.prompt_len,
            in_dim=common_dim,      # 512
            out_dim=vision_dim,     # 768
            token_hidden_mult=token_hidden_mult,
            dropout=dropout,
            zero_last=True,
        )"""
                

        # Type embeddings tell the shared transformer which tokens are text vs vision
        self.text_type_embed = nn.Parameter(torch.zeros(1, common_dim))
        self.vision_type_embed = nn.Parameter(torch.zeros(1, common_dim))

        # Layer embeddings tell the transformer which prompt depth is being refined
        self.layer_embed = nn.Parameter(torch.zeros(prompt_depth, common_dim))

        # Learnable gate. Initial sigmoid(-3) ~= 0.047, so the prompt update starts small.
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        #self._init_stable()

    def _init_stable(self): #why this line
        """
        Small output projection initialization prevents the new module from
        strongly changing prompts at the first iteration.
        """
        nn.init.normal_(self.common_to_text.weight, std=1e-3)
        nn.init.zeros_(self.common_to_text.bias)

        nn.init.normal_(self.common_to_vision.weight, std=1e-3)
        nn.init.zeros_(self.common_to_vision.bias)

        nn.init.normal_(self.text_type_embed, std=1e-3)
        nn.init.normal_(self.vision_type_embed, std=1e-3)
        nn.init.normal_(self.layer_embed, std=1e-3)

    def _target_prompt_indices(self, n_prompts: int):
        """
        Converts user layer numbers into prompt-list indices.

        If after_layer=False:
            COMMON_PROMPT_LAYERS=[3,6,10] -> prompt indices [3,6,10]

        If after_layer=True:
            COMMON_PROMPT_LAYERS=[3,6,10] -> prompt indices [4,7,11]
            This is closer to the old 'after layer' hidden-state injection.
        """
        """indices = []

        for layer in self.target_layers:
            idx = layer + 1 if self.after_layer else layer

            if 0 <= idx < n_prompts:
                indices.append(idx)"""

        #return sorted(set(indices))
        return [idx for idx in self.target_prompt_indices if idx < n_prompts]

    def _refine_pair(
        self,
        text_prompt: torch.Tensor,
        vision_prompt: torch.Tensor,
        shared_prompt: torch.Tensor,
        prompt_index: int,
    ):
        """
        text_prompt:   [n_ctx, 512]
        vision_prompt: [n_ctx, 768]
        shared_prompt: [n_ctx, common_dim]
        """

        text_orig_dtype = text_prompt.dtype
        text_orig_device = text_prompt.device
        vision_orig_dtype = vision_prompt.dtype
        vision_orig_device = vision_prompt.device

        work_device = self.text_to_common.weight.device
        work_dtype = self.text_to_common.weight.dtype

        t = text_prompt.to(device=work_device, dtype=work_dtype)
        v = vision_prompt.to(device=work_device, dtype=work_dtype)
        s = shared_prompt.to(device=work_device, dtype=work_dtype)

        # Project to common shared space
        t_common = t #self.text_to_common(t)       # [n_ctx, common_dim]
        v_common = v #t #self.vision_to_common(v)     # [n_ctx, common_dim]

        layer_bias = self.layer_embed[prompt_index].unsqueeze(0) # why this

        t_common = t_common #+ self.text_type_embed + layer_bias
        v_common = v_common #+ self.vision_type_embed + layer_bias # why this line

        # Joint prompt sequence: text prompts + vision prompts
        # Shape for MultiheadAttention: [L, B, D]
        """if 0 <= prompt_index < 6:
            # shared attention
            joint_vis = t_common.unsqueeze(1) #torch.cat([t_common, v_common], dim=0).unsqueeze(1) #doubt at this line
            blocks_vis = self.layer_blocks_vis[str(prompt_index)]
            #y = joint.clone()
            #z = joint.clone()
            for blk in blocks_vis:
                    joint_vis = blk(joint_vis)"""
        #else:  
            # shared attention
        joint = t_common.unsqueeze(1) #  v_common.unsqueeze(1) #torch.cat([t_common, v_common], dim=0).unsqueeze(1) #t_common.unsqueeze(1) #torch.cat([t_common, v_common], dim=0).unsqueeze(1)  #t_common.unsqueeze(1) #torch.cat([t_common, v_common], dim=0).unsqueeze(1) #doubt at this line
        blocks = self.layer_blocks[str(prompt_index)]
        #y = joint.clone()
        #z = joint.clone()
        #blocks_vis = self.layer_blocks_vis[str(prompt_index)]
        #y = joint.clone()
        #z = joint.clone()
        #vision as commomn
        #for blk in blocks_vis:
        #        joint = blk(joint)
        #text as common
        for blk in blocks:
            joint = blk(joint)
                  
        #for blk in self.shared_blocks:
        #        joint = blk(joint)

        joint = joint.squeeze(1)
        #joint_vis = joint_vis.squeeze(1)
        n_text = t_common.shape[0]
        t_refined_common = joint[:n_text]
        #v_refined_common = joint[n_text:]
        #separate attention
        """t_joint = t_common.unsqueeze(1)  # [n_ctx, 1, common_dim]
        v_joint = v_common.unsqueeze(1)  # [n_ctx, 1, common_dim]

        blocks = self.layer_blocks[str(prompt_index)]

        for blk in blocks:
            t_joint = blk(t_joint)
            v_joint = blk(v_joint)

        t_refined_common = t_joint.squeeze(1)  # [n_ctx, common_dim]
        v_refined_common = v_joint.squeeze(1)  # [n_ctx, common_dim]"""

        # Project back to modality-specific prompt dimensions
        t_delta = self.common_to_txt_proj(t_refined_common) #self.common_to_text(t_refined_common)
        v_delta = self.common_to_vis_proj(t_refined_common) #self.common_to_vision(v_refined_common)

        gate = self.residual_scale * torch.sigmoid(self.gate)

        """refined_text = text_prompt + gate.to(
            dtype=text_orig_dtype,
            device=text_orig_device,
        ) * t_delta.to(dtype=text_orig_dtype, device=text_orig_device)"""

        refined_text = text_prompt + t_delta.to(dtype=text_orig_dtype, device=text_orig_device)
        #refined_text = t_delta.to(dtype=text_orig_dtype, device=text_orig_device)
        """refined_vision = vision_prompt + gate.to(
            dtype=vision_orig_dtype,
            device=vision_orig_device,
        ) * v_delta.to(dtype=vision_orig_dtype, device=vision_orig_device)"""

        refined_vision = vision_prompt + v_delta.to(dtype=vision_orig_dtype, device=vision_orig_device)
        #refined_vision = v_delta.to(dtype=vision_orig_dtype, device=vision_orig_device)
        return refined_text, refined_vision

    def forward(self, text_prompts, vision_prompts, shared_prompts):
        """
        text_prompts:   list of [n_ctx, 512]
        vision_prompts: list of [n_ctx, 768]

        Returns:
            refined_text_prompts
            refined_vision_prompts
        """

        assert len(text_prompts) == len(vision_prompts), (
            f"Text/vision prompt list length mismatch: "
            f"{len(text_prompts)} vs {len(vision_prompts)}"
        )

        refined_text = list(text_prompts)
        refined_vision = list(vision_prompts)
        refined_shared = list(shared_prompts)

        target_indices = self._target_prompt_indices(len(refined_text))

        for idx in target_indices:
            refined_text[idx], refined_vision[idx] = self._refine_pair(
                refined_text[idx],
                refined_vision[idx],
                refined_shared[idx],
                prompt_index=idx,
            )

        return refined_text, refined_vision


class CrossModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
                
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.HICROPL.PROMPT_DEPTH >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.HICROPL.N_CTX
        ctx_init = cfg.TRAINER.HICROPL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        v_dim = 768
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.cross_prompts_depth = cfg.TRAINER.HICROPL.PROMPT_DEPTH
        self.cross_layer = cfg.TRAINER.HICROPL.CROSS_LAYER
        # ---------------------------------------------------------
        # Prompt-only shared common transformer
        # ---------------------------------------------------------
        self.common_prompt_enable = cfg.TRAINER.HICROPLReason.COMMON_PROMPT_ENABLE

        if self.common_prompt_enable:
            self.common_prompt_transformer = CommonPromptTransformer(
                text_dim=ctx_dim,                         # usually 512
                vision_dim=v_dim,                         # usually 768 for ViT-B/16
                common_dim=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_DIM,
                num_heads=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_HEADS,
                depth=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_DEPTH,
                prompt_depth=self.cross_prompts_depth,
                prompt_len=n_ctx,
                target_layers=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_LAYERS,
                after_layer=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_AFTER_LAYER,
                dropout=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_DROPOUT,
                residual_scale=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_RESIDUAL_SCALE,
                gate_init=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_GATE_INIT,
                mixer_depth=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_MIXER_DEPTH,
                token_hidden_mult=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_TOKEN_HIDDEN_MULT,
                channel_hidden_mult=cfg.TRAINER.HICROPLReason.COMMON_PROMPT_CHANNEL_HIDDEN_MULT,
            )

            if cfg.TRAINER.HICROPL.PREC == "fp16":
                self.common_prompt_transformer = self.common_prompt_transformer.half()

            print(
                "[CommonPromptTransformer] enabled | "
                f"text_dim={ctx_dim}, vision_dim={v_dim}, "
                f"common_dim={cfg.TRAINER.HICROPLReason.COMMON_PROMPT_DIM}, "
                f"target_layers={cfg.TRAINER.HICROPLReason.COMMON_PROMPT_LAYERS}, "
                f"after_layer={cfg.TRAINER.HICROPLReason.COMMON_PROMPT_AFTER_LAYER}, "
                f"prompt_depth={self.cross_prompts_depth}"
            )
        else:
            self.common_prompt_transformer = None

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # ---- frozen no-prompt CLIP for per-layer context extraction ----
        clip_model_frozen_ctx = load_clip_to_cpu(cfg, True)
        if cfg.TRAINER.HICROPL.PREC in ["fp32", "amp"]:
            clip_model_frozen_ctx.float()
        else:
            clip_model_frozen_ctx.half()
        clip_model_frozen_ctx = clip_model_frozen_ctx.cuda()

        self.frozen_text_ctx_encoder = FrozenTextLayerContextEncoder(clip_model_frozen_ctx)
        self.frozen_visual_ctx_encoder = FrozenVisionLayerContextEncoder(clip_model_frozen_ctx.visual)

        self.textselfatt = SelfAttention(d_model= ctx_dim, n_head=8)
        self.visionselfatt = SelfAttention(d_model= v_dim, n_head=8)
        # ---- initialisation cross-attention blocks (same-modality) ----
        text_init_attn = CrossPromptAttention(
            hidden_size=ctx_dim,
            encoder_hidden_size=ctx_dim,
            num_attention_heads=8,
        )
        visual_init_attn = CrossPromptAttention(
            hidden_size=v_dim,
            encoder_hidden_size=v_dim,
            num_attention_heads=8,
        )

        self.text_init_attn_nets = _get_clones(text_init_attn, self.cross_prompts_depth)
        self.visual_init_attn_nets = _get_clones(visual_init_attn, self.cross_prompts_depth)

        if cfg.TRAINER.HICROPL.PREC == "fp16":
            self.text_init_attn_nets = self.text_init_attn_nets.half()
            self.visual_init_attn_nets = self.visual_init_attn_nets.half()

        ######## cross-modal text token initialization ########
        if ctx_init and (n_ctx) <= 4:
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
        print(f"HiCroPL design: Hierarchical Cross-modal Prompt Learning")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of HiCroPL context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        # create deeper prompts by nn.ParameterList
        cross_prompts_text = nn.ParameterList([self.ctx] + [nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype)) for _ in range(self.cross_prompts_depth - 1)])
        for single_para in cross_prompts_text[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_text = cross_prompts_text
        #shared parameters for deeper text prompts
        shared_cross_prompts_text = nn.ParameterList([self.ctx] + [nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype)) for _ in range(self.cross_prompts_depth - 1)])
        for single_para in shared_cross_prompts_text[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.shared_cross_prompts_text = shared_cross_prompts_text
        ######## cross-modal text token initialization end ########

        ######## cross-modal visual token initialization ########
        visual_vectors = torch.empty(n_ctx, v_dim, dtype=dtype)
        nn.init.normal_(visual_vectors, std=0.02)
        cross_prompts_visual = nn.ParameterList([nn.Parameter(visual_vectors) for _ in range(self.cross_prompts_depth)])
        self.cross_prompts_visual = cross_prompts_visual
        ######## cross-modal visual token initialization end ########

        ######## knowledge mapper network and LKP network initialization ########
        self.text2visual_net = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8)
        self.visual2text_net = CrossPromptAttention(hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8)
        if cfg.TRAINER.HICROPL.PREC == "fp16":
            self.text2visual_net, self.visual2text_net = self.text2visual_net.half(), self.visual2text_net.half()

        attn_pooling_text = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets = _get_clones(attn_pooling_text, self.cross_layer)
        attn_pooling_visual = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_nets = _get_clones(attn_pooling_visual, self.cross_prompts_depth - self.cross_layer)
        text_proxy_token = torch.randn(1, ctx_dim, dtype=dtype)
        self.text_proxy_token = nn.ParameterList([nn.Parameter(text_proxy_token) for _ in range(self.cross_layer)])
        visual_proxy_token = torch.randn(1, v_dim, dtype=dtype)
        self.visual_proxy_token = nn.ParameterList([nn.Parameter(visual_proxy_token) for _ in range(self.cross_layer, self.cross_prompts_depth)])
        if cfg.TRAINER.HICROPL.PREC == "fp16":
            self.attn_pooling_text_nets, self.attn_pooling_visual_nets = self.attn_pooling_text_nets.half(), self.attn_pooling_visual_nets.half()
        ######## knowledge mapper network and LKP network initialization end ########

        ######## preparation for distillation ########
        # visual
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu_teacher(cfg, True)
        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Make teacher visual dtype consistent with training precision
            if cfg.TRAINER.HICROPL.PREC in ["fp32", "amp"]:
                self.ZS_image_encoder.float()
            else:  # "fp16"
                self.ZS_image_encoder.half()

            self.ZS_image_encoder.eval()
            for p in self.ZS_image_encoder.parameters():
                p.requires_grad_(False)
        # text
        with open(f"gpt_file/{CoPrompt_dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_temp, cfg.DATASET.NAME
        )
        self.fixed_embeddings = clip_weights
        # Tokenized GPT prompts for zero-shot text-context extraction
        self.gpt_tokenized_prompts, self.gpt_prompt_strings = gpt_tokenized_prompts(
            classnames=classnames,
            gpt3_prompt=gpt3_prompt,
            dataset_name=cfg.DATASET.NAME,
            mode="first",   # or "random"
        )
        ######## preparation for distillation end ########

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # construct the text, a photo of a <class>.

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn): [n_cls, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, n_tkn, n_dim]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names 
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor, [n_cls, 77]
        self.name_lens = name_lens
        self.t2v_trm_blocks = nn.ModuleList([
            CrossModalTRM(
            target_dim=v_dim,      # visual prompt dim = 768
            source_dim=ctx_dim,    # text proxy dim = 512
            num_heads=8,
            steps=1,
            warmup=1,
            )
            for _ in range(self.cross_layer)
        ])

        self.v2t_trm_blocks = nn.ModuleList([
            CrossModalTRM(
                target_dim=ctx_dim,    # text prompt dim = 512
                source_dim=v_dim,      # visual proxy dim = 768
                num_heads=8,
                steps=1,
                warmup=1,
            )
            for _ in range(self.cross_prompts_depth - self.cross_layer)
        ])

        # ------------------------------------------------------------
        # Layerwise cross-modal alignment settings
        # ------------------------------------------------------------
        self.align_start = cfg.TRAINER.HICROPLReason.ALIGN_START
        self.align_end = cfg.TRAINER.HICROPLReason.ALIGN_END
        self.layer_align_lambda = cfg.TRAINER.HICROPLReason.ALIGN_LAMBDA #cfg.TRAINER.HICROPL.ALIGN_LAMBDA

        #assert 0 <= self.align_start <= self.align_end <= self.cross_prompts_depth, \
        #    f"ALIGN_START/ALIGN_END must be within [0, {self.cross_prompts_depth - 1}]"

        self.align_layers = list(range(self.align_start, self.align_end + 1))

        t2v_hidden = cfg.TRAINER.HICROPLReason.ALIGN_T2V_HIDDEN
        v2t_hidden = cfg.TRAINER.HICROPLReason.ALIGN_V2T_HIDDEN
        align_dropout = cfg.TRAINER.HICROPLReason.ALIGN_DROPOUT

        self.layer_text_to_vision_mlps = nn.ModuleList([
            CrossModalMLPProjector(
                in_dim=ctx_dim,      # 512
                out_dim=v_dim,       # 768
                hidden_dim=t2v_hidden,
                dropout=align_dropout,
            )
            for _ in self.align_layers
        ])

        self.layer_vision_to_text_mlps = nn.ModuleList([
            CrossModalMLPProjector(
                in_dim=v_dim,        # 768
                out_dim=ctx_dim,     # 512
                hidden_dim=v2t_hidden,
                dropout=align_dropout,
            )
            for _ in self.align_layers
        ])

        if cfg.TRAINER.HICROPL.PREC == "fp16":
            self.layer_text_to_vision_mlps = self.layer_text_to_vision_mlps.half()
            self.layer_vision_to_text_mlps = self.layer_vision_to_text_mlps.half()

        #if cfg.TRAINER.HICROPL.PREC == "fp16":
        #    self.t2v_trm_blocks = self.t2v_trm_blocks.half()
        #    self.v2t_trm_blocks = self.v2t_trm_blocks.half()
        #self.z_L_prev = None
        #self.z_H_prev = None
        #self.z_Lv_prev = None
        #self.z_Hv_prev = None

    """def reset_ds_statetxt(self):
        self.z_L_prev = None
        self.z_H_prev = None

    def reset_ds_statevision(self):
        self.z_Lv_prev = None
        self.z_Hv_prev = None"""
    def _build_frozen_initialized_prompts(self, image):
        """
        Build per-depth prompt initialisations using:
        query = learnable prompt init
        key/value = frozen same-layer encoder context
        """
        #tokenized_prompts = self.tokenized_prompts.to(self.ctx.device)
        tokenized_prompts = self.gpt_tokenized_prompts.to(self.ctx.device)
        # frozen same-layer contexts
        frozen_text_ctxs = self.frozen_text_ctx_encoder(
            tokenized_prompts=tokenized_prompts,
            n_ctx=self.n_ctx,
            prompt_depth=self.cross_prompts_depth,
        )   # list of [n_ctx, 512]

        frozen_visual_ctxs = self.frozen_visual_ctx_encoder(
            image=image,
            n_ctx=self.n_ctx,
            prompt_depth=self.cross_prompts_depth,
        )   # list of [n_ctx, 768]
        return frozen_text_ctxs, frozen_visual_ctxs

        """init_cross_prompts_text = []
        init_cross_prompts_visual = []

        for d in range(self.cross_prompts_depth):
            # query = learnable prompt init
            q_text = self.cross_prompts_text[d]      # [n_ctx, 512]
            q_vis = self.cross_prompts_visual[d]     # [n_ctx, 768]

            # key/value = same-layer frozen encoder context
            kv_text = frozen_text_ctxs[d].to(dtype=q_text.dtype, device=q_text.device)
            kv_vis = frozen_visual_ctxs[d].to(dtype=q_vis.dtype, device=q_vis.device)

            # initialise prompt with same-modality cross-attention
            init_text_d = self.text_init_attn_nets[d](q_text, kv_text, kv_text)      # [n_ctx, 512]
            init_vis_d = self.visual_init_attn_nets[d](q_vis, kv_vis, kv_vis)        # [n_ctx, 768]

            init_cross_prompts_text.append(init_text_d)
            init_cross_prompts_visual.append(init_vis_d)

        return init_cross_prompts_text, init_cross_prompts_visual"""


    def reset_ds_state(self):
        for blk in self.t2v_trm_blocks:
            blk.reset_ds_state()
        for blk in self.v2t_trm_blocks:
            blk.reset_ds_state()         


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

    def forward(self,image):
        # ----- SAME-MODALITY FROZEN-CONTEXT INITIALISATION -----
        init_cross_prompts_text, init_cross_prompts_visual = self._build_frozen_initialized_prompts(image)
        #self attention text
        #print("cross_prompts_text len", len(self.cross_prompts_text))
        text_selfatt_cross = []
        for i in range(len(self.cross_prompts_text)):
            #print("shape text",self.cross_prompts_text[i].shape)
            text_att = self.textselfatt(self.cross_prompts_text[i])
            text_selfatt_cross.append(text_att)
        #print("text_selfatt_cross len", len(text_selfatt_cross))
        #self attention vision
        #print("cross_prompts_visual len", len(self.cross_prompts_visual))
        vision_selfatt_cross = []
        for i in range(len(self.cross_prompts_visual)):
            #print("shape vision",self.cross_prompts_visual[i].shape)
            vis_att = self.visionselfatt(self.cross_prompts_visual[i])
            vision_selfatt_cross.append(vis_att)  
        #print("vision_selfatt_cross len", len(vision_selfatt_cross))         
        # first layer text token
        ctx =  self.cross_prompts_text[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, 4, 512]"""
        #zero-shot crossattent
        """ctx = init_cross_prompts_text[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)"""
        prefix = self.token_prefix
        suffix = self.token_suffix
        # construct first layer text input
        text_input = self.construct_prompts(ctx, prefix, suffix)  # [n_cls, 77, 512]

        ######## T->I mapping ########
        visual_prompts = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0)  # [self.cross_layer, n_ctx, 768]
        text_prompts =  torch.cat([self.cross_prompts_text[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0)  # [self.cross_layer, n_ctx, 512]
        #visual_prompts = torch.cat([vision_selfatt_cross[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0)  # [self.cross_layer, n_ctx, 768]
        #text_prompts =  torch.cat([text_selfatt_cross[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0) 
        #zero-shot crossattent
        #visual_prompts = torch.stack([init_cross_prompts_visual[i] for i in range(self.cross_layer)], dim=0)
        #text_prompts = torch.stack([init_cross_prompts_text[i] for i in range(self.cross_layer)], dim=0)
        visual_prompts_copy = visual_prompts.clone()
        

         
        # LKP's work
        proxy_text_tokens = []
        for i in range(self.cross_layer):
            # For T->I mapping, the text prompts should be compressed, text_proxy_token as Q, cross_prompts_text[i] as K, V.
            text_proxy_token = self.attn_pooling_text_nets[i](
                token_query=self.text_proxy_token[i],  # [1, ctx_dim]
                sequence_key= self.cross_prompts_text[i], #init_cross_prompts_text[i],  #self.cross_prompts_text[i],  # [n_ctx, ctx_dim]
                sequence_value= self.cross_prompts_text[i] #init_cross_prompts_text[i] #self.cross_prompts_text[i]  # [n_ctx, ctx_dim]
            )
            proxy_text_tokens.append(text_proxy_token)
        proxy_text_prompts = torch.cat(proxy_text_tokens, dim=0)  # [self.cross_layer, 1, ctx_dim]
        visual_prompts = visual_prompts.view(-1, visual_prompts.shape[-1])  # [self.cross_layer * n_ctx, 768]
        #text_selfatt = torch.cat(text_selfatt_cross, dim=0) 
        proxy_text_prompts = proxy_text_prompts.view(-1, proxy_text_prompts.shape[-1])  # [self.cross_layer, 512]
        #text_selfatt = text_selfatt.view(-1, text_selfatt.shape[-1]) 
        # cross modal action for [0: self.cross_layer]: T->I
        #trm---block---
        # -------- T -> I mapping (correct TRM) --------
        updated_visual_prompt_list = []

        for i in range(self.cross_layer):
            base_visual_i = self.cross_prompts_visual[i] #init_cross_prompts_visual[i] #self.cross_prompts_visual[i]   # [n_ctx, 768]
            proxy_text_i = proxy_text_tokens[i]            # [1, 512]

            refined_visual_i = self.t2v_trm_blocks[i](base_visual_i, proxy_text_i)
            updated_visual_prompt_list.append(refined_visual_i)

        updated_visual_prompts = torch.stack(updated_visual_prompt_list, dim=0)  # [cross_layer, n_ctx, 768]
        #updated_visual_prompts= visual_prompts
        #updated_visual_prompts = self.text2visual_net(visual_prompts, proxy_text_prompts, proxy_text_prompts)  # [self.cross_layer * n_ctx, 768]
        #updated_visual_prompts = updated_visual_prompts.view(self.cross_layer, -1, updated_visual_prompts.shape[-1])  # [self.cross_layer, n_ctx, 768]
        #for i in range(self.cross_layer):
        #    self.cross_prompts_visual[i].data.copy_(updated_visual_prompts[i])
            #vision_selfatt_cross[i].data.copy_(updated_visual_prompts[i])

        #trm block end------------
        #recursive reasoning: updated_visual_prompts are also used for updating text prompts in the next layer
        """updated_visual_prompts=visual_prompts
        if (self.z_L_prev is not None) and (self.z_H_prev is not None):
                z_L= self.z_L_prev
                z_H= self.z_H_prev
        else:
                z_L= visual_prompts.clone()
                z_H= visual_prompts.clone()
        #with torch.no_grad():
        for h in range(2): #higher recursive
                    for l in range(2): #lower recursive
                        z_L = self.text2visual_net(z_L, proxy_text_prompts, proxy_text_prompts)
                    z_H = self.text2visual_net(z_L, proxy_text_prompts, proxy_text_prompts) # think in different way to update this        
        #for l in range(2): #lower recursive
        #        z_L = self.text2visual_net(z_L+z_H, proxy_text_prompts, proxy_text_prompts)
        #z_H = self.text2visual_net(z_L, proxy_text_prompts, proxy_text_prompts)
        z_H = self.text2visual_net(z_H, proxy_text_prompts, proxy_text_prompts)
        self.z_L_prev = z_L.detach()
        self.z_H_prev = z_H.detach()
        updated_visual_prompts=z_H+updated_visual_prompts
        updated_visual_prompts = self.text2visual_net(updated_visual_prompts, proxy_text_prompts, proxy_text_prompts)        
        #end of recursive reasoning: updated_visual_prompts are not used for updating text prompts in the next layer
        #hicropl visual prompt----------------------
        #updated_visual_prompts = self.text2visual_net(visual_prompts, proxy_text_prompts, proxy_text_prompts)  # [self.cross_layer * n_ctx, 768]
        updated_visual_prompts = updated_visual_prompts.view(self.cross_layer, -1, updated_visual_prompts.shape[-1])  # [self.cross_layer, n_ctx, 768]
        for i in range(self.cross_layer):
            self.cross_prompts_visual[i].data.copy_(updated_visual_prompts[i])"""
        ######## T->I mapping end ########

        ######## I->T mapping ########
        text_prompts = torch.cat([self.cross_prompts_text[i].unsqueeze(0) for i in range(self.cross_layer, self.cross_prompts_depth)], dim=0)  # [all_layer - self.cross_layer, n_ctx, 512]
        visual_prompts = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in range(self.cross_layer, self.cross_prompts_depth)], dim=0)  # [all_layer - self.cross_layer, n_ctx, 768]
        #text_prompts = torch.cat([text_selfatt_cross[i].unsqueeze(0) for i in range(self.cross_layer, self.cross_prompts_depth)], dim=0)  # [all_layer - self.cross_layer, n_ctx, 512]
        #visual_prompts = torch.cat([vision_selfatt_cross[i].unsqueeze(0) for i in range(self.cross_layer, self.cross_prompts_depth)], dim=0)
        #zero shot
        """text_prompts = torch.stack(
            [init_cross_prompts_text[i] for i in range(self.cross_layer, self.cross_prompts_depth)],
            dim=0
        )
        visual_prompts = torch.stack(
            [init_cross_prompts_visual[i] for i in range(self.cross_layer, self.cross_prompts_depth)],
            dim=0
        )"""
        text_prompts_copy = text_prompts.clone()
        # LKP's work
        proxy_visual_tokens = []
        for i in range(self.cross_layer, self.cross_prompts_depth):
           
            # For I->T mapping, the visual prompts should be compressed, visual_proxy_token as Q, cross_prompts_visual[i] as K, V.
            visual_proxy_token = self.attn_pooling_visual_nets[i - self.cross_layer](
                
                token_query=self.visual_proxy_token[i - self.cross_layer],  # [1, v_dim]
                sequence_key= self.cross_prompts_visual[i], #self.cross_prompts_visual[i], #init_cross_prompts_visual[i],   #self.cross_prompts_visual[i],  # [n_ctx, v_dim]
                sequence_value= self.cross_prompts_visual[i] #self.cross_prompts_visual[i] #init_cross_prompts_visual[i] #self.cross_prompts_visual[i]  # [n_ctx, v_dim]
            )
            proxy_visual_tokens.append(visual_proxy_token)
            proxy_visual_prompts = torch.cat(proxy_visual_tokens, dim=0)  # [self.cross_prompts_depth - self.cross_layer, 1, v_dim]
        text_prompts = text_prompts.view(-1, text_prompts.shape[-1])  # [(all_layer - self.cross_layer) * n_ctx, 512]
        proxy_visual_prompts = proxy_visual_prompts.view(-1, proxy_visual_prompts.shape[-1])  # [(all_layer - self.cross_layer) * n_ctx, 768]
        #visual_selfatt =  torch.cat(vision_selfatt_cross, dim=0)
        #visual_selfatt =  visual_selfatt.view(-1, visual_selfatt.shape[-1])
        # cross modal action for [0: self.cross_layer]: I->T
        #trm block start---
        # -------- I -> T mapping (correct TRM) --------
        updated_text_prompt_list = []

        for j in range(self.cross_prompts_depth - self.cross_layer):
            layer_idx = self.cross_layer + j
           
            base_text_j = self.cross_prompts_text[layer_idx] #init_cross_prompts_text[layer_idx] #self.cross_prompts_text[layer_idx]   # [n_ctx, 512]
            proxy_visual_j = proxy_visual_tokens[j]            # [1, 768]

            refined_text_j = self.v2t_trm_blocks[j](base_text_j, proxy_visual_j)
            updated_text_prompt_list.append(refined_text_j)

        updated_text_prompts = torch.stack(updated_text_prompt_list, dim=0)  # [depth-cross_layer, n_ctx, 512]
        #updated_text_prompts = text_prompts
        #updated_text_prompts = self.visual2text_net(text_prompts, proxy_visual_prompts, proxy_visual_prompts)  # [(all_layer - self.cross_layer) * n_ctx, 512]
        #updated_text_prompts = updated_text_prompts.view(self.cross_prompts_depth - self.cross_layer, -1, updated_text_prompts.shape[-1])  # [self.cross_prompts_depth - self.cross_layer, n_ctx, 512]
        #for i in range(self.cross_layer, self.cross_prompts_depth):
        #    self.cross_prompts_text[i].data.copy_(updated_text_prompts[i - self.cross_layer])
            #text_selfatt_cross[i].data.copy_(updated_text_prompts[i - self.cross_layer])
        # trm block end------------
        #recursive reasoning: updated_visual_prompts are also used for updating text prompts in the next layer
        """updated_text_prompts=text_prompts
        if (self.z_Lv_prev is not None) and (self.z_Hv_prev is not None):
                z_Lv= self.z_Lv_prev
                z_Hv= self.z_Hv_prev
        else:
                z_Lv= text_prompts.clone()
                z_Hv= text_prompts.clone()
        #with torch.no_grad():
        for h in range(2): #higher recursive
                    for l in range(2): #lower recursive
                        z_Lv = self.visual2text_net(z_Lv, proxy_visual_prompts, proxy_visual_prompts)
                    z_Hv = self.visual2text_net(z_Lv, proxy_visual_prompts, proxy_visual_prompts) # think in different way to update this        
        #for l in range(2): #lower recursive
        #        z_Lv = self.visual2text_net(z_Lv+z_Hv,proxy_visual_prompts, proxy_visual_prompts)
        #z_Hv = self.visual2text_net(z_Lv, proxy_visual_prompts, proxy_visual_prompts)
        z_Hv = self.visual2text_net(z_Hv, proxy_visual_prompts, proxy_visual_prompts)
        self.z_Lv_prev = z_Lv.detach()
        self.z_Hv_prev = z_Hv.detach()
        updated_text_prompts=z_Hv+updated_text_prompts
        updated_text_prompts = self.visual2text_net(updated_text_prompts, proxy_visual_prompts, proxy_visual_prompts)
        #end of recursive reasoning: updated_visual_prompts are not used for updating text prompts in the next layer        
        #updated_text_prompts = self.visual2text_net(text_prompts, proxy_visual_prompts, proxy_visual_prompts)  # [(all_layer - self.cross_layer) * n_ctx, 512]
        updated_text_prompts = updated_text_prompts.view(self.cross_prompts_depth - self.cross_layer, -1, updated_text_prompts.shape[-1])  # [self.cross_prompts_depth - self.cross_layer, n_ctx, 512]
        for i in range(self.cross_layer, self.cross_prompts_depth):
            self.cross_prompts_text[i].data.copy_(updated_text_prompts[i - self.cross_layer])"""
        ######## I->T mapping end ########
        #After both refinements, assemble the full prompt lists instead of overwriting parameters:
        # full visual prompt list
        final_cross_prompts_visual = []
        for i in range(self.cross_prompts_depth):
            if i < self.cross_layer:
                final_cross_prompts_visual.append(updated_visual_prompts[i])
            else:
                #final_cross_prompts_visual.append(vision_selfatt_cross[i])
                final_cross_prompts_visual.append(self.cross_prompts_visual[i])
                #final_cross_prompts_visual.append(init_cross_prompts_visual[i]) # ablation: without updating deeper visual prompts
        # full text prompt list
        final_cross_prompts_text = []
        for i in range(self.cross_prompts_depth):
            if i < self.cross_layer:
                #final_cross_prompts_text.append(text_selfatt_cross[i])
                final_cross_prompts_text.append(self.cross_prompts_text[i])
                #final_cross_prompts_text.append(init_cross_prompts_text[i])
            else:
                final_cross_prompts_text.append(updated_text_prompts[i - self.cross_layer])
        #end 
        # extract deeper prompts
        #cross_prompts_text_deeper = [self.cross_prompts_text[i] for i in range(1, len(self.cross_prompts_text))]
        #cross_prompts_visual_deeper = [self.cross_prompts_visual[i] for i in range(1, len(self.cross_prompts_visual))]
        #cross_prompts_text_deeper = [text_selfatt_cross[i] for i in range(1, len(text_selfatt_cross))]
        #cross_prompts_visual_deeper = [vision_selfatt_cross[i] for i in range(1, len(vision_selfatt_cross))]
        #cross_prompts_text_deeper = final_cross_prompts_text[1:] #[self.cross_prompts_text[i] for i in range(1, len(self.cross_prompts_text))]
        #cross_prompts_visual_deeper = final_cross_prompts_visual[1:] #[self.cross_prompts_visual[i] for i in range(1, len(self.cross_prompts_visual))]
        #return text_input, self.cross_prompts_visual[0], cross_prompts_text_deeper, cross_prompts_visual_deeper, init_cross_prompts_text, init_cross_prompts_visual
        #return text_input, self.cross_prompts_visual[0], cross_prompts_text_deeper, cross_prompts_visual_deeper
        #return text_input, final_cross_prompts_visual[0], final_cross_prompts_text, final_cross_prompts_visual
        #return text_input, final_cross_prompts_visual[0], cross_prompts_text_deeper, cross_prompts_visual_deeper, init_cross_prompts_text, init_cross_prompts_visual

        # ---------------------------------------------------------
        # Prompt-only common transformer refinement
        # ---------------------------------------------------------
        # At this point, final_cross_prompts_text and final_cross_prompts_visual
        # contain the actual prompts that will be injected into the respective encoders.
        # We refine only these prompt tensors, not full CLIP hidden tokens.
        if self.common_prompt_transformer is not None:
            final_cross_prompts_text, final_cross_prompts_visual = self.common_prompt_transformer(
                self.cross_prompts_text, #final_cross_prompts_text,
                self.cross_prompts_visual, #final_cross_prompts_visual,
                self.shared_cross_prompts_text,
            )

        # ---------------------------------------------------------
        # Rebuild shallow text input using refined text prompt 0
        # ---------------------------------------------------------
        ctx = final_cross_prompts_text[0]

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix

        text_input = self.construct_prompts(ctx, prefix, suffix)  # [n_cls, 77, 512]

        # ---------------------------------------------------------
        # Extract refined deeper prompts
        # ---------------------------------------------------------
        cross_prompts_text_deeper = final_cross_prompts_text[1:]
        cross_prompts_visual_deeper = final_cross_prompts_visual[1:]

        return (
            text_input,
            final_cross_prompts_visual[0],
            cross_prompts_text_deeper,
            cross_prompts_visual_deeper,
            init_cross_prompts_text,
            init_cross_prompts_visual,
        )


class CommonSharedEncoderBlock(nn.Module):
    """
    One transformer encoder block shared by both text and vision modalities.

    Input shape:
        x: [L, N, D_common]

    This is intentionally modality-agnostic:
        - text enters after projection 512 -> common_dim
        - vision enters after projection 768 -> common_dim
    """

    def __init__(self, common_dim=512, num_heads=8, dropout=0.0):
        super().__init__()

        self.ln_1 = nn.LayerNorm(common_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        self.ln_2 = nn.LayerNorm(common_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(common_dim, common_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(common_dim * 4, common_dim)),
        ]))

    def forward(self, x):
        # x: [L, N, common_dim]
        x_norm = self.ln_1(x)
        attn_out = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )[0]
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class CommonSharedEncoder(nn.Module):
    """
    Shared encoder injected into both text and vision streams after selected layers.

    This solves the dimension mismatch problem:
        text tokens:   512 -> common_dim -> 512
        vision tokens: 768 -> common_dim -> 768

    The transformer block itself is shared between modalities.
    """

    def __init__(
        self,
        text_dim=512,
        vision_dim=768,
        common_dim=512,
        num_heads=8,
        injection_layers=(3, 6, 10),
        dropout=0.0,
        residual_scale=0.1,
    ):
        super().__init__()

        self.injection_layers = set(int(x) for x in injection_layers)
        self.residual_scale = residual_scale

        # Modality-specific input adapters
        self.text_to_common = nn.Linear(text_dim, common_dim)
        self.vision_to_common = nn.Linear(vision_dim, common_dim)

        # One shared transformer block per injection position.
        # Each block is shared by text and vision at that layer.
        self.shared_blocks = nn.ModuleDict({
            str(layer): CommonSharedEncoderBlock(
                common_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for layer in self.injection_layers
        })

        # Modality-specific output adapters
        self.common_to_text = nn.Linear(common_dim, text_dim)
        self.common_to_vision = nn.Linear(common_dim, vision_dim)

        self._init_output_adapters()

    def _init_output_adapters(self):
        """
        Small initialization prevents the new shared encoder from destroying
        pretrained CLIP features at the first iteration.
        """
        nn.init.normal_(self.common_to_text.weight, std=1e-4)
        nn.init.zeros_(self.common_to_text.bias)

        nn.init.normal_(self.common_to_vision.weight, std=1e-4)
        nn.init.zeros_(self.common_to_vision.bias)

    def forward(self, x, layer_number: int, modality: str):
        """
        x:
            text   -> [77, n_cls, 512]
            vision -> [197 + n_ctx, batch, 768]

        layer_number:
            human-readable layer number, e.g. 3, 6, 10
        """

        layer_number = int(layer_number)

        if layer_number not in self.injection_layers:
            return x

        orig_dtype = x.dtype
        orig_device = x.device

        if modality == "text":
            in_proj = self.text_to_common
            out_proj = self.common_to_text
        elif modality == "vision":
            in_proj = self.vision_to_common
            out_proj = self.common_to_vision
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Keep the shared encoder numerically stable.
        # Usually this module will be fp32 even if CLIP is fp16.
        work_dtype = in_proj.weight.dtype
        work_device = in_proj.weight.device

        y = x.to(device=work_device, dtype=work_dtype)
        y = in_proj(y)
        y = self.shared_blocks[str(layer_number)](y)
        y = out_proj(y)

        y = y.to(device=orig_device, dtype=orig_dtype)

        # Residual injection
        #return y
        return x + self.residual_scale * y
    
class ClassNormalizedOrthogonalProbeLoss(nn.Module):
    """
    Class-Normalized Orthogonal Probe Distillation Loss.

    Purpose:
        1. Create one normalized text probe per class.
        2. Create one normalized vision probe per class.
        3. Pull tuned text class features to text probes.
        4. Pull tuned image features to their class vision probes.
        5. Anchor probes to frozen CLIP features to preserve pretrained geometry.
        6. Align same-class text/vision probes.
        7. Push different-class probes apart.

    Expected input shapes:
        text_features:        [C, D]
        image_features:       [B, D]
        fixed_text_features:  [C, D]
        fixed_image_features: [B, D]
        labels:               [B]
    """

    def __init__(
        self,
        init_text_features: torch.Tensor,
        anchor_weight: float = 1.0,
        pair_weight: float = 0.5,
        orth_weight: float = 0.05,
        orth_margin: float = 0.05,
        init_std: float = 0.001,
        eps: float = 1e-7,
    ):
        super().__init__()

        init_text_features = init_text_features #F.normalize(init_text_features.float(), dim=-1, eps=eps)
        num_classes, embed_dim = init_text_features.shape

        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.vis_dim = int(embed_dim * 1.5)  # vision probes can have higher dimension than text probes
        self.anchor_weight = float(anchor_weight)
        self.pair_weight = float(pair_weight)
        self.orth_weight = float(orth_weight)
        self.orth_margin = float(orth_margin)
        self.eps = float(eps)

        # Initialize probes from frozen text embeddings.
        # This gives the probes pretrained CLIP semantic geometry at epoch 0.
        text_init = torch.empty(self.num_classes, self.embed_dim, dtype=init_text_features.dtype) #init_std * torch.randn_like(init_text_features)
        nn.init.normal_(text_init, std=0.02)
        vision_init = torch.empty(self.num_classes, self.embed_dim, dtype=init_text_features.dtype) #init_std * torch.randn_like(init_text_features) # doubt in this initialization
        nn.init.normal_(vision_init, std=0.02)

        self.text_probes = nn.Parameter(text_init)
        self.vision_probes = nn.Parameter(vision_init)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1, eps=self.eps)

    def get_normalized_probes(self):
        text_probes = self.text_probes #self._norm(self.text_probes)
        vision_probes = self.vision_probes #self._norm(self.vision_probes)
        return text_probes, vision_probes

    def _off_diagonal_hinge_loss(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        Penalize only off-diagonal similarities larger than margin.

        sim_matrix: [C, C]
        diagonal = same class
        off diagonal = different classes
        """
        C = sim_matrix.shape[0]
        eye = torch.eye(C, device=sim_matrix.device, dtype=torch.bool)
        off_diag = sim_matrix.masked_select(~eye)

        if off_diag.numel() == 0:
            return torch.zeros((), device=sim_matrix.device, dtype=sim_matrix.dtype)

        return F.relu(off_diag - self.orth_margin).pow(2).mean()

    def _orthogonal_probe_loss(
        self,
        text_probes: torch.Tensor,
        vision_probes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Orthogonal separation among different classes.

        Includes:
            text-text off-class separation
            vision-vision off-class separation
            text-vision off-class separation
        """
        sim_tt = text_probes @ text_probes.t()
        sim_vv = vision_probes @ vision_probes.t()
        sim_tv = text_probes @ vision_probes.t()

        loss_tt = self._off_diagonal_hinge_loss(sim_tt)
        loss_vv = self._off_diagonal_hinge_loss(sim_vv)
        loss_tv = self._off_diagonal_hinge_loss(sim_tv)

        return (loss_tt + loss_vv + loss_tv) / 3.0

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        fixed_text_features: torch.Tensor,
        fixed_image_features: torch.Tensor,
        labels: torch.Tensor,
    ):
        labels = labels.long()

        text_features = text_features #self._norm(text_features)
        image_features = image_features #self._norm(image_features)

        fixed_text_features = fixed_text_features #self._norm(fixed_text_features.detach())
        fixed_image_features = fixed_image_features #self._norm(fixed_image_features.detach())

        text_probes, vision_probes = self.get_normalized_probes()

        # ---------------------------------------------------------
        # 1. Tuned feature -> class probe alignment
        # ---------------------------------------------------------
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        loss_text_align = 1.0 - torch.mean(cos(text_features, text_probes)) #1.0 - (text_features * text_probes).sum(dim=-1).mean()
        batch_vision_probes = vision_probes[labels]
        loss_image_align = 1.0 - torch.mean(cos(image_features, batch_vision_probes)) #1.0 - (image_features * batch_vision_probes).sum(dim=-1).mean()

        loss_align = loss_text_align + loss_image_align

        # ---------------------------------------------------------
        # 2. Frozen CLIP feature -> probe anchor
        # This keeps probes close to pretrained CLIP geometry.
        # ---------------------------------------------------------
        loss_text_anchor = 1.0 - torch.mean(cos(fixed_text_features, text_probes)) #1.0 - (fixed_text_features * text_probes).sum(dim=-1).mean()

        batch_fixed_image = fixed_image_features
        batch_vision_probes = vision_probes[labels]
        loss_image_anchor = 1.0 - torch.mean(cos(batch_fixed_image, batch_vision_probes)) #1.0 - (batch_fixed_image * batch_vision_probes).sum(dim=-1).mean()

        loss_anchor = loss_text_anchor + loss_image_anchor

        # ---------------------------------------------------------
        # 3. Same-class text probe <-> vision probe alignment
        # ---------------------------------------------------------
        loss_pair = 1.0 - (text_probes * vision_probes).sum(dim=-1).mean()

        # ---------------------------------------------------------
        # 4. Different-class orthogonal separation
        # ---------------------------------------------------------
        loss_orth = self._orthogonal_probe_loss(text_probes, vision_probes)

        # ---------------------------------------------------------
        # Final probe loss
        # ---------------------------------------------------------
        """loss_probe = (
            loss_align
            + self.anchor_weight * loss_anchor
            + self.pair_weight * loss_pair
            + self.orth_weight * loss_orth
        )"""

        loss_probe = (
            loss_align + loss_anchor 
        )

        loss_dict = {
            "loss_probe": loss_probe.detach(),
            "loss_probe_align": loss_align.detach(),
            "loss_probe_text_align": loss_text_align.detach(),
            "loss_probe_image_align": loss_image_align.detach(),
            "loss_probe_anchor": loss_anchor.detach(),
            "loss_probe_pair": loss_pair.detach(),
            "loss_probe_orth": loss_orth.detach(),
        }

        return loss_probe, loss_dict    
    
class ContrastiveFeatureDistillationLoss(nn.Module):
    """
    Contrastive distillation between tuned features and frozen CLIP features.

    Text:
        tuned text class feature should match its frozen text class feature.
        Other frozen text class features are negatives.

    Image:
        tuned image feature should match the frozen image prototype of its class.
        Frozen prototypes of other classes in the batch are negatives.

    Expected shapes:
        text_features:        [C, D]
        image_features:       [B, D]
        fixed_text_features:  [C, D]
        fixed_image_features: [B, D]
        labels:               [B]
    """

    def __init__(
        self,
        temperature: float = 0.07,
        text_weight: float = 1.0,
        image_weight: float = 1.0,
        symmetric_text: bool = True,
        symmetric_image: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__()

        self.temperature = float(temperature)
        self.text_weight = float(text_weight)
        self.image_weight = float(image_weight)
        self.symmetric_text = bool(symmetric_text)
        self.symmetric_image = bool(symmetric_image)
        self.eps = float(eps)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1, eps=self.eps)

    def _text_contrastive_loss(
        self,
        text_features: torch.Tensor,
        fixed_text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        text_features:       [C, D]
        fixed_text_features: [C, D]

        Positive pairs are diagonal elements.
        Negatives are off-diagonal class pairs.
        """
        text_features = text_features #self._norm(text_features)
        fixed_text_features = fixed_text_features #self._norm(fixed_text_features.detach())

        C = text_features.shape[0]
        targets = torch.arange(C, device=text_features.device)

        logits = text_features @ fixed_text_features.t()
        logits = logits / self.temperature

        loss_tuned_to_fixed = F.cross_entropy(logits, targets)

        if self.symmetric_text:
            loss_fixed_to_tuned = F.cross_entropy(logits.t(), targets)
            loss = 0.5 * (loss_tuned_to_fixed + loss_fixed_to_tuned)
        else:
            loss = loss_tuned_to_fixed

        return loss

    def _build_batch_fixed_image_prototypes(
        self,
        fixed_image_features: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Build class prototypes from frozen image features inside the batch.

        fixed_image_features: [B, D]
        labels:               [B]

        Returns:
            prototypes:        [K, D]
            target_indices:    [B]
            unique_labels:     [K]

        K = number of unique classes in the current batch.
        """
        labels = labels.long()
        unique_labels = torch.unique(labels, sorted=True)

        prototypes = []
        for cls in unique_labels:
            mask = labels == cls
            proto = fixed_image_features[mask].mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)

        # Map each label to its index in unique_labels
        target_indices = torch.empty_like(labels)
        for idx, cls in enumerate(unique_labels):
            target_indices[labels == cls] = idx

        return prototypes, target_indices, unique_labels

    def _image_class_contrastive_loss(
        self,
        image_features: torch.Tensor,
        fixed_image_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Class-aware image contrastive distillation.

        Positive:
            tuned image feature -> frozen image prototype of same class

        Negatives:
            tuned image feature -> frozen image prototypes of other classes
            appearing in the batch.
        """
        labels = labels.long()

        image_features = image_features #self._norm(image_features)
        fixed_image_features = fixed_image_features #self._norm(fixed_image_features.detach())

        prototypes, target_indices, unique_labels = self._build_batch_fixed_image_prototypes(
            fixed_image_features=fixed_image_features,
            labels=labels,
        )

        prototypes = self._norm(prototypes)

        # If the batch contains only one class, there is no negative class.
        # In that case, fall back to direct cosine consistency for images.
        if prototypes.shape[0] <= 1:
            loss = 1.0 - (image_features * prototypes[target_indices]).sum(dim=-1).mean()
            return loss

        logits = image_features @ prototypes.t()
        logits = logits / self.temperature

        loss_img_to_proto = F.cross_entropy(logits, target_indices)

        if self.symmetric_image:
            # Prototype-to-image direction with multi-positive handling.
            # Usually not needed. Keep False first for stability.
            proto_logits = logits.t()  # [K, B]

            losses = []
            for proto_idx, cls in enumerate(unique_labels):
                positive_mask = labels == cls
                log_prob = F.log_softmax(proto_logits[proto_idx], dim=0)
                losses.append(-log_prob[positive_mask].mean())

            loss_proto_to_img = torch.stack(losses).mean()
            loss = 0.5 * (loss_img_to_proto + loss_proto_to_img)
        else:
            loss = loss_img_to_proto

        return loss

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        fixed_text_features: torch.Tensor,
        fixed_image_features: torch.Tensor,
        labels: torch.Tensor,
    ):
        loss_text = self._text_contrastive_loss(
            text_features=text_features,
            fixed_text_features=fixed_text_features,
        )

        loss_image = self._image_class_contrastive_loss(
            image_features=image_features,
            fixed_image_features=fixed_image_features,
            labels=labels,
        )

        loss =  loss_text + loss_image

        loss_dict = {
            "loss_contrastive_distill": loss.detach(),
            "loss_contrastive_text": loss_text.detach(),
            "loss_contrastive_image": loss_image.detach(),
        }

        return loss, loss_dict
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRelationalCLIPDistillLoss(nn.Module):
    """
    Adaptive Relational Cross-Modal Consistency Loss.

    It improves simple cosine distillation by preserving:
    1. tuned-to-frozen angular consistency
    2. same-modality relational geometry
    3. cross-modal dark class distribution

    Expected shapes:
        text_features:        [C, D]
        image_features:       [B, D]
        fixed_text_features:  [C, D]
        fixed_image_features: [B, D]
        labels:               [B]
    """

    def __init__(
        self,
        tau_rel=0.07,
        tau_xmodal=0.07,
        anchor_margin=0.20,   # radians, about 11.5 degrees
        w_anchor=0.25,
        w_rel=1.0,
        w_xmodal=1.0,
        min_teacher_weight=0.20,
        eps=1e-7,
    ):
        super().__init__()

        self.tau_rel = tau_rel
        self.tau_xmodal = tau_xmodal
        self.anchor_margin = anchor_margin

        self.w_anchor = w_anchor
        self.w_rel = w_rel
        self.w_xmodal = w_xmodal

        self.min_teacher_weight = min_teacher_weight
        self.eps = eps

    def _norm(self, x):
        return F.normalize(x.float(), dim=-1, eps=self.eps)

    def _adaptive_anchor_loss(self, z, z_fixed):
        """
        Penalize only when tuned features move outside an allowed angular margin.
        This avoids over-constraining prompt tuning.
        """
        z = self._norm(z)
        z_fixed = self._norm(z_fixed.detach())

        cos = (z * z_fixed).sum(dim=-1)
        cos = cos.clamp(-1.0 + self.eps, 1.0 - self.eps)

        target_cos = math.cos(self.anchor_margin)

        loss = F.relu(target_cos - cos).pow(2).mean()
        return loss

    def _relational_kl_loss(self, z, z_fixed):
        """
        Preserve pairwise similarity structure inside one modality.
        For text:  class-class geometry
        For image: image-image geometry inside the batch
        """
        z = z #self._norm(z)
        z_fixed = z_fixed #self._norm(z_fixed.detach())

        n = z.shape[0]

        if n <= 1:
            return z.new_tensor(0.0)

        sim_student = z @ z.t()
        sim_teacher = z_fixed @ z_fixed.t()

        #sim_student = sim_student / self.tau_rel
        #sim_teacher = sim_teacher / self.tau_rel

        # remove self-similarity diagonal
        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        sim_student = sim_student.masked_fill(eye, -1e4)
        sim_teacher = sim_teacher.masked_fill(eye, -1e4)

        log_p_student = F.log_softmax(sim_student, dim=1)
        p_teacher = F.softmax(sim_teacher.detach(), dim=1)

        loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        return loss

    def _cross_modal_dark_kl_loss(
        self,
        text_features,
        image_features,
        fixed_text_features,
        fixed_image_features,
        labels,
    ):
        """
        Preserve frozen CLIP image-to-text semantic distribution.

        We mask the ground-truth class, so this loss focuses on dark knowledge:
        which negative classes are semantically close according to frozen CLIP.
        """
        text_features = self._norm(text_features)
        image_features = self._norm(image_features)

        fixed_text_features = self._norm(fixed_text_features.detach())
        fixed_image_features = self._norm(fixed_image_features.detach())

        B = image_features.shape[0]
        C = text_features.shape[0]

        if C <= 1:
            return image_features.new_tensor(0.0)

        student_logits = image_features @ text_features.t()
        teacher_logits = fixed_image_features @ fixed_text_features.t()

        student_logits = student_logits / self.tau_xmodal
        teacher_logits = teacher_logits / self.tau_xmodal

        # teacher confidence weighting
        with torch.no_grad():
            teacher_prob_full = F.softmax(teacher_logits, dim=1)
            teacher_weight = teacher_prob_full.gather(
                1, labels.view(-1, 1)
            ).squeeze(1)
            teacher_weight = teacher_weight.clamp(min=self.min_teacher_weight)

        # mask positive class; preserve only negative-class dark knowledge
        mask = torch.ones_like(student_logits, dtype=torch.bool)
        mask[torch.arange(B, device=labels.device), labels] = False

        student_neg = student_logits.masked_fill(~mask, -1e4)
        teacher_neg = teacher_logits.masked_fill(~mask, -1e4)

        log_p_student = F.log_softmax(student_neg, dim=1)
        p_teacher = F.softmax(teacher_neg.detach(), dim=1)

        loss_per_sample = F.kl_div(
            log_p_student,
            p_teacher,
            reduction="none",
        ).sum(dim=1)

        loss = (teacher_weight * loss_per_sample).sum() / (
            teacher_weight.sum() + self.eps
        )

        return loss

    def forward(
        self,
        text_features,
        image_features,
        fixed_text_features,
        fixed_image_features,
        labels,
    ):
        fixed_text_features = fixed_text_features.to(
            device=text_features.device,
            dtype=text_features.dtype,
        )

        fixed_image_features = fixed_image_features.to(
            device=image_features.device,
            dtype=image_features.dtype,
        )

        # 1. adaptive feature anchor
        loss_anchor_text = self._adaptive_anchor_loss(
            text_features,
            fixed_text_features,
        )

        loss_anchor_image = self._adaptive_anchor_loss(
            image_features,
            fixed_image_features,
        )

        loss_anchor = loss_anchor_text + loss_anchor_image

        # 2. same-modality relational geometry
        loss_rel_text = self._relational_kl_loss(
            text_features,
            fixed_text_features,
        )

        loss_rel_image = self._relational_kl_loss(
            image_features,
            fixed_image_features,
        )

        loss_rel = loss_rel_text + loss_rel_image

        # 3. cross-modal dark knowledge
        loss_xmodal = self._cross_modal_dark_kl_loss(
            text_features=text_features,
            image_features=image_features,
            fixed_text_features=fixed_text_features,
            fixed_image_features=fixed_image_features,
            labels=labels,
        )

        """loss = (
            self.w_anchor * loss_anchor
            + self.w_rel * loss_rel
            + self.w_xmodal * loss_xmodal
        )"""
        loss  = loss_rel
        logs = {
            "loss_arcmc": loss.detach(),
            "loss_arcmc_anchor": loss_anchor.detach(),
            "loss_arcmc_anchor_text": loss_anchor_text.detach(),
            "loss_arcmc_anchor_image": loss_anchor_image.detach(),
            "loss_arcmc_rel": loss_rel.detach(),
            "loss_arcmc_rel_text": loss_rel_text.detach(),
            "loss_arcmc_rel_image": loss_rel_image.detach(),
            "loss_arcmc_xmodal": loss_xmodal.detach(),
        }

        return loss, logs    

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CrossModalPromptLearner(cfg, classnames, clip_model)
        # ---------------------------------------------------------
        # Class-Normalized Orthogonal Probe Distillation
        # Keep this module inside prompt_learner so your existing
        # requires_grad filtering will train it automatically.
        # ---------------------------------------------------------
        probe_cfg = cfg.TRAINER.HICROPLReason

        self.probe_enable = bool(probe_cfg.PROBE_ENABLE)
        self.probe_lambda = float(probe_cfg.PROBE_LAMBDA)
        # ---------------------------------------------------------
        # Point-wise Orthogonal Consistency Loss
        # ---------------------------------------------------------
        self.poc_enable = True #bool(probe_cfg.POC_ENABLE)
        self.poc_lambda = 1.0 #float(probe_cfg.POC_LAMBDA)

        # ---------------------------------------------------------
        # DAPT Eq. (13): dataset-level visual prototype intra loss
        # ---------------------------------------------------------
        self.dapt_intra_enable = bool(probe_cfg.DAPT_INTRA_ENABLE)
        self.dapt_intra_lambda = 1.0#float(probe_cfg.DAPT_INTRA_LAMBDA)
        self.dapt_intra_mode = "one_minus_cos" #'one_minus_cos' #"l2_cos" #str(probe_cfg.DAPT_INTRA_MODE)

        # This will be filled after computing prototypes from the full training dataset.
        # Shape after initialization: [num_classes, feature_dim]
        self.register_buffer("dapt_visual_prototypes", torch.empty(0), persistent=False)

        self.poc_text_point_weight = 1.0 #float(probe_cfg.POC_TEXT_POINT_WEIGHT)
        self.popc_text_point_weight = 1.0 #float(probe_cfg.POPC_TEXT_POINT_WEIGHT)
        self.popc_text_orth_weight = 1.0 #float(probe_cfg.POPC_TEXT_ORTH_WEIGHT)
        self.poc_text_orth_weight = 1.0 #float(probe_cfg.POC_TEXT_ORTH_WEIGHT)

        self.poc_image_point_weight = 1.0 #float(probe_cfg.POC_IMAGE_POINT_WEIGHT)
        self.popc_image_point_weight = 1.0 #float(probe_cfg.POPC_IMAGE_POINT_WEIGHT)
        self.popc_image_cluster_weight = 1.0 #float(probe_cfg.POPC_IMAGE_CLUSTER_WEIGHT)
        self.popc_image_orth_weight = 1.0 #float(probe_cfg.POPC_IMAGE_ORTH_WEIGHT)
        self.poc_image_orth_weight = 1.0 #float(probe_cfg.POC_IMAGE_ORTH_WEIGHT)
        # Final image-loss scaling
        self.popc_image_loss_weight = 1.0 #float(probe_cfg.POPC_IMAGE_LOSS_WEIGHT)
        self.arcmc_loss = AdaptiveRelationalCLIPDistillLoss(
            tau_rel=0.07,
            tau_xmodal=0.07,
            anchor_margin=0.20,
            w_anchor=0.25,
            w_rel=1.0,
            w_xmodal=1.0,
        )
        self.arcmc_lambda = 1.0
        self.contrastive_distill_loss = ContrastiveFeatureDistillationLoss(
            temperature=0.07,
            text_weight=1.0,
            image_weight=1.0,
            symmetric_text=False,
            symmetric_image=False,
        ) 
        if self.probe_enable:
            self.prompt_learner.probe_loss = ClassNormalizedOrthogonalProbeLoss(
                init_text_features=self.prompt_learner.fixed_embeddings,
                anchor_weight=probe_cfg.PROBE_ANCHOR_WEIGHT,
                pair_weight=probe_cfg.PROBE_PAIR_WEIGHT,
                orth_weight=probe_cfg.PROBE_ORTH_WEIGHT,
                orth_margin=probe_cfg.PROBE_ORTH_MARGIN,
                init_std=probe_cfg.PROBE_INIT_STD,
            )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.lambd = cfg.TRAINER.HICROPL.LAMBD
        # AD loss weights
        self.ad_lambda = cfg.TRAINER.HICROPLReason.AD_LAMBDA
        self.ad_text_weight = cfg.TRAINER.HICROPLReason.AD_TEXT_WEIGHT
        self.ad_vision_weight = cfg.TRAINER.HICROPLReason.AD_VISION_WEIGHT
        self.trm_start_layer = cfg.TRAINER.HICROPLReason.TRM_START_LAYER
        self.trm_end_layer = cfg.TRAINER.HICROPLReason.TRM_END_LAYER
        self.trm_steps = cfg.TRAINER.HICROPLReason.TRM_STEPS
        # keep your existing cross-modal weight
        self.cm_lambda = 10.0

    def _safe_norm(self, x, eps=1e-7):
        """
        Normalize features in float32 for numerical stability.
        Returns float32 normalized features.
        """
        return F.normalize(x.float(), dim=-1, eps=eps)

    def _angular_pointwise_loss(self, z, z_fixed, eps=1e-7):
        """
        Point-wise angular consistency:
            tuned feature should stay close to its own frozen CLIP feature.

        This replaces:
            1 - cosine(tuned, frozen)

        with:
            arccos(cosine)^2

        This is a direct angular distance on the unit hypersphere.
        """
        z = self._safe_norm(z, eps=eps)
        z_fixed = self._safe_norm(z_fixed.detach(), eps=eps)

        cos_pos = torch.sum(z * z_fixed, dim=1)
        cos_pos = cos_pos.clamp(-1.0 + eps, 1.0 - eps)

        theta = torch.acos(cos_pos)

        return theta.pow(2).mean()


    def _text_pointwise_orth_loss(
        self,
        text_features,
        text_features_fixed,
        point_weight=1.0,
        orth_weight=0.5,
        eps=1e-7,
    ):
        """
        Text branch loss:
            1. tuned text feature stays close to corresponding frozen text feature
            2. different class text features become orthogonal

        Mathematical target:
            text_features @ text_features.T ≈ I
        """
        z = self._safe_norm(text_features, eps=eps)
        z_fixed = self._safe_norm(text_features_fixed.detach(), eps=eps)

        n_cls = z.shape[0]

        if n_cls <= 1:
            return z.new_tensor(0.0)

        # --------------------------------------------------
        # 1. Point-wise angular consistency
        # --------------------------------------------------
        cos_pos = torch.sum(z * z_fixed, dim=1)
        cos_pos = cos_pos.clamp(-1.0 + eps, 1.0 - eps)

        theta = torch.acos(cos_pos)
        loss_point = theta.pow(2).mean()

        # --------------------------------------------------
        # 2. Strict class orthogonality
        # --------------------------------------------------
        gram = z @ z.t()

        identity = torch.eye(
            n_cls,
            device=z.device,
            dtype=z.dtype,
        )

        # Since z is normalized, diagonal should be 1.
        # Off-diagonal should be 0.
        loss_orth = (gram - identity).pow(2).sum() / (
            n_cls * (n_cls - 1)
        )

        loss = loss_point #+ loss_orth

        return loss


    def _image_frozen_proto_cluster_orth_loss(
        self,
        image_features,
        image_features_fixed,
        labels,
        point_weight=1.0,
        cluster_weight=1.0,
        orth_weight=0.1,
        eps=1e-7,
    ):
        """
        Image branch loss.

        Main idea:
            1. Keep each tuned image feature close to its own frozen image feature.
            2. Build class prototypes from frozen image features in the current batch.
            3. Pull tuned image features toward their corresponding frozen-image class prototype.
            4. Build tuned image class centers.
            5. Make tuned image class centers orthogonal.

        This avoids using text features as image prototypes.
        """

        z = self._safe_norm(image_features, eps=eps)
        z_fixed = self._safe_norm(image_features_fixed.detach(), eps=eps)

        labels = labels.view(-1)
        batch_size = z.shape[0]

        if batch_size <= 1:
            return z.new_tensor(0.0)

        # --------------------------------------------------
        # 1. Point-wise frozen image consistency
        # --------------------------------------------------
        cos_point = torch.sum(z * z_fixed, dim=1)
        cos_point = cos_point.clamp(-1.0 + eps, 1.0 - eps)

        theta_point = torch.acos(cos_point)
        loss_point = theta_point.pow(2).mean()

        # --------------------------------------------------
        # 2. Create frozen image class prototypes from batch
        # --------------------------------------------------
        unique_labels, inverse = labels.unique(sorted=True, return_inverse=True)
        num_classes_in_batch = unique_labels.shape[0]

        # If only one class appears in the batch, clustering still works,
        # but inter-class orthogonality cannot be computed.
        frozen_centers = z.new_zeros(num_classes_in_batch, z.shape[1])
        frozen_centers.index_add_(0, inverse, z_fixed.detach())

        counts = torch.bincount(
            inverse,
            minlength=num_classes_in_batch,
        ).clamp_min(1)

        frozen_centers = frozen_centers / counts.unsqueeze(1)
        frozen_centers = self._safe_norm(frozen_centers, eps=eps)

        # Assign every tuned image to its corresponding frozen-image class prototype
        target_proto = frozen_centers[inverse]

        # --------------------------------------------------
        # 3. Cluster tuned image features around frozen image prototypes
        # --------------------------------------------------
        cos_cluster = torch.sum(z * target_proto.detach(), dim=1)
        cos_cluster = cos_cluster.clamp(-1.0 + eps, 1.0 - eps)

        theta_cluster = torch.acos(cos_cluster)
        loss_cluster = theta_cluster.pow(2).mean()

        # --------------------------------------------------
        # 4. Tuned image class centers
        # --------------------------------------------------
        if num_classes_in_batch <= 1:
            loss_orth = z.new_tensor(0.0)
        else:
            tuned_centers = z.new_zeros(num_classes_in_batch, z.shape[1])
            tuned_centers.index_add_(0, inverse, z)

            tuned_centers = tuned_centers / counts.unsqueeze(1)
            tuned_centers = self._safe_norm(tuned_centers, eps=eps)

            # --------------------------------------------------
            # 5. Strict orthogonality among tuned class centers
            # --------------------------------------------------
            gram = tuned_centers @ tuned_centers.t()

            identity = torch.eye(
                num_classes_in_batch,
                device=z.device,
                dtype=z.dtype,
            )

            loss_orth = (gram - identity).pow(2).sum() / (
                num_classes_in_batch * (num_classes_in_batch - 1)
            )

        loss = ( loss_point 
             
        )

        return loss


    def _popc_loss(
        self,
        text_features,
        image_features,
        text_features_fixed,
        image_features_fixed,
        labels,
    ):
        """
        POPC: Point-wise Orthogonal Prototype Consistency Loss.

        Total:
            text point-wise consistency
            + text class orthogonality
            + image point-wise consistency
            + image frozen-prototype clustering
            + image class-center orthogonality
        """

        loss_text = self._text_pointwise_orth_loss(
            text_features=text_features,
            text_features_fixed=text_features_fixed,
            point_weight=self.popc_text_point_weight,
            orth_weight=self.popc_text_orth_weight,
        )

        loss_image = self._image_frozen_proto_cluster_orth_loss(
            image_features=image_features,
            image_features_fixed=image_features_fixed,
            labels=labels,
            point_weight=self.popc_image_point_weight,
            cluster_weight=self.popc_image_cluster_weight,
            orth_weight=self.popc_image_orth_weight,
        )

        loss = loss_image + loss_text

        return loss, {
            "loss_popc": loss.detach(),
            "loss_popc_text": loss_text.detach(),
            "loss_popc_image": loss_image.detach(),
        }

    def _build_frozen_pre_states(self, frozen_root, frozen_post_states):
        """
        Build frozen pre-block token states for each layer.

        frozen_root: tokens before block 0
        frozen_post_states[i]: output after block i

        Then:
          pre[0] = frozen_root
          pre[l] = frozen_post_states[l-1] for l >= 1
        """
        pre_states = [frozen_root]
        for i in range(1, len(frozen_post_states)):
            pre_states.append(frozen_post_states[i - 1].detach())
        return pre_states

    def _cast_to_attn_dtype_device(self, x, attn_mod):
        if getattr(attn_mod, "in_proj_weight", None) is not None:
            return x.to(dtype=attn_mod.in_proj_weight.dtype, device=attn_mod.in_proj_weight.device)
        return x

    def _self_attn_out_from_block(self, block, q_x, k_x, v_x):
        """
        Compute Self-Attn output using the block's own ln_1 and attn.

        Inputs are [L, N, D].
        """
        out_dtype = q_x.dtype
        out_device = q_x.device

        qn = block.ln_1(q_x)
        kn = block.ln_1(k_x)
        vn = block.ln_1(v_x)

        qn = self._cast_to_attn_dtype_device(qn, block.attn)
        kn = self._cast_to_attn_dtype_device(kn, block.attn)
        vn = self._cast_to_attn_dtype_device(vn, block.attn)

        attn_mask = getattr(block, "attn_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=qn.dtype, device=qn.device)

        out = block.attn(qn, kn, vn, need_weights=False, attn_mask=attn_mask)[0]
        return out.to(dtype=out_dtype, device=out_device)

    def _attention_distillation_loss(self, block, tuned_pre, frozen_pre):
        """
        Eq. (3):
            || SelfAttn(Q,K,V) - SelfAttn(Q,Ks,Vs) ||_1

        current  = SelfAttn(tuned, tuned, tuned)
        ideal    = SelfAttn(tuned, frozen, frozen)
        """
        frozen_pre = frozen_pre.detach()

        current_out = self._self_attn_out_from_block(
            block=block,
            q_x=tuned_pre,
            k_x=tuned_pre,
            v_x=tuned_pre,
        )

        ideal_out = self._self_attn_out_from_block(
            block=block,
            q_x=tuned_pre,
            k_x=frozen_pre,
            v_x=frozen_pre,
        )

        return F.l1_loss(current_out.float(), ideal_out.float(), reduction="mean")

    def compute_layerwise_same_modality_ad_loss(
        self,
        text_pre_states,
        vision_pre_states,
        frozen_text_pre_states,
        frozen_visual_pre_states,
    ):
        loss_text_list = []
        loss_vision_list = []

        for layer_idx in self.prompt_learner.align_layers:
            # -------------------------
            # text AD loss
            # -------------------------
            tuned_text_l = self._prepare_text_full_context(text_pre_states[layer_idx])
            frozen_text_l = self._prepare_text_full_context(frozen_text_pre_states[layer_idx])
            frozen_text_l = frozen_text_l.to(device=tuned_text_l.device, dtype=tuned_text_l.dtype)

            text_block = self.text_encoder.transformer.resblocks[layer_idx]
            loss_text_l = self._attention_distillation_loss(
                block=text_block,
                tuned_pre=tuned_text_l,
                frozen_pre=frozen_text_l,
            )
            loss_text_list.append(loss_text_l)

            # -------------------------
            # vision AD loss
            # -------------------------
            tuned_vis_l = self._prepare_vision_full_context(
                vision_pre_states[layer_idx], is_frozen=False
            )
            frozen_vis_l = self._prepare_vision_full_context(
                frozen_visual_pre_states[layer_idx], is_frozen=True
            )
            frozen_vis_l = frozen_vis_l.to(device=tuned_vis_l.device, dtype=tuned_vis_l.dtype)

            vision_block = self.image_encoder.transformer.resblocks[layer_idx]
            loss_vision_l = self._attention_distillation_loss(
                block=vision_block,
                tuned_pre=tuned_vis_l,
                frozen_pre=frozen_vis_l,
            )
            loss_vision_list.append(loss_vision_l)

        if len(loss_text_list) == 0:
            zero = torch.zeros((), device=text_pre_states[0].device, dtype=torch.float32)
            return zero, zero, zero

        loss_text = torch.stack(loss_text_list).mean()
        loss_vision = torch.stack(loss_vision_list).mean()

        loss_same_modal_ad = (
            loss_text +
            loss_vision
        )

        return loss_same_modal_ad, loss_text, loss_vision    

    def _cosine_same_modality_loss(self, tuned, frozen):
        """
        tuned, frozen: [L, N, D]
        cosine over feature dim, then mean over tokens and batch/classes
        """
        tuned = F.normalize(tuned.float(), dim=-1)
        frozen = F.normalize(frozen.float(), dim=-1)
        frozen+=tuned.detach()  # stop gradient on frozen
        return 1.0 - (tuned * frozen).sum(dim=-1).mean()


    def _prepare_text_full_context(self, layer_x):
        """
        text tuned/frozen states are already aligned in sequence length
        layer_x: [L, Ncls, 512]
        """
        return layer_x


    def _prepare_vision_full_context(self, layer_x, is_frozen=False):
        """
        tuned vision states include appended prompt tokens at the tail.
        frozen vision states do not.

        tuned vision  : [197 + n_ctx, B, 768]
        frozen vision : [197, B, 768]
        """
        if not is_frozen:
            n_ctx = self.prompt_learner.n_ctx
            layer_x = layer_x[:-n_ctx, :, :]   # keep only CLS + patch tokens
        return layer_x    

    def _adaptive_token_pool(self, x, out_tokens):
        """
        x: [L, N, D]
        1) average over batch/class dimension -> [L, D]
        2) adaptively pool token axis L -> out_tokens
        returns: [out_tokens, D]
        """
        x = x.mean(dim=1)                     # [L, D]
        x = x.transpose(0, 1).unsqueeze(0)    # [1, D, L]
        x = F.adaptive_avg_pool1d(x, out_tokens)
        x = x.squeeze(0).transpose(0, 1)      # [out_tokens, D]
        return x


    def _pool_text_full_context(self, layer_x):
        """
        layer_x: [L_text, Ncls, 512]
        Use the full text sequence, then compress to K tokens.
        """
        k = self.prompt_learner.n_ctx
        return self._adaptive_token_pool(layer_x, k)   # [k, 512]


    def _pool_vision_full_context(self, layer_x, is_frozen=False):
        """
        layer_x:
        tuned vision  : [197 + n_ctx, B, 768]
        frozen vision : [197, B, 768]

        For tuned vision, drop the appended prompt tail before alignment so that
        both tuned and frozen vision represent the same base CLS+patch token stream.
        Then compress to K tokens.
        """
        k = self.prompt_learner.n_ctx
        n_ctx = self.prompt_learner.n_ctx

        if not is_frozen:
            layer_x = layer_x[:-n_ctx, :, :]  # remove appended prompt tokens

        return self._adaptive_token_pool(layer_x, k)   # [k, 768]

    def _pool_text_prompt_slots(self, layer_x):
        """
        layer_x: [L, Ncls, 512]
        prompt slots are at positions 1 : 1+n_ctx
        returns: [n_ctx, 512]
        """
        n_ctx = self.prompt_learner.n_ctx
        #return layer_x.mean(dim=1)
        return layer_x[1:1 + n_ctx, :, :].mean(dim=1)

    def _pool_vision_prompt_slots(self, layer_x):
        """
        layer_x: [L_with_prompts, B, 768]
        prompt slots are appended at the tail
        returns: [n_ctx, 768]
        """
        n_ctx = self.prompt_learner.n_ctx
        #return layer_x.mean(dim=1)
        return layer_x[-n_ctx:, :, :].mean(dim=1)

    def _cosine_alignment_loss(self, pred, target):
        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        #target +=pred.detach()  # stop gradient on target
        return 1.0 - (pred * target).sum(dim=-1).mean()
    
    def compute_layerwise_same_modality_alignment_loss(
        self,
        text_layer_states,
        vision_layer_states,
        frozen_text_layer_states,
        frozen_visual_layer_states,
    ):
        loss_text_list = []
        loss_vision_list = []

        for layer_idx in self.prompt_learner.align_layers:
            # -------------------------
            # text-to-text alignment
            # -------------------------
            tuned_text_l = self._prepare_text_full_context(text_layer_states[layer_idx])          # [L, Ncls, 512]
            frozen_text_l = self._prepare_text_full_context(frozen_text_layer_states[layer_idx])  # [L, Ncls, 512]
            frozen_text_l = frozen_text_l.to(device=tuned_text_l.device, dtype=tuned_text_l.dtype)

            loss_text_l = self._cosine_same_modality_loss(tuned_text_l, frozen_text_l)
            loss_text_list.append(loss_text_l)

            # -------------------------
            # vision-to-vision alignment
            # -------------------------
            tuned_vis_l = self._prepare_vision_full_context(
                vision_layer_states[layer_idx], is_frozen=False
            )                                                                                     # [197, B, 768]
            frozen_vis_l = self._prepare_vision_full_context(
                frozen_visual_layer_states[layer_idx], is_frozen=True
            )                                                                                     # [197, B, 768]
            frozen_vis_l = frozen_vis_l.to(device=tuned_vis_l.device, dtype=tuned_vis_l.dtype)

            loss_vision_l = self._cosine_same_modality_loss(tuned_vis_l, frozen_vis_l)
            loss_vision_list.append(loss_vision_l)

        if len(loss_text_list) == 0:
            zero = torch.zeros((), device=text_layer_states[0].device, dtype=torch.float32)
            return zero, zero, zero

        loss_text = torch.stack(loss_text_list).mean()
        loss_vision = torch.stack(loss_vision_list).mean()
        loss_same_modal =  (loss_text + loss_vision)

        return loss_same_modal, loss_text, loss_vision

    def compute_layerwise_cross_modal_alignment_loss(
        self,
        text_layer_states,
        vision_layer_states,
        frozen_text_layer_states,
        frozen_visual_layer_states,
    ):
        loss_t2v_list = []
        loss_v2t_list = []

        for proj_idx, layer_idx in enumerate(self.prompt_learner.align_layers):
            # tuned text -> frozen vision
            tuned_text_l = self._pool_text_prompt_slots(text_layer_states[layer_idx])             # [n_ctx, 512]
            frozen_vis_l = self._pool_vision_prompt_slots(frozen_visual_layer_states[layer_idx])  # [n_ctx, 768]
            frozen_vis_l = frozen_vis_l.to(device=tuned_text_l.device, dtype=tuned_text_l.dtype)

            pred_vis_l = self.prompt_learner.layer_text_to_vision_mlps[proj_idx](tuned_text_l)
            loss_t2v_l = self._cosine_alignment_loss(pred_vis_l, frozen_vis_l)
            loss_t2v_list.append(loss_t2v_l)

            # tuned vision -> frozen text
            tuned_vis_l = self._pool_vision_prompt_slots(vision_layer_states[layer_idx])          # [n_ctx, 768]
            frozen_txt_l = self._pool_text_prompt_slots(frozen_text_layer_states[layer_idx])      # [n_ctx, 512]
            frozen_txt_l = frozen_txt_l.to(device=tuned_vis_l.device, dtype=tuned_vis_l.dtype)

            pred_txt_l = self.prompt_learner.layer_vision_to_text_mlps[proj_idx](tuned_vis_l)
            loss_v2t_l = self._cosine_alignment_loss(pred_txt_l, frozen_txt_l)
            loss_v2t_list.append(loss_v2t_l)

        if len(loss_t2v_list) == 0:
            zero = torch.zeros((), device=text_layer_states[0].device, dtype=torch.float32)
            return zero, zero, zero

        loss_t2v = torch.stack(loss_t2v_list).mean()
        loss_v2t = torch.stack(loss_v2t_list).mean()
        #loss_cm_layer = 0.5 * (loss_t2v + loss_v2t)
        loss_cm_layer = (loss_t2v + loss_v2t)

        return loss_cm_layer, loss_t2v, loss_v2t    

    def reset_deep_supervision_state(self):

        self.prompt_learner.reset_ds_state()
        #self.prompt_learner.reset_ds_statetxt()
        #self.prompt_learner.reset_ds_statevision()


    def _norm(self, x, eps=1e-7):
        return F.normalize(x.float(), dim=-1, eps=eps).type_as(x)


    def _pointwise_orthogonal_consistency_loss(
        self,
        z,
        z_fixed,
        labels=None,
        point_weight=1.0,
        orth_weight=0.5,
        use_abs_orth=True,
    ):
        """
        Point-wise pretrained CLIP consistency + strict orthogonal separation.

        This avoids copying the full frozen CLIP pairwise class-relation matrix.

        Args:
            z:        tuned features,  [N, D]
            z_fixed:  frozen features, [N, D]
            labels:   optional labels.
                    For text features, keep labels=None.
                    For image features, provide labels to avoid separating same-class samples.
            point_weight: weight for point-wise angular consistency.
            orth_weight:  weight for strict orthogonal separation.
            use_abs_orth:
                True  = penalize both positive and negative correlation.
                False = penalize only positive correlation.
        """

        z = self._norm(z)
        z_fixed =  self._norm(z_fixed.detach())

        n = z.shape[0]

        if n <= 1:
            return z.new_tensor(0.0)

        # --------------------------------------------------
        # 1. Point-wise angular consistency
        # --------------------------------------------------
        cos_pos = torch.sum(z * z_fixed, dim=1)
        cos_pos = cos_pos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(cos_pos)
        loss_point = theta.pow(2).mean()

        # --------------------------------------------------
        # 2. Strict orthogonal separation
        # --------------------------------------------------
        gram = z @ z.t()

        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        pair_mask = ~eye

        # For image features, avoid pushing same-class samples apart.
        if labels is not None:
            labels = labels.view(-1)
            diff_class_mask = labels[:, None] != labels[None, :]
            pair_mask = pair_mask & diff_class_mask
            return loss_point 

        if pair_mask.sum() == 0:
            loss_orth = z.new_tensor(0.0)
        else:
            off_diag = gram[pair_mask]

            if use_abs_orth:
                # Strict orthogonality:
                # for i != j, z_i dot z_j should approach 0.
                loss_orth = off_diag.pow(2).mean()
            else:
                # Only reduce positive similarity.
                loss_orth = F.relu(off_diag).pow(2).mean()

        loss =  loss_point #+  loss_orth

        return loss  

    def set_dapt_visual_prototypes(self, prototypes: torch.Tensor):
        """
        Store dataset-level frozen visual prototypes.

        prototypes:
            [C, D], where C is number of classes and D is CLIP image feature dimension.

        These prototypes are computed once using the frozen zero-shot CLIP image encoder
        over the entire training dataset.
        """
        if prototypes is None:
            raise ValueError("DAPT visual prototypes cannot be None.")

        if prototypes.dim() != 2:
            raise ValueError(
                f"Expected prototypes with shape [num_classes, dim], got {prototypes.shape}"
            )

        prototypes = F.normalize(prototypes.float(), dim=-1, eps=1e-7)

        self.dapt_visual_prototypes = prototypes

    def _dapt_visual_intra_loss(
        self,
        image_features: torch.Tensor,
        labels: torch.Tensor,
        eps: float = 1e-7,
    ):
        """
        DAPT Eq. (13), cosine-similarity form.

        Original Eq. (13):
            L_intra = sum_c sum_i 1[y_i = c] || z_tilde_i - s_c ||_2^2

        Here:
            z_tilde_i = tuned/prompted image feature
            s_c       = frozen zero-shot CLIP visual prototype for class c

        Because both vectors are normalized:
            ||z - s||_2^2 = 2 - 2*cos(z, s)

        Therefore, this function supports:
            mode="l2_cos":        mean(2 - 2*cos)
            mode="one_minus_cos": mean(1 - cos)
        """

        if self.dapt_visual_prototypes.numel() == 0:
            raise RuntimeError(
                "DAPT visual prototypes are empty. "
                "Call build_dapt_visual_prototypes() in the trainer before training."
            )

        labels = labels.view(-1).long()

        z = F.normalize(image_features.float(), dim=-1, eps=eps)

        prototypes = self.dapt_visual_prototypes.to(
            device=z.device,
            dtype=z.dtype,
        )

        if labels.max().item() >= prototypes.shape[0]:
            raise ValueError(
                f"Label index {labels.max().item()} is outside prototype matrix "
                f"with {prototypes.shape[0]} classes."
            )

        target_proto = prototypes[labels]
        target_proto = F.normalize(target_proto, dim=-1, eps=eps)

        cos = torch.sum(z * target_proto.detach(), dim=1)
        cos = cos.clamp(-1.0 + eps, 1.0 - eps)

        if self.dapt_intra_mode == "one_minus_cos":
            loss = (1.0 - cos).mean()

        elif self.dapt_intra_mode == "l2_cos":
            # Equivalent to squared L2 distance for normalized vectors.
            loss = (2.0 - 2.0 * cos).mean()

        else:
            raise ValueError(
                f"Unknown DAPT_INTRA_MODE={self.dapt_intra_mode}. "
                "Use 'l2_cos' or 'one_minus_cos'."
            )

        return loss  

    def _zscore_logits(self, logits, eps=1e-6):
        logits = logits.float()
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, unbiased=False, keepdim=True)
        return (logits - mean) / (std + eps)


    def _smooth_l1(self, x, beta=0.5):
        abs_x = x.abs()
        return torch.where(
            abs_x < beta,
            0.5 * x.pow(2) / beta,
            abs_x - 0.5 * beta,
        )


    def _confidence_gated_logit_margin_loss(
        self,
        student_logits,
        teacher_logits,
        labels,
        topk=8,
        tau_conf=1.0,
        min_weight=0.05,
        beta=0.5,
        eps=1e-6,
    ):
        """
        Confidence-Gated Logit Margin Alignment.

        This aligns logit-space decision margins instead of copying the full
        frozen CLIP probability distribution.

        student_logits: tuned/prompted logits, [B, C]
        teacher_logits: frozen zero-shot CLIP logits, [B, C]
        labels: ground-truth labels, [B]
        """

        labels = labels.view(-1).long()

        student_logits = student_logits.float()
        teacher_logits = teacher_logits.detach().float()

        B, C = student_logits.shape

        if C <= 1:
            return student_logits.new_tensor(0.0)

        # --------------------------------------------------
        # 1. Per-sample logit normalization
        # --------------------------------------------------
        s = self._zscore_logits(student_logits, eps=eps)
        t = self._zscore_logits(teacher_logits, eps=eps).detach()

        # --------------------------------------------------
        # 2. Teacher confidence gate
        # --------------------------------------------------
        gt_teacher = teacher_logits.gather(1, labels.unsqueeze(1)).squeeze(1)

        gt_mask = torch.zeros_like(teacher_logits, dtype=torch.bool)
        gt_mask.scatter_(1, labels.unsqueeze(1), True)

        max_neg_teacher = teacher_logits.masked_fill(gt_mask, -1e4).max(dim=1).values

        teacher_margin = gt_teacher - max_neg_teacher

        weight = torch.sigmoid(teacher_margin / tau_conf)
        weight = min_weight + (1.0 - min_weight) * weight
        weight = weight.detach()

        # --------------------------------------------------
        # 3. Select hard classes
        # --------------------------------------------------
        k = min(topk, C)

        teacher_topk = teacher_logits.topk(k=k, dim=1).indices
        student_topk = student_logits.detach().topk(k=k, dim=1).indices

        selected = torch.cat(
            [
                labels.unsqueeze(1),
                teacher_topk,
                student_topk,
            ],
            dim=1,
        )

        # Remove duplicate issue is not critical; repeated classes only reweight
        # hard classes slightly. This is acceptable and stable.

        s_sel = s.gather(1, selected)
        t_sel = t.gather(1, selected)

        # first column is ground-truth because we inserted labels first
        s_gt = s_sel[:, :1]
        t_gt = t_sel[:, :1]

        s_margin = s_gt - s_sel
        t_margin = t_gt - t_sel

        # --------------------------------------------------
        # 4. Ignore ground-truth-to-ground-truth margin
        # --------------------------------------------------
        valid = selected != labels.unsqueeze(1)

        margin_diff = s_margin - t_margin

        loss_all = self._smooth_l1(margin_diff, beta=beta)
        loss_all = loss_all * valid.float()

        denom = valid.float().sum(dim=1).clamp_min(1.0)

        loss_per_sample = loss_all.sum(dim=1) / denom

        loss = (weight * loss_per_sample).sum() / weight.sum().clamp_min(eps)

        return loss

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        with torch.no_grad():
            image_features_fixed = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        # Compute the prompted image and text features
                # prompt learner returns tuned prompts + frozen per-layer references
        text_input, visual_ctx, cross_prompts_text_deeper, cross_prompts_visual_deeper, \
        init_cross_prompts_text, init_cross_prompts_visual = self.prompt_learner(image)
        #text_input, visual_ctx, cross_prompts_text_deeper, cross_prompts_visual_deeper, init_cross_prompts_text, init_cross_prompts_visual = self.prompt_learner(image)
        #text_features, text_layer_states = self.text_encoder(text_input, tokenized_prompts, cross_prompts_text_deeper, init_cross_prompts_text, init_cross_prompts_visual,return_layer_states=True)
        # tuned encoders now return pre/post layer states + frozen roots
        """text_features, text_pre_states, text_post_states, text_frozen_root = self.text_encoder(
            text_input,
            tokenized_prompts,
            cross_prompts_text_deeper,
            init_cross_prompts_text,
            init_cross_prompts_visual,
            return_layer_states=True,
        )"""
        text_features, text_pre_states, text_post_states, text_frozen_root = self.text_encoder(
            text_input,
            tokenized_prompts,
            cross_prompts_text_deeper,
            init_cross_prompts_text,
            init_cross_prompts_visual,
            return_layer_states=True,
        )
        """text_features, text_pre_states, text_post_states, text_frozen_root = self.text_encoder(
            text_input,
            tokenized_prompts,
            cross_prompts_text_deeper,
            init_cross_prompts_text,
            init_cross_prompts_visual,
            return_layer_states=True,
            trm_start_layer=self.trm_start_layer,
            trm_end_layer=self.trm_end_layer,
            trm_steps=self.trm_steps,
        )"""

        #image_features, vision_layer_states = self.image_encoder(image.type(self.dtype), visual_ctx, cross_prompts_visual_deeper, init_cross_prompts_visual,init_cross_prompts_text,return_layer_states=True)
        """image_features, vision_pre_states, vision_post_states, vision_frozen_root = self.image_encoder(
            image.type(self.dtype),
            visual_ctx,
            cross_prompts_visual_deeper,
            init_cross_prompts_visual,
            init_cross_prompts_text,
            return_layer_states=True,
        )"""
        image_features, vision_pre_states, vision_post_states, vision_frozen_root = self.image_encoder(
            image.type(self.dtype),
            visual_ctx,
            cross_prompts_visual_deeper,
            init_cross_prompts_visual,
            init_cross_prompts_text,
            return_layer_states=True,
        )
        """image_features, vision_pre_states, vision_post_states, vision_frozen_root = self.image_encoder(
            image.type(self.dtype),
            visual_ctx,
            cross_prompts_visual_deeper,
            init_cross_prompts_visual,
            init_cross_prompts_text,
            return_layer_states=True,
            trm_start_layer=self.trm_start_layer,
            trm_end_layer=self.trm_end_layer,
            trm_steps=self.trm_steps,
        )"""

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features + image_features_fixed
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features + self.prompt_learner.fixed_embeddings.half()
        #text_features = text_features + self.prompt_learner.fixed_embeddings.type_as(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # prompted logits
        logits = logit_scale * image_features @ text_features.t()

        with torch.no_grad():
            fixed_image = image_features_fixed #F.normalize(image_features_fixed.float(), dim=-1, eps=1e-7)

            fixed_text = self.prompt_learner.fixed_embeddings.to(
                device=fixed_image.device,
                dtype=fixed_image.dtype,
            )
            fixed_text = F.normalize(fixed_text.float(), dim=-1, eps=1e-7)

            teacher_logits = logit_scale.detach().float() * fixed_image @ fixed_text.t()

        if self.prompt_learner.training:
            loss_cls = F.cross_entropy(logits, label)

            loss_logit_margin = self._confidence_gated_logit_margin_loss(
                student_logits=logits,
                teacher_logits=teacher_logits,
                labels=label,
                topk=8,
                tau_conf=1.0,
                min_weight=0.05,
                beta=0.5,
            )
            # ---------------------------------------------------------
            # DAPT Eq. (13): visual intra-dispersion loss
            # Dataset-level prototype version
            # ---------------------------------------------------------
            if self.dapt_intra_enable:
                loss_dapt_intra = self._dapt_visual_intra_loss(
                    image_features=image_features,
                    labels=label,
                )
            else:
                loss_dapt_intra = logits.new_tensor(0.0)
            text_features_fixed = self.prompt_learner.fixed_embeddings
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            score = cos(text_features, text_features_fixed)
            loss_distill_text = 1.0 - torch.mean(score)
            #arcos implementation for better numerical stability
            score_txt = torch.acos(torch.clamp(score, min=-1.0, max=1.0))
            loss_distill_text_arcos = score_txt.pow(2).mean()
            score = cos(image_features, image_features_fixed)
            #arcos implementation for better numerical stability
            score_img = torch.acos(torch.clamp(score, min=-1.0, max=1.0))
            loss_distill_image_arcos = score_img.pow(2).mean()
            loss_distill_image = 1.0 - torch.mean(score)
            loss_distill = loss_distill_text + loss_distill_image
            loss_distill_arcos = loss_distill_text_arcos + loss_distill_image_arcos
            loss_arcmc, arcmc_logs = self.arcmc_loss(
                text_features=text_features,
                image_features=image_features,
                fixed_text_features=self.prompt_learner.fixed_embeddings,
                fixed_image_features=image_features_fixed,
                labels=label,
            )
            #-----------------------------------------------------------------
            #contrastive feature distillation loss between tuned features and frozen CLIP features
            loss_distill_contrastive, distill_logs = self.contrastive_distill_loss(
                text_features=text_features,
                image_features=image_features,
                fixed_text_features=self.prompt_learner.fixed_embeddings,
                fixed_image_features=image_features_fixed,
                labels=label,
            )

            # ---------------------------------------------------------
            # Point-wise Orthogonal Consistency Loss
            # ---------------------------------------------------------
            loss_popc_new, popc_logs = self._popc_loss(
                text_features=text_features,
                image_features=image_features,
                text_features_fixed=text_features_fixed,
                image_features_fixed=image_features_fixed,
                labels=label,
            )
            if self.poc_enable:
                loss_poc_text = self._pointwise_orthogonal_consistency_loss(
                    z=text_features,
                    z_fixed=self.prompt_learner.fixed_embeddings,
                    labels=None,
                    point_weight=self.poc_text_point_weight,
                    orth_weight=self.poc_text_orth_weight,
                    use_abs_orth=True,
                )

                loss_poc_image = self._pointwise_orthogonal_consistency_loss(
                    z=image_features,
                    z_fixed=image_features_fixed,
                    labels=label,
                    point_weight=self.poc_image_point_weight,
                    orth_weight=self.poc_image_orth_weight,
                    use_abs_orth=True,
                )

                loss_poc = loss_poc_text + loss_poc_image
            #else:
            #    loss_poc = logits.new_tensor(0.0)

            # ---------------------------------------------------------
            # New class-normalized orthogonal probe distillation loss
            # ---------------------------------------------------------
            if self.probe_enable:
                loss_probe, probe_logs = self.prompt_learner.probe_loss(
                    text_features=text_features,
                    image_features=image_features,
                    fixed_text_features=self.prompt_learner.fixed_embeddings,
                    fixed_image_features=image_features_fixed,
                    labels=label,
                )
                loss = loss_cls + self.probe_lambda * loss_probe + 6.0 * loss_popc_new #4.0 * loss_popc_new
                return loss, logits
            #------------
            # new layerwise cross-modal cosine alignment
            """loss_cm_layer, loss_t2v, loss_v2t = self.compute_layerwise_cross_modal_alignment_loss(
                text_layer_states=text_post_states,
                vision_layer_states=vision_post_states,
                frozen_text_layer_states= text_post_states, #init_cross_prompts_text,
                frozen_visual_layer_states= vision_post_states, #init_cross_prompts_visual,
            )"""
            #same modality
            """loss_sm_layer, loss_text_layer, loss_vision_layer = self.compute_layerwise_same_modality_alignment_loss(
                text_layer_states=text_post_states,
                vision_layer_states=vision_post_states,
                frozen_text_layer_states=init_cross_prompts_text,
                frozen_visual_layer_states=init_cross_prompts_visual,
            )
            # build frozen pre-block states for exact Eq. (3)
            frozen_text_pre_states = self._build_frozen_pre_states(
                frozen_root=text_frozen_root,
                frozen_post_states=init_cross_prompts_text,
            )
            frozen_visual_pre_states = self._build_frozen_pre_states(
                frozen_root=vision_frozen_root,
                frozen_post_states=init_cross_prompts_visual,
            )"""

            # Eq. (3) same-modality AD loss on layers 3..9 (via align_layers)
            """loss_ad_layer, loss_ad_text, loss_ad_vision = self.compute_layerwise_same_modality_ad_loss(
                text_pre_states=text_pre_states,
                vision_pre_states=vision_pre_states,
                frozen_text_pre_states=frozen_text_pre_states,
                frozen_visual_pre_states=frozen_visual_pre_states,
            )"""
            #print("training logit test:", logits)
            return loss_cls + self.probe_lambda * loss_popc_new, logits #+ self.probe_lambda * loss_popc_new, logits  # + self.lambd * loss_distill,logits self.ad_lambda * loss_ad_layer
                # Important: do not let prompt-level TRM state leak across test batches
        if (label is None) and hasattr(self, "reset_deep_supervision_state"):
            self.reset_deep_supervision_state()
        #print("Testing, only return logits:",logits)
        return logits


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os
    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=0)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_random.pt")
    return clip_weights

def gpt_tokenized_prompts(classnames, gpt3_prompt, dataset_name, mode="first"):
    """
    Build one tokenized GPT prompt per class for prompt-slot initialisation.

    mode:
      - "first": use the first GPT prompt for each class
      - "random": use a random GPT prompt for each class
    Returns:
      tokenized_prompts: [n_cls, 77]
      prompt_strings: list[str]
    """
    prompt_strings = []

    for classname in classnames:
        # classnames in your code are already normalized later with replace("_", " ")
        cname = classname.replace("_", " ")

        # Expected gpt3_prompt structure: dict[classname] -> list[str]
        # If dataset-specific formatting is already done upstream, just read prompts
        if cname in gpt3_prompt:
            candidates = gpt3_prompt[cname]
        elif classname in gpt3_prompt:
            candidates = gpt3_prompt[classname]
        else:
            # safe fallback
            candidates = [f"a photo of a {cname}."]

        if not isinstance(candidates, list):
            candidates = [candidates]

        if mode == "random":
            chosen = random.choice(candidates)
        else:
            chosen = candidates[0]

        prompt_strings.append(chosen)

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompt_strings])
    return tokenized_prompts, prompt_strings

@TRAINER_REGISTRY.register()
class HiCroPLReason(TrainerX):
    def check_cfg(self, cfg):
        #assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.HICROPL.PREC in ["fp16","fp32","amp"]

    @torch.no_grad()
    def build_dapt_visual_prototypes(self):
        """
        Build DAPT Eq. (12) visual prototypes from the entire training dataset.

        Eq. (12):
            s_c = mean_{(x_i, y_i) in D_c} f(x_i)

        Here:
            f(x_i) is the frozen zero-shot CLIP image feature from
            self.model.prompt_learner.ZS_image_encoder.

        Output:
            prototypes: [num_classes, feature_dim]
        """

        print("Building DAPT visual prototypes from full training dataset...")

        # Handle DataParallel safely, although this should be called before wrapping.
        model = self.model.module if hasattr(self.model, "module") else self.model

        model.eval()
        zs_image_encoder = model.prompt_learner.ZS_image_encoder
        zs_image_encoder.eval()

        num_classes = len(self.dm.dataset.classnames)

        feature_sums = None
        class_counts = torch.zeros(
            num_classes,
            device=self.device,
            dtype=torch.float32,
        )

        for batch in self.train_loader_x:
            image = batch["img"].to(self.device)
            label = batch["label"].to(self.device).long()

            # Frozen zero-shot CLIP image features
            image_features = zs_image_encoder(image.type(model.dtype))
            image_features = F.normalize(image_features.float(), dim=-1, eps=1e-7)

            if feature_sums is None:
                feature_dim = image_features.shape[-1]
                feature_sums = torch.zeros(
                    num_classes,
                    feature_dim,
                    device=self.device,
                    dtype=torch.float32,
                )

            feature_sums.index_add_(0, label, image_features)

            ones = torch.ones_like(label, dtype=torch.float32, device=self.device)
            class_counts.index_add_(0, label, ones)

        if feature_sums is None:
            raise RuntimeError("Could not build DAPT prototypes: train_loader_x is empty.")

        missing = torch.where(class_counts == 0)[0]
        if len(missing) > 0:
            print(
                "[Warning] Some classes have zero samples when building DAPT prototypes: "
                f"{missing.detach().cpu().tolist()}"
            )

        prototypes = feature_sums / class_counts.clamp_min(1.0).unsqueeze(1)
        prototypes = F.normalize(prototypes, dim=-1, eps=1e-7)

        model.set_dapt_visual_prototypes(prototypes)

        print(
            f"DAPT visual prototypes built: shape={tuple(prototypes.shape)}, "
            f"min_count={class_counts.min().item():.0f}, "
            f"max_count={class_counts.max().item():.0f}"
        )

        if bool(self.cfg.TRAINER.HICROPLReason.DAPT_SAVE_PROTOTYPES):
            save_path = osp.join(self.cfg.OUTPUT_DIR, "dapt_visual_prototypes.pt")
            torch.save(
                {
                    "prototypes": prototypes.detach().cpu(),
                    "class_counts": class_counts.detach().cpu(),
                    "classnames": self.dm.dataset.classnames,
                },
                save_path,
            )
            print(f"Saved DAPT visual prototypes to: {save_path}")

        # Return model to train mode after prototype construction.
        model.train()    

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.HICROPL.PREC == "fp32" or cfg.TRAINER.HICROPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        """name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)"""
        #-----------------------------------------
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        if bool(cfg.TRAINER.HICROPLReason.DAPT_INTRA_ENABLE):
            self.build_dapt_visual_prototypes()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        #self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None
        self.scaler = GradScaler() if cfg.TRAINER.HICROPL.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        #print("image", image.shape, "label", label.shape)
        #print("image dtype", image.dtype)
        #print("label dtype", label.dtype)
        #print("image ", image)
        #print("label",label)
        nsup = 1
        if hasattr(self.model, "reset_deep_supervision_state"):
            self.model.reset_deep_supervision_state()
        #loss_sum = 0.0
        steps_done = 0

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.HICROPL.PREC
        """for step in range(nsup):
            steps_done += 1
            if prec == "amp":
                with autocast():
                    loss,logits = model(image, label)
                    #print("loss", loss)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss,logits = model(image, label)
                #print("loss", loss, "logits shape", logits.shape)
                explicit_all = REGULARIZER_REGISTRY.get("margin_mean_var_allclass_loss_explicit")
                explicit_all_loss =explicit_all(logits,label,variance_mode="all_pairs")
                #loss= loss + explicit_all_loss
                optim.zero_grad()
                loss.backward()
                optim.step()
            loss_sum += float(loss.item()) 
        avg_loss = loss_sum / max(steps_done, 1)"""
        """loss_sum = 0.0
        #steps_done = 0
        if prec == "amp":
            for _ in range(nsup):
                with autocast():
                    loss,logits = self.model(image, label)
                    #loss = loss_raw / nsup
                self.optim.zero_grad()    
                self.scaler.scale(loss).backward()
                loss_sum += float(loss.item())

                self.scaler.unscale_(self.optim)
                #self._accumulate_eff_gradnorm()
                self.scaler.step(self.optim)
                self.scaler.update()

        else:
            for _ in range(nsup):
                self.optim.zero_grad(set_to_none=True)
                loss,logits = self.model(image, label)
                #loss = loss_raw / nsup
                self.optim.zero_grad()
                loss.backward()
                loss_sum += float(loss.item())
            #if (q_hat>0.5).all().item():
            #    break    
                #self._accumulate_eff_gradnorm()                
                self.optim.step()
        avg_loss = loss_sum / nsup"""
        #---------------------------------
        #model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.HICROPL.PREC
        if prec == "amp":
            with autocast():
                loss,logits = model(image, label)
                #print("loss", loss)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss,logits = model(image, label)
            #print("loss", loss, "logits shape", logits.shape)
            explicit_all = REGULARIZER_REGISTRY.get("margin_mean_var_allclass_loss_explicit")
            explicit_all_loss =explicit_all(logits,label,variance_mode="all_pairs")
            #loss= loss + explicit_all_loss
            optim.zero_grad()
            loss.backward()
            optim.step()        

        loss_summary = {"loss": loss.item()}
        #loss_summary = {"loss": avg_loss}

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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)