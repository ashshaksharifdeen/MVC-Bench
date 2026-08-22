from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

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
        design_details = {"trainer": 'HiCroPL',
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

class CastedLinearvision(nn.Module):
    def __init__(self,
                 in_features: int, #dimension of the vector you feed into this layer
                 out_features: int, #dimension of the vector you want out
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype, device=input.device), bias=self.bias.to(dtype=input.dtype, device=input.device) if self.bias is not None else None)

class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int, #dimension of the vector you feed into this layer
                 out_features: int, #dimension of the vector you want out
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype, device=input.device), bias=self.bias.to(dtype=input.dtype, device=input.device) if self.bias is not None else None)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class ImageBatchAggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        """
        x: [Lt, B, C, Dt]
        returns: [Lt, C, Dt]
        """

        # [Lt, B, C, Dt] -> [Lt, C, B, Dt]
        x = x.permute(0, 2, 1, 3)

        # attention score over image dimension B
        attn = self.score(x)
        # [Lt, C, B, 1]

        attn = torch.softmax(attn, dim=2)

        # weighted sum over B
        x = (x * attn).sum(dim=2)
        # [Lt, C, Dt]

        return x    
    
class ResidualAttentionBlock_HiCroPLReason(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        layers: int,
        attn_mask: torch.Tensor = None,
        add_prompt=False,
        text_layer=False,
        i=0,
        design_details=None,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_2 = LayerNorm(d_model)

        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.cross_prompt_nctx = design_details["vision_ctx"]

        self.i = i
        self.add_prompt = add_prompt if i != 0 else False

    def attention(self, x: torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(
                dtype=x.dtype,
                device=x.device,
            )

        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=self.attn_mask,
        )[0]

    def forward(self, inputs):
        x, cross_prompts_deeper = inputs

        if self.add_prompt:
            if not self.text_layer:
                # Remove the previous visual prompt tokens.
                prefix = x[:-self.cross_prompt_nctx, :, :]

                # Insert this layer's visual prompt.
                visual_context = cross_prompts_deeper[self.i - 1]
                visual_context = visual_context.expand(
                    x.shape[1],
                    -1,
                    -1,
                ).permute(1, 0, 2)

                x = torch.cat(
                    [prefix, visual_context],
                    dim=0,
                )

            else:
                # Preserve SOS and all tokens after the prompt slots.
                prefix = x[:1, :, :]
                suffix = x[1 + self.cross_prompt_nctx :, :, :]

                # Insert this layer's text prompt.
                textual_context = cross_prompts_deeper[self.i - 1]
                textual_context = textual_context.expand(
                    x.shape[1],
                    -1,
                    -1,
                ).permute(1, 0, 2)

                x = torch.cat(
                    [prefix, textual_context, suffix],
                    dim=0,
                )

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return [x, cross_prompts_deeper]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int 
                ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, inputs):
        # q: (Lq,E_q) or (Lq,N,E_q)
        # k,v: (Lk,E_k) or (Lk,N,E_k)
        x=inputs
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x 



class FrozenTextLayerContextEncoder(nn.Module):
    """
    Extract per-layer text contexts from a frozen no-prompt CLIP text encoder.
    Returns one [n_ctx, 512] tensor per prompt depth.
    """
    def __init__(self, width, heads, layers, attn_mask, text_layer, design_details):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                         text_layer, i,
                                                                         design_details) 
                                             for i in range(layers)])

    @torch.no_grad()
    def forward(self, x):
        ctxs = []
        for blk in self.resblocks:
            with torch.no_grad():
                x = blk(x)
                ctxs.append(x)
        return ctxs


class FrozenVisionLayerContextEncoder(nn.Module):
    """
    Extract per-layer visual contexts from a frozen no-prompt CLIP ViT encoder.
    Returns one [n_ctx, 768] tensor per prompt depth.

    Since the frozen visual encoder has no appended prompt slots, we use the
    last n_ctx tokens as the position-aligned proxy for the prompt insertion tail.
    """
    def __init__(self, width, heads, layers, attn_mask, text_layer, design_details):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                         text_layer, i,
                                                                         design_details)
                                             for i in range(layers)])

    @torch.no_grad()
    def forward(self, x):
        ctxs = []
        for blk in self.resblocks:
            with torch.no_grad():
                x = blk(x)
                ctxs.append(x)
        return ctxs    
    
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
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)

        q_was_2d = (q_proj.dim() == 2)
        k_was_2d = (k_proj.dim() == 2)
        v_was_2d = (v_proj.dim() == 2)
        all_were_2d = q_was_2d and k_was_2d and v_was_2d

        # Determine target batch size from any 3D tensor
        batch_sizes = [t.shape[1] for t in (q_proj, k_proj, v_proj) if t.dim() == 3]
        target_batch = batch_sizes[0] if len(batch_sizes) > 0 else 1

        # Sanity check: existing 3D tensors must agree on batch size
        for b in batch_sizes:
            if b != target_batch:
                raise ValueError(f"Batch mismatch in CrossPromptAttention: {batch_sizes}")

        # Convert only the tensors that are 2D
        if q_was_2d:
            q_proj = q_proj.unsqueeze(1).expand(-1, target_batch, -1)
        if k_was_2d:
            k_proj = k_proj.unsqueeze(1).expand(-1, target_batch, -1)
        if v_was_2d:
            v_proj = v_proj.unsqueeze(1).expand(-1, target_batch, -1)

        attn_out = self.attn(
            self.ln_1(q_proj),
            self.ln_1(k_proj),
            self.ln_1(v_proj),
            need_weights=False
        )[0]

        q_proj = q_proj + attn_out
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))

        # Only squeeze back when all inputs were originally 2D
        if all_were_2d:
            q_proj = q_proj.squeeze(1)

        return q_proj    

class ResidualAttentionBlock_HiCroPL(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.cross_prompt_nctx = design_details['vision_ctx']
        self.i = i
        if self.i != 0:
            self.add_prompt = add_prompt
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        x = inputs[0]
        cross_prompts_deeper = inputs[1]

        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:  # Depending on the hyper-parameter K, self.add_prompt is set to True when i < K ,
            # Also see if this is textual transformer layer or not
            if not self.text_layer:  # visual
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.cross_prompt_nctx, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = cross_prompts_deeper[self.i-1]
                visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:  # text
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.cross_prompt_nctx:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = cross_prompts_deeper[self.i-1]
                textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, cross_prompts_deeper]

class TokenMixMLP(nn.Module):
    """
    MLP over sequence length, like MLP-Mixer token mixing.
    Input: [L, N, D]
    """
    def __init__(self, seq_len: int, hidden_mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(seq_len, int(seq_len * hidden_mult))
        self.fc1 = nn.Linear(seq_len, hidden)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden, seq_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # [L, N, D] -> [N, D, L]
        y = x.permute(1, 2, 0).contiguous()
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        # [N, D, L] -> [L, N, D]
        return y.permute(2, 0, 1).contiguous()


class PromptMixerBlock(nn.Module):
    """
    Token-mixing MLP + channel MLP.
    Input/output: [L, N, D]
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        token_hidden_mult: float = 2.0,
        channel_hidden_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_tok = LayerNorm(d_model)
        self.token_mixer = TokenMixMLP(seq_len, token_hidden_mult, dropout)

        self.ln_chn = LayerNorm(d_model)
        ch_hidden = max(d_model, int(d_model * channel_hidden_mult))
        self.channel_mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d_model, ch_hidden)),
            ("act", QuickGELU()),
            ("drop1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(ch_hidden, d_model)),
            ("drop2", nn.Dropout(dropout)),
        ]))

    def forward(self, x: torch.Tensor):
        x = x + self.token_mixer(self.ln_tok(x))
        x = x + self.channel_mlp(self.ln_chn(x))
        return x        

# LKP
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(AttentionPooling, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = LayerNorm(hidden_size) #nn.LayerNorm(hidden_size)
        self.ln_2 = LayerNorm(hidden_size) #nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_size * 4, hidden_size)),
        ]))

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
        token_query = token_query + self.mlp(self.ln_2(token_query))
        #token_query = self.ln_2(token_query)

        if squeeze_back:
            token_query = token_query.squeeze(1)        # back to (1, E)

        return token_query          

class ResidualAttentionBlock_HigherEncoderMaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()
        self.H_cycles = 2
        self.L_cycles = 2
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        self.attn_pooling = AttentionPooling(hidden_size=d_model, num_attention_heads=n_head)
        self._ds_prev_zH = None
        self._ds_prev_zL = None
        self._ds_prev_zH_txt = None
        self._ds_prev_zL_txt = None
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def reset_ds_state(self):
        self._ds_prev_zH = None
        self._ds_prev_zL = None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def L_level(self,z_L,z_H,v):
        # kv: [1 + (L-T) + T, N, D]
        #kv = torch.cat([z_H, x_main, z_L], dim=0)
        q  = self.ln_1(z_L)
        k = self.ln_1(z_H)
        v = self.ln_1(v)
        attn_out = self.attn(q, k, v, need_weights=False, attn_mask=None)[0]
        #z_L = z_L + attn_out
        #z_L = z_L + self.mlp(self.ln_2(z_L))
        z_L = self.mlp(self.ln_2(attn_out))
        return z_L
    
    def H_level(self,z_H, z_L):
            q = self.ln_1(z_H)
            kv = self.ln_1(z_L)
            #v = self.ln_1(z_L)
            attn_out = self.attn(q, kv, kv, need_weights=False, attn_mask=None)[0]
            #z_H = z_H + attn_out
            #z_H = z_H + self.mlp(self.ln_2(z_H))
            z_H = self.mlp(self.ln_2(attn_out))
            return z_H
    
    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.text_layer:
             self.q_head = CastedLinearvision(x.shape[2], 2, bias=True)
          
        """if len(inputs) == 3:
           x, compound_prompts_deeper, counter = inputs
           halt_mask = None
        else:
           x, compound_prompts_deeper, counter, halt_mask = inputs"""

        #x_in = x  # for freezing halted samples
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both
                #print("before deep layer concatenation, x shape:",x.shape) torch.Size([199, 4, 768])
                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        #print("prefix shape:",prefix.shape) prefix shape: torch.Size([197, 4, 768])
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        #print("visual_context shape:",visual_context.shape) visual_context shape: torch.Size([2, 4, 768])
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)
                        #print("after deep layer concatenation, x shape:",x.shape) torch.Size([199, 4, 768])
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        #print("before text prefix x shape:",x.shape)>before text prefix x shape: torch.Size([77, 50, 512])
                        prefix = x[:1, :, :]
                        #print("after text prefix x shape:",prefix.shape) after text prefix x shape: torch.Size([1, 50, 512])
                        suffix = x[1 + self.compound_prompt_nctx:, :, :] #torch.Size([74, 50, 512])>>
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter] #textual context shape: torch.Size([2, 512])
                        #print("textual context shape:",textual_context.shape) textual context shape: torch.Size([2, 512])
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        #print("textual context expanded shape:",textual_context.shape)textual context expanded shape: torch.Size([2, 50, 512])
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        #print("x concatenated shape:",x.shape)x concatenated shape: torch.Size([77, 50, 512])
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        #print("after full deep layer intergration, x shape:",x.shape) torch.Size([199, 4, 768])
        if not self.text_layer:
           #add hrm here 197 for zh and last 2 for z_l
           # x is [L, N, D]
            L, N, D = x.shape
            T = self.compound_prompt_nctx #self.design_details.get("maple_length", 0)  # prompt length

            if T > 0 and L > T:
              x_main = x[:-T]      # [L-T, N, D]  CLS+patch tokens
              tuna_cxt = x[-T:]  # [L-T-1, N, D]  patch tokens (context)
              if (self._ds_prev_zH is not None) and (self._ds_prev_zL is not None) and (self._ds_prev_zH.shape == tuna_cxt.shape) and (self._ds_prev_zL.shape == tuna_cxt.shape): 
                      z_H = self._ds_prev_zH.to(dtype=tuna_cxt.dtype, device=tuna_cxt.device)
                      z_L = self._ds_prev_zL.to(dtype=tuna_cxt.dtype, device=tuna_cxt.device)
                      #print("visual latent states loaded from previous step.")
              else:
                      z_L    = tuna_cxt.clone()      # [T,   N, D]  prompt tokens (latent)
                      z_H    = tuna_cxt.clone() #torch.zeros(tuna_cxt.shape[0],tuna_cxt.shape[1],tuna_cxt.shape[2],dtype=tuna_cxt.dtype,device=tuna_cxt.device)  #x_main[:1]  # [1,   N, D]  CLS token (high-level latent)
                      #print("visual latent states initialized freshly.")
            """with torch.no_grad():
               #z_H_ng, z_L_ng =  z_H, z_L
               for h in range(self.H_cycles):
                    for l in range(self.L_cycles):
                       z_L  = self.L_level(z_L,z_H,tuna_cxt)
                       #print("z_L aftrer L level shape:",z_L.shape) z_L aftrer L level shape: torch.Size([2, 4, 768])
                       # skip the last H update
                    z_H= self.H_level(z_H, z_L)
            #z_L_new = self.L_level(z_L_ng, z_H_ng)
            #z_H_new = self.H_level(z_H_ng, z_L_new)
            #z_L=z_L.detach()
            #z_H=z_H.detach()
            for l in range(self.L_cycles):
                z_L  = self.L_level(z_L,z_H,tuna_cxt)
                #print("z_L aftrer L level shape:",z_L.shape) z_L aftrer L level shape: torch.Size([2, 4, 768])
                # skip the last H update
            z_H= self.H_level(z_H, z_L)""" 
            #x+=z_H
            self._ds_prev_zH = z_H.detach()
            self._ds_prev_zL = z_L.detach()
            q_cxt =z_H[0,:,:].to(dtype=x_main.dtype, device=x_main.device)
            q_logits= self.q_head(q_cxt)  # [N, 2]
            x = torch.cat([x_main, z_H+tuna_cxt], dim=0)
            # optionally also replace CLS token
            #x[:1] = z_H_ng        
        else:
            prefix_cxt = x[:1, :, :]
            tune_txt_cxt=x[1:1+self.compound_prompt_nctx,:,:]
            suffix_cxt = x[1 + self.compound_prompt_nctx:, :, :]
            if (self._ds_prev_zH_txt is not None) and (self._ds_prev_zL_txt is not None) and (self._ds_prev_zH_txt.shape == tune_txt_cxt.shape) and (self._ds_prev_zL_txt.shape == tune_txt_cxt.shape):
                    z_H = self._ds_prev_zH_txt.to(dtype=tune_txt_cxt.dtype, device=tune_txt_cxt.device)
                    z_L = self._ds_prev_zL_txt.to(dtype=tune_txt_cxt.dtype, device=tune_txt_cxt.device)
                    #print("text latent states loaded from previous step.")
            else:
                    z_L= tune_txt_cxt.clone()
                    z_H= tune_txt_cxt.clone()
                    #print("text latent states initialized freshly.")
            """with torch.no_grad():
               #z_H_ng, z_L_ng =  z_H, z_L
               for h in range(self.H_cycles):
                    for l in range(self.L_cycles):
                       z_L  = self.L_level(z_L, z_H, tune_txt_cxt)
                       #print("z_L aftrer L level shape:",z_L.shape) z_L aftrer L level shape: torch.Size([2, 4, 768])
                       # skip the last H update
                    z_H= self.H_level(z_H, z_L)
            #z_L_new = self.L_level(z_L_ng, z_H_ng)
            #z_H_new = self.H_level(z_H_ng, z_L_new) 
            #z_L=z_L.detach()
            #z_H=z_H.detach()
            for l in range(self.L_cycles):
                z_L  = self.L_level(z_L,z_H,tune_txt_cxt)
                #print("z_L aftrer L level shape:",z_L.shape) z_L aftrer L level shape: torch.Size([2, 4, 768])
                # skip the last H update
                z_H= self.H_level(z_H, z_L)"""
            self._ds_prev_zH_txt = z_H.detach()
            self._ds_prev_zL_txt = z_L.detach()
            #q_cxt =z_H[0,:,:].to(dtype=x.dtype, device=x.device)
            #q_logits= self.q_head(q_cxt)  # [N, 2]
            """zH_new = z_H.detach()
            zL_new = z_L.detach()

            if halt_mask is None or self._ds_prev_zH is None:
                    self._ds_prev_zH = zH_new
                    self._ds_prev_zL = zL_new
            else:
                    m = halt_mask.view(1, -1, 1).to(dtype=zH_new.dtype, device=zH_new.device)
                    m = m.expand_as(zH_new)  # [T,N,D] (or [1,N,D] if you change zH shape)
                    self._ds_prev_zH = zH_new * (1.0 - m) + self._ds_prev_zH.to(zH_new.dtype) * m
                    self._ds_prev_zL = zL_new * (1.0 - m) + self._ds_prev_zL.to(zL_new.dtype) * m
            #x+=z_H
            if halt_mask is not None and halt_mask.numel() == x.shape[1]:
               # halt_mask: [N] bool; True means "halted => keep x_in"
               m = halt_mask.view(1, -1, 1).to(dtype=x.dtype, device=x.device)  # [1,N,1]
               x = x * (1.0 - m) + x_in * m"""
            x = torch.cat([prefix_cxt, z_H+tune_txt_cxt, suffix_cxt], dim=0)

        #----------------------------------end of hrm-------------------------
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if not self.text_layer:
            return [x, compound_prompts_deeper, counter,q_logits]  # return again as a list, so that nn.seq can work
        else:
            return [x, compound_prompts_deeper, counter]  
  
    
class ResidualAttentionBlock_HRMMaPLe(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        design_details=None,
        text_layer=False,
        i=0,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.compound_prompt_nctx = int(design_details["maple_length"])
        self.trm_steps = 3 #int(design_details.get("trm_steps", 1))
        self.trm_warmup = 6 #int(design_details.get("trm_warmup", 1))
        self.first_layer = (i == 0)
        # switch: "attn" or "mlp"
        self.trm_token_mixer = str(design_details.get("trm_token_mixer", "mlp")).lower()
        self.use_mlp_trm = self.trm_token_mixer == "mlp"
        
        # only for fallback attention-based TRM
        if not self.use_mlp_trm:
            self.attn_pooling = AttentionPooling(hidden_size=d_model, num_attention_heads=n_head)
        
        # MLP-based TRM
        if self.use_mlp_trm:
            tok_mult = float(design_details.get("trm_mlp_token_hidden_mult", 2.0))
            ch_mult = float(design_details.get("trm_mlp_channel_hidden_mult", 4.0))
            drop = float(design_details.get("trm_dropout", 0.0))
            T = self.compound_prompt_nctx

            if not self.text_layer:
                self.z_mixer = PromptMixerBlock(
                seq_len=6,
                d_model=d_model,
                token_hidden_mult=tok_mult,
                channel_hidden_mult=ch_mult,
                dropout=drop,)
            
            else:
                self.z_mixer = PromptMixerBlock(
                seq_len=6,
                d_model=d_model,
                token_hidden_mult=tok_mult,
                channel_hidden_mult=ch_mult,
                dropout=drop,)

            # z update sees [x_anchor, y, z] -> length = 3T
            """self.z_mixer = PromptMixerBlock(
                seq_len=3 * T,
                d_model=d_model,
                token_hidden_mult=tok_mult,
                channel_hidden_mult=ch_mult,
                dropout=drop,
            )"""

            # y update sees [y, z] -> length = 2T
            self.y_mixer = PromptMixerBlock(
                seq_len=2 * T,
                d_model=d_model,
                token_hidden_mult=tok_mult,
                channel_hidden_mult=ch_mult,
                dropout=drop,
            )

        
        # deep-supervision state
        self._ds_prev_y = None
        self._ds_prev_z = None

    def reset_ds_state(self):
        self._ds_prev_y = None
        self._ds_prev_z = None

    def effective_layer_cost(self) -> int:
        # final transformer block + internal TRM refinement cost
        return 2 + 2 * (self.trm_warmup + self.trm_steps)

    def _self_attn(self, x: torch.Tensor, use_mask: bool):
        attn_mask = None
        if use_mask and self.attn_mask is not None:
            if self.attn_mask.shape[0] == x.shape[0]:
                attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def attention(self, x: torch.Tensor):
        return self._self_attn(x, use_mask=True)

    def _block(self, x: torch.Tensor, use_mask: bool = False):
        x = x + self._self_attn(self.ln_1(x), use_mask=use_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

    def _expand_prompt(self, prompt_tokens: torch.Tensor, ref_x: torch.Tensor):
        # prompt_tokens: [T, D] -> [T, N, D]
        return prompt_tokens.to(dtype=ref_x.dtype, device=ref_x.device) \
                            .expand(ref_x.shape[1], -1, -1) \
                            .permute(1, 0, 2) \
                            .contiguous()

    def _inject_deeper_prompt(self, x, compound_prompts_deeper, counter):
        if self.first_layer or len(compound_prompts_deeper) == 0:
            return x, counter

        if counter > len(compound_prompts_deeper) - 1:
            return x, counter

        p = self._expand_prompt(compound_prompts_deeper[counter], x)

        if not self.text_layer:
            # vision: prompt tokens are appended at the end
            prefix = x[:-self.compound_prompt_nctx, :, :] ##print("prefix shape:",prefix.shape) prefix shape: torch.Size([197, 4, 768])
            x = torch.cat([prefix, p], dim=0) ##print("visual_context shape:",visual_context.shape) visual_context shape: torch.Size([2, 4, 768]) 
        else:
            # text: prompt tokens are between SOS and suffix
            prefix = x[:1, :, :]
            suffix = x[1 + self.compound_prompt_nctx:, :, :]
            x = torch.cat([prefix, p, suffix], dim=0)

        counter += 1
        return x, counter

    def _split_main_and_prompt(self, x):
        T = self.compound_prompt_nctx

        if not self.text_layer:
            x_main = x[:-T, :, :] #torch.Size([197, 4, 768])
            y0 = x[-T:, :, :] #torch.Size([2, 4, 768])
            return x_main, y0, None, None

        prefix = x[:1, :, :] ##print("after text prefix x shape:",prefix.shape) after text prefix x shape: torch.Size([1, 50, 512])
        y0 = x[1:1 + T, :, :] #torch.Size([2, 50, 512])
        suffix = x[1 + T:, :, :] #torch.Size([74, 50, 512])>>
        x_main = torch.cat([prefix, suffix], dim=0)
        return x_main, y0, prefix, suffix
    
    def _summarize_main_tokens(self, x_main, T: int):
        """
        Produce a fixed-length anchor from x_main so the MLP sees a fixed sequence length.
        """
        if not self.text_layer:
            # vision: CLS token is a good anchor
            anchor =x_main #x_main[:1, :, :]
        else:
            # text: mean is safer than just SOS
            anchor =x_main #x_main.mean(dim=0, keepdim=True)

        return  anchor #anchor.expand(T, -1, -1).contiguous()
    
    def _update_z_mlp(self, x_main, y, z):
        T = y.shape[0]
        x_anchor = self._summarize_main_tokens(x_main, T)   # [T, N, D]
        seq = torch.cat([x_anchor, y, z], dim=0)            # [3T, N, D]
        seq = self.z_mixer(seq)
        return seq[-T:, :, :]

    def _update_y_mlp(self, y, z):
        T = y.shape[0]
        seq = torch.cat([y, z], dim=0)                      # [2T, N, D]
        seq = self.y_mixer(seq)
        return seq[:T, :, :]

    def _update_z(self, x_main, y, z):
        # z <- f(x + y + z)
        seq = torch.cat([x_main, y, z], dim=0)
        seq = self._block(seq, use_mask=False)
        return seq[-z.shape[0]:, :, :] #torch.Size([2, 50, 512]) or torch.Size([2, 4, 768])

    def _update_y(self, y, z):
        # y <- f(y + z)
        seq = torch.cat([y, z], dim=0)
        seq = self._block(seq, use_mask=False)
        return seq[:y.shape[0], :, :] #torch.Size([2, 50, 512]) or torch.Size([2, 4, 768])

    def _trm_rollout(self, x_main, y, z, n_steps: int, grad_enabled: bool):
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            """for _ in range(n_steps):
                z = self.attn_pooling(z,x_main,x_main)
                #z = self._update_z(x_main, y, z)
            y=self.attn_pooling(y,z,z)"""
            #y = self._update_y(y, z)
            
            if self.use_mlp_trm:
                for _ in range(n_steps):  
                    z = self._update_z_mlp(x_main, y, z)
                y = self._update_y_mlp(y, z)
            else:
                for _ in range(n_steps):    
                    # fallback to your previous attention-pooling version
                    #z = self.attn_pooling(z, x_main, x_main)
                    #print("x_main shape for attn pooling:", x_main.shape) #torch.Size([197, 4, 768]) or torch.Size([77, 50, 512])
                    #print("y shape for attn pooling:", y.shape) #torch.Size([2, 4, 768]) or torch.Size([2, 50, 512])
                    #print("z shape for attn pooling:", z.shape) #torch.Size([2, 4, 768]) or torch.Size([2, 50, 512])
                    z = self.attn_pooling(x_main, y,z)
                y = self.attn_pooling(y, z, z)
        return y, z

    def forward(self, inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]

        # inject deeper prompt tokens first
        x, counter = self._inject_deeper_prompt(x, compound_prompts_deeper, counter)

        # split main tokens and prompt tokens
        x_main, y0, prefix, suffix = self._split_main_and_prompt(x)
        
        """if (counter >3) and (counter<= len(compound_prompts_deeper)-1):   #(counter >2) and 
            # load deep-supervision states if shapes match
            if (
                self._ds_prev_y is not None
                and self._ds_prev_z is not None
                and self._ds_prev_y.shape == y0.shape
                and self._ds_prev_z.shape == y0.shape
            ):
                y = self._ds_prev_y.to(dtype=y0.dtype, device=y0.device)
                z = self._ds_prev_z.to(dtype=y0.dtype, device=y0.device)
            else:
                y = y0
                z = torch.zeros_like(y0)

            # warmup refinement without gradients
            if self.trm_warmup > 0:
                for _ in range(self.trm_warmup):
                    y, z = self._trm_rollout(y0, y, z, self.trm_warmup, grad_enabled=False)
                #y = y.detach()
                #z = z.detach()

            # final refinement with gradients
            if self.trm_steps > 0:
                y, z = self._trm_rollout(y0, y, z, self.trm_steps, grad_enabled=True)

            # save state for next supervision step
            self._ds_prev_y = y.detach()
            self._ds_prev_z = z.detach()
            y = y + y0  # residual connection for prompt tokens
            # rebuild sequence with refined prompt tokens only
            if not self.text_layer:
                x = torch.cat([x_main, y], dim=0)
            else:
                x = torch.cat([prefix, y, suffix], dim=0)

        else:
            if not self.text_layer:
                x = torch.cat([x_main, y0], dim=0)
            else:
                x = torch.cat([prefix, y0, suffix], dim=0)
        # normal frozen CLIP block
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))"""
        #intermidiate recursion end
        if not self.text_layer:
                x = torch.cat([x_main, y0], dim=0)
        else:
                x = torch.cat([prefix, y0, suffix], dim=0)
        # normal frozen CLIP block
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))        


        """if (counter >6) and (counter<= len(compound_prompts_deeper)-1): 
            # normal frozen CLIP block
            #x = x + self.attention(self.ln_1(x))
            #x = x + self.mlp(self.ln_2(x))
            if (
                self._ds_prev_y is not None
                and self._ds_prev_z is not None
                and self._ds_prev_y.shape == y0.shape
                and self._ds_prev_z.shape == y0.shape
            ):
                y = self._ds_prev_y.to(dtype=y0.dtype, device=y0.device)
                z = self._ds_prev_z.to(dtype=y0.dtype, device=y0.device)
            else:
                y = y0
                z = torch.zeros_like(y0)
            
   
            for i in range(self.trm_warmup + self.trm_steps):
                    y = y + y0  # residual connection for prompt tokens
                    # rebuild sequence with refined prompt tokens only
                    if not self.text_layer:
                        x = torch.cat([x_main, y], dim=0)
                    else:
                        x = torch.cat([prefix, y, suffix], dim=0) 
                    x = x + self.attention(self.ln_1(x))
                    x = x + self.mlp(self.ln_2(x))
                    _j, y, pr, su = self._split_main_and_prompt(x)
            _j, y, pr, su = self._split_main_and_prompt(x)        
            self._ds_prev_y = y.detach()
            self._ds_prev_z = z.detach()

        else:
            if not self.text_layer:
                x = torch.cat([x_main, y0], dim=0)
            else:
                x = torch.cat([prefix, y0, suffix], dim=0)
            # normal frozen CLIP block
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))"""    


        return [x, compound_prompts_deeper, counter]


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()
        #self.H_cycles = 2
        #self.L_cycles = 2
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']

        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    
    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.text_layer:
             self.q_head = CastedLinearvision(x.shape[2], 2, bias=True)
          
        """if len(inputs) == 3:
           x, compound_prompts_deeper, counter = inputs
           halt_mask = None
        else:
           x, compound_prompts_deeper, counter, halt_mask = inputs"""

        #x_in = x  # for freezing halted samples
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both
                #print("before deep layer concatenation, x shape:",x.shape) torch.Size([199, 4, 768])
                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1): #even it is 12 layer transformer block only upto 9 layers learnable prompts will be injected.
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        #print("prefix shape:",prefix.shape) prefix shape: torch.Size([197, 4, 768])
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        #print("visual_context shape:",visual_context.shape) visual_context shape: torch.Size([2, 4, 768])
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)
                        #print("after deep layer concatenation, x shape:",x.shape) torch.Size([199, 4, 768])
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        #print("before text prefix x shape:",x.shape)>before text prefix x shape: torch.Size([77, 50, 512])
                        prefix = x[:1, :, :]
                        #print("after text prefix x shape:",prefix.shape) after text prefix x shape: torch.Size([1, 50, 512])
                        suffix = x[1 + self.compound_prompt_nctx:, :, :] #torch.Size([74, 50, 512])>>
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter] #textual context shape: torch.Size([2, 512])
                        #print("textual context shape:",textual_context.shape) textual context shape: torch.Size([2, 512])
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        #print("textual context expanded shape:",textual_context.shape)textual context expanded shape: torch.Size([2, 50, 512])
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        #print("x concatenated shape:",x.shape)x concatenated shape: torch.Size([77, 50, 512])
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]  # return again as a list, so that nn.seq can work

    def effective_layer_cost(self) -> int:
        # Vanilla MaPLe: count each transformer block as 1 "effective layer"
        # If you want attention+MLP as 2 units, change to return 2.
        return 1    

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        
        elif current_trainer == 'HRMMaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_HRMMaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        elif current_trainer == 'HighEncoderMaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_HigherEncoderMaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])         
        elif current_trainer == 'HiCroPL' or current_trainer == 'MPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_HiCroPL(width, heads, attn_mask, True,
                                                                         # The fourth parameter indicates whether a prompt needs to be added (true or false)
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_HiCroPL(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            """self.resblocks = nn.Sequential(*[ResidualAttentionBlock_HiCroPLReason(width, heads, layers, attn_mask, True,
                                                                         # The fourth parameter indicates whether a prompt needs to be added (true or false)
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_HiCroPLReason(width, heads, layers, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])"""
        elif current_trainer == 'HiCroPLReason' or current_trainer == 'MPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_HiCroPLReason(width, heads, layers, attn_mask, True,
                                                                         # The fourth parameter indicates whether a prompt needs to be added (true or false)
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_HiCroPLReason(width, heads, layers, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])   
        
        else:
            # Corresponds to default CoOp or CoCoOp
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    

class VisionTransformer_HiCroPL(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, img_prompts, cross_prompts_visual_deeper):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = img_prompts.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer([x, cross_prompts_visual_deeper])
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x    


class VisionTransformer_HiCroPLReason(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def _vision_get_prompt_tokens(self, x, n_ctx):
        """
        x: [197 + n_ctx, B, 768]
        return visual prompt slots: [n_ctx, B, 768]
        """
        return x[-n_ctx:, :, :]


    def _vision_replace_prompt_tokens(self, x, prompt_tokens, n_ctx):
        """
        Replace visual prompt slots at the tail.
        x: [197 + n_ctx, B, 768]
        prompt_tokens: [n_ctx, B, 768] or [n_ctx, 768]
        """
        if prompt_tokens.dim() == 2:
            prompt_tokens = prompt_tokens.unsqueeze(1).expand(-1, x.shape[1], -1)

        prompt_tokens = prompt_tokens.to(dtype=x.dtype, device=x.device)

        prefix = x[:-n_ctx, :, :]
        return torch.cat([prefix, prompt_tokens], dim=0)


    def _run_block_payload(
        self,
        blk,
        payload,
        disable_prompt_injection=False,
        prompt_override=None,
    ):
        """
        Run one visual transformer block with optional TRM-control flags.
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

    """def forward(self, x: torch.Tensor, img_prompts, cross_prompts_visual_deeper, init_cross_prompts_visual):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        # raw frozen visual tokens: before prompt concat
        with torch.no_grad():
            frozen_x = self.ln_pre(x)
            frozen_x = frozen_x.permute(1, 0, 2).detach()   # NLD -> LND
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = img_prompts.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #outputs = self.transformer([x, cross_prompts_visual_deeper])
        outputs = self.transformer([x, cross_prompts_visual_deeper, frozen_x, None, init_cross_prompts_visual])
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x"""
    """def forward(self,x: torch.Tensor, img_prompts, cross_prompts_visual_deeper, init_cross_prompts_visual, init_cross_prompts_text, return_layer_states=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )
        x = x + self.positional_embedding.to(x.dtype)

        with torch.no_grad():
            frozen_x = self.ln_pre(x)
            frozen_x = frozen_x.permute(1, 0, 2).detach()

        if self.VPT_shallow:
            visual_ctx = img_prompts.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        init_payload = (init_cross_prompts_text, init_cross_prompts_visual)
        payload = [x, cross_prompts_visual_deeper, frozen_x, None, init_payload]

        layer_states = []
        for blk in self.transformer.resblocks:
            payload = blk(payload)
            layer_states.append(payload[0])   # [L_with_prompts, B, 768]

        x = payload[0]
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        if return_layer_states:
            return x, layer_states

        return x"""
    """def forward(
        self,
        x: torch.Tensor,
        img_prompts,
        cross_prompts_visual_deeper,
        init_cross_prompts_visual,
        init_cross_prompts_text,
        return_layer_states=False
    ):"""
    def forward(
        self,
        x: torch.Tensor,
        img_prompts,
        cross_prompts_visual_deeper,
    ):
        x = self.conv1(x)

        x = x.reshape(
            x.shape[0],
            x.shape[1],
            -1,
        )

        x = x.permute(0, 2, 1)

        class_token = (
            self.class_embedding.to(x.dtype)
            + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        )

        x = torch.cat(
            [class_token, x],
            dim=1,
        )

        x = x + self.positional_embedding.to(x.dtype)

        if self.VPT_shallow:
            visual_ctx = img_prompts.to(
                dtype=x.dtype,
                device=x.device,
            ).expand(
                x.shape[0],
                -1,
                -1,
            )

            x = torch.cat(
                [x, visual_ctx],
                dim=1,
            )

        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        payload = [
            x,
            cross_prompts_visual_deeper,
        ]

        for block in self.transformer.resblocks:
            payload = block(payload)

        x = payload[0]
        x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    

class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        #print("after convolution shape:",x.shape) torch.Size([4, 768, 14, 14])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [batch, 768, 14 ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1(196+1), width]
        x = x + self.positional_embedding.to(x.dtype)
        #print("shape after adding positional embedding:",x.shape) shape after adding positional embedding: torch.Size([4, 197, 768])
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
            #print("shape after adding visual_ctx:",x.shape) shape after adding visual_ctx: torch.Size([4, 199, 768])
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD (Batch,seq_len,dim) -> LND(seq_len,Batch,dim)
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
        x = outputs[0]
        #q_logits = outputs[3]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj #[B, output_dim]

        return x #q_logits
    
    def forward_intermediates(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        """
        Returns:
          feats_per_block: list of [B, output_dim] image features after each transformer block
          effective_depths: list of cumulative effective depth after each block
        """
        # ----- same tokenization as forward() -----
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1
        )
        x = x + self.positional_embedding.to(x.dtype)

        if self.VPT_shallow:
            # IMPORTANT: match dtype/device of x
            visual_ctx = shared_ctx.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # ----- manual per-block stepping -----
        feats_per_block = []
        effective_depths = []
        cum_depth = 0
        counter = 0

        for blk in self.transformer.resblocks:
            out = blk([x, compound_deeper_prompts, counter])
            x, _, counter = out  # x is still LND

            # extract feature at this depth (same as forward)
            x_nld = x.permute(1, 0, 2)       # LND -> NLD
            feat = self.ln_post(x_nld[:, 0, :])
            if self.proj is not None:
                feat = feat @ self.proj
            feats_per_block.append(feat)

            cost = blk.effective_layer_cost() if hasattr(blk, "effective_layer_cost") else 1
            cum_depth += cost
            effective_depths.append(cum_depth)

        return feats_per_block, effective_depths


class VisionTransformer_HigherEncoderMaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        #print("after convolution shape:",x.shape) torch.Size([4, 768, 14, 14])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [batch, 768, 14 ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1(196+1), width]
        x = x + self.positional_embedding.to(x.dtype)
        #print("shape after adding positional embedding:",x.shape) shape after adding positional embedding: torch.Size([4, 197, 768])
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
            #print("shape after adding visual_ctx:",x.shape) shape after adding visual_ctx: torch.Size([4, 199, 768])
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD (Batch,seq_len,dim) -> LND(seq_len,Batch,dim)
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
        x = outputs[0]
        q_logits = outputs[3]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj #[B, output_dim]

        return x, q_logits      
    
class VisionTransformer_HRMMaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        #print("after convolution shape:",x.shape) torch.Size([4, 768, 14, 14])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [batch, 768, 14 ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1(196+1), width]
        x = x + self.positional_embedding.to(x.dtype)
        #print("shape after adding positional embedding:",x.shape) shape after adding positional embedding: torch.Size([4, 197, 768])
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            #visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            visual_ctx = shared_ctx.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)
            #print("shape after adding visual_ctx:",x.shape) shape after adding visual_ctx: torch.Size([4, 199, 768])
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD (Batch,seq_len,dim) -> LND(seq_len,Batch,dim)
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
        x = outputs[0]
        #q_logits = outputs[3]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj #[B, output_dim]

        return x # , q_logits

    def forward_intermediates(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        """
        Returns:
          img_feats_per_block: list length = num_blocks, each [B, embed_dim]
          halt_logits_per_block: list length = num_blocks, each [B] or None
          effective_depths: list length = num_blocks, each int (cumulative effective layers)
        """
        # ---- same stem as forward() ----
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)

        if self.VPT_shallow:
            #visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            visual_ctx = shared_ctx.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        state = [x, compound_deeper_prompts, 0]  # [tokens, deep_prompts, counter]

        img_feats_per_block = []
        halt_logits_per_block = []
        effective_depths = []
        cum_eff = 0

        for blk in self.transformer.resblocks:
            state = blk(state)
            x_lnd = state[0]

            q_logits = None
            if isinstance(state, (list, tuple)) and len(state) >= 4:
                q_logits = state[3]

            # effective depth accounting
            if hasattr(blk, "effective_layer_cost"):
                cum_eff += int(blk.effective_layer_cost())
            else:
                cum_eff += 1  # fallback
            effective_depths.append(cum_eff)

            # compute image embedding at this outer-block depth
            x_nld = x_lnd.permute(1, 0, 2)      # LND -> NLD
            feat = self.ln_post(x_nld[:, 0, :]) # CLS
            if self.proj is not None:
                feat = feat @ self.proj

            img_feats_per_block.append(feat)

            if q_logits is None:
                halt_logits_per_block.append(None)
            else:
                halt_logits_per_block.append(q_logits[:, 0].float())
        return img_feats_per_block, halt_logits_per_block, effective_depths
    


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer == "MaPLe":
                self.visual = VisionTransformer_MaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif trainer == "HRMMaPLe":         
                self.visual = VisionTransformer_HRMMaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )    
            elif trainer == "HighEncoderMaPLe":          
                self.visual = VisionTransformer_HigherEncoderMaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif trainer == "HiCroPL":
                self.visual = VisionTransformer_HiCroPL(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details,
                )     
            elif trainer == "HiCroPLReason":
                self.visual = VisionTransformer_HiCroPLReason(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details,
                )       
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
