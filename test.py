from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from refiners.fluxion.utils import load_from_safetensors, manual_seed
from refiners.foundationals.latent_diffusion.stable_diffusion_1.ella_adapter import SD1ELLAAdapter
from refiners.foundationals.latent_diffusion.ella_adapter import PerceiverAttentionBlock as _PerceiverAttentionBlock
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion import StableDiffusion_1

test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
flag = 1

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))

class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        ln2_output = self.ln_2(x, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, ln2_output], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)

        for i, p_block in enumerate(self.perceiver_blocks):
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class ELLA(nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width=768,
        layers=6,
        heads=8,
        num_latents=64,
        input_dim=2048,
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)
        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )

        return encoder_hidden_states


if __name__ == "__main__":
    manual_seed(2)
    llm_text_embedding = torch.rand(1, 128, 2048, dtype=dtype)
    time_step = torch.randint(0, 320, (1, 1))
    # latents = torch.rand(1, 64, 768, dtype=dtype)
    # time_embedding = torch.rand(1, 1, 768, dtype=dtype)
    # class TestChain(fl.Chain):
    #     def init_context(self) -> dict:
    #         return {"diffusion": {"timestep": None},
    #                 "ella": {"timestep_embedding": time_embedding, "latents": None}}
    
    # sd15 = StableDiffusion_1(device=test_device, dtype=dtype)
    unet = SD1UNet(in_channels=4, device=test_device, dtype=dtype)
    unet.set_timestep(time_step)
    unet.set_clip_text_embedding(torch.rand(1, 77, 768, dtype=dtype))

    adapter = SD1ELLAAdapter(target= unet, weights=load_from_safetensors("tests/weights/ELLA-Adapter/ella-sd1.5-tsc-t5xl.safetensors"))
    adapter.inject()
    adapter.set_llm_text_embedding(llm_text_embedding)

    ours = adapter.latents_encoder

    theirs = ELLA()
    theirs.to(device=test_device, dtype=dtype)
    theirs.load_state_dict(load_from_safetensors("tests/weights/QQGYLab/ELLA/ella-sd1.5-tsc-t5xl.safetensors"))

    y1 = ours(torch.rand(1, 1))
    y2 = theirs(llm_text_embedding, time_step)

    # ours = TestChain(_PerceiverAttentionBlock(768, 8, 768))
    # theirs = PerceiverAttentionBlock(768, 8, 768)
    # q_w, k_w, v_w = theirs.attn.state_dict()["in_proj_weight"].chunk(3, dim=0)
    # q_b, k_b, v_b = theirs.attn.state_dict()["in_proj_bias"].chunk(3, dim=0)
    # weights = {"Distribute.Linear_1.weight" : q_w, "Distribute.Linear_1.bias" : q_b,
    #                "Distribute.Linear_2.weight" : k_w, "Distribute.Linear_2.bias" : k_b,
    #                 "Distribute.Linear_3.weight" : v_w, "Distribute.Linear_3.bias" : v_b,
    #                 "Linear.weight": theirs.attn.state_dict()['out_proj.weight'], "Linear.bias": theirs.attn.state_dict()['out_proj.bias']}
    
    # ours[0]["Sum"]["Parallel"]["Chain"]["Attention"].load_state_dict(weights)

    # weights = {"weight" : theirs.mlp.state_dict()["c_fc.weight"], "bias" : theirs.mlp.state_dict()["c_fc.bias"]}
    # print(ours[0]["Residual"]["Chain"].state_dict().keys())
    # ours[0]["Residual"]["Chain"]["Linear_1"].load_state_dict(weights)
    # weights = {"weight" : theirs.mlp.state_dict()["c_proj.weight"], "bias" : theirs.mlp.state_dict()["c_proj.bias"]}
    # ours[0]["Residual"]["Chain"]["Linear_2"].load_state_dict(weights)
    # y1 = ours(llm_text_embedding, latents)
    # y2 = theirs(llm_text_embedding, latents , time_embedding)

    if isinstance(y1, Tensor):
        y1 = [y1]
        y2 = [y2]
    for yy1, yy2 in zip(y1, y2):
        print(yy1.shape, yy2.shape)
        print(yy1[0, 0, :10])
        print(yy2[0, 0, :10])
        print(torch.allclose(yy1, yy2))