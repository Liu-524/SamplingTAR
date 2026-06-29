"""Shared utilities for the typo2 VLM evaluation scripts (Gemma / InternVL / Qwen).

These definitions were previously duplicated verbatim across every ``typo2_*.py``
entry point. They are model-agnostic except for the three ``get_attention_scores_*``
vision-encoder hooks, which differ by VLM architecture and are therefore exported
under explicit names; each script aliases the variant it needs to
``get_attention_scores``.

Note: ``RandomTextBorderTransform`` is intentionally NOT here — it already lives in
``proj_utils`` and the scripts import it from there. Helpers that close over
script-level globals (``preprocess``, ``make_gen_kwargs``, ``decode_samples``)
remain in their scripts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from contextlib import contextmanager



# --- SAE scaffolding --------------------------------------------------------

class DummySAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super(DummySAE, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(n_heads)
        ])


def build_random_sae_list(head_dim, n_heads, layers, device, expand_ratio=16):
    sae_list = {}

    for layer in layers:
        
        sae = DummySAE(head_dim, head_dim * expand_ratio, n_heads=n_heads)
        for encoder in sae.encoders:
            encoder[0].weight.data = torch.randn_like(encoder[0].weight.data)
            encoder[0].bias.data = torch.zeros_like(encoder[0].bias.data)
            encoder[0].weight.data = F.normalize(encoder[0].weight.data, dim=1, p=2)
        
        sae.per_head_recon = False
        # sae.add_residual = False
        sae_list[layer] = sae.to(device)
    return sae_list


# --- Attribution scoring ----------------------------------------------------

def calcualte_score(text_loc, score_maps, all_masks):
    masks = all_masks[text_loc]  # (B, 196)
    score_maps = score_maps[:, :]  # remove cls token
    score_maps = torch.clamp(score_maps, min=0.0, max=1.0)
    score_maps = score_maps.permute(0, 2, 1)  # (B, C, 196)
    pos_scores = (score_maps * masks.unsqueeze(1)).sum(-1)  # (B, C)
    neg_scores = (score_maps * (1 - masks).unsqueeze(1)).sum(-1)  # (B, C)
    final_scores = pos_scores / (pos_scores + neg_scores + 1e-6)
    return final_scores  # (B, C)


# --- Head ablation ----------------------------------------------------------

@contextmanager
def ablate_attn_head_list(model, layer_spec, get_out_proj, n_heads, input_data=None):
    # head_list: list of (layer, head)
    hooks = []
    def hook_fn(module, input, output, layer, heads):
        if input_data is not None:
            input = [input_data[layer]]
        if len(output.shape) == 3:
            B, N, C = output.shape
        else:
            N, C = output.shape
            B = 1
        head_dim = C // n_heads
        x = input[0].view(B, N, n_heads, head_dim)
        for head in heads:
            x[:, :, head, :] = 0.0
        if len(output.shape) == 2:
            x = x.view(N, C)
            return F.linear(x, module.weight, module.bias)
        x = x.view(B, N, C)
        return F.linear(x, module.weight, module.bias)

    for layer in layer_spec:
        heads = layer_spec[layer]
        hook = get_out_proj(model, layer).register_forward_hook(partial(hook_fn, layer=layer, heads=heads))
        hooks.append(hook)
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


# --- z-threshold sweep helpers ---------------------------------------------

def fmt_dt(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def build_layer_spec(layer_scores, layers, n_heads, sigma):
    """Heads with score > layer_mean + sigma * layer_std (z-threshold)."""
    spec = {}
    for layer in layers:
        spec[layer] = []
        ls = np.array(layer_scores[layer])
        mu, sd = ls.mean(), ls.std()
        thresh = mu + sd * sigma
        for head in range(n_heads):
            if layer_scores[layer][head] > thresh:
                spec[layer].append(head)
    return spec


# --- Vision-encoder attention hooks (architecture-specific) ----------------

def get_attention_scores_gemma(model, input_tensor,layer=-1):
    # This function remains unchanged
    qs, ks, vs, xs = {}, {}, {}, {}
    def get_qkv_hook(i, module, input, output):
        x = input[0]
        xs[i] = x
        N = 1
        B, L, C = x.shape
        attn = module.self_attn
        x = module.layer_norm1(x)
        q, k, v = F.linear(x, attn.q_proj.weight, attn.q_proj.bias), F.linear(x, attn.k_proj.weight, attn.k_proj.bias), F.linear(x, attn.v_proj.weight, attn.v_proj.bias)
        q = q.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        k = k.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        v = v.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        qs[i], ks[i], vs[i] = q, k, v
        return output

    hooks = []
    for i, block in enumerate(model.encoder.layers):
        if layer == -1 or i == layer:
            hook = block.register_forward_hook(partial(get_qkv_hook, i))
            hooks.append(hook)
    try:
        with torch.no_grad():
            _ = model(input_tensor)
    finally:
        for hook in hooks:
            hook.remove()

    attention_scores = {}
    for i in qs:
        d_k = qs[i].size(-1)
        scores = torch.matmul(qs[i], ks[i].transpose(-2, -1)) / np.sqrt(d_k)
        attention_scores[i] = scores
    return xs, qs, ks, vs, attention_scores


def get_attention_scores_ivl(model, input_tensor,layer=-1):
    # This function remains unchanged
    qs, ks, vs, xs = {}, {}, {}, {}
    def get_qkv_hook(i, module, input, output):
        x = input[0]
        xs[i] = x
        N = 1
        B, L, C = x.shape
        attn = module.attention
        x = module.layernorm_before(x)
        q, k, v = F.linear(x, attn.q_proj.weight, attn.q_proj.bias), F.linear(x, attn.k_proj.weight, attn.k_proj.bias), F.linear(x, attn.v_proj.weight, attn.v_proj.bias)
        q = q.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        k = k.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        v = v.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        qs[i], ks[i], vs[i] = q, k, v
        return output

    hooks = []
    for i, block in enumerate(model.encoder.layer):
        if layer == -1 or i == layer:
            hook = block.register_forward_hook(partial(get_qkv_hook, i))
            hooks.append(hook)
    try:
        with torch.no_grad():
            _ = model(input_tensor)
    finally:
        for hook in hooks:
            hook.remove()

    attention_scores = {}
    for i in qs:
        d_k = qs[i].size(-1)
        scores = torch.matmul(qs[i], ks[i].transpose(-2, -1)) / np.sqrt(d_k)
        attention_scores[i] = scores
    return xs, qs, ks, vs, attention_scores


def get_attention_scores_qwen(model, input_tensor, grid_thw,layer=-1):
    # This function remains unchanged
    qs, ks, vs, xs = {}, {}, {}, {}
    def get_qkv_hook(i, module, input, output):
        x = input[0]
        xs[i] = x
        N = 1
        L, C = x.shape
        attn = module.attn
        x = module.norm1(x)
        q, k, v = F.linear(x, attn.qkv.weight, attn.qkv.bias).chunk(3, dim=-1)
        q = q.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        k = k.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        v = v.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        qs[i], ks[i], vs[i] = q, k, v
        return output

    hooks = []
    for i, block in enumerate(model.blocks):
        if layer == -1 or i == layer:
            hook = block.register_forward_hook(partial(get_qkv_hook, i))
            hooks.append(hook)
    try:
        with torch.no_grad():
            _ = model(input_tensor, grid_thw)
    finally:
        for hook in hooks:
            hook.remove()

    attention_scores = {}
    for i in qs:
        d_k = qs[i].size(-1)
        scores = torch.matmul(qs[i], ks[i].transpose(-2, -1)) / np.sqrt(d_k)
        attention_scores[i] = scores
    return xs, qs, ks, vs, attention_scores


# --- Qwen-specific model helpers -------------------------------------------

def init_qwen_model(path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration

    # default: Load the model on the available device(s)

    if 'A' in path:
        model =Qwen3VLMoeForConditionalGeneration.from_pretrained(
            path, dtype="auto", device_map="auto", cache_dir="./cache"
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype="auto", device_map="auto", cache_dir="./cache"
        )
    processor = AutoProcessor.from_pretrained(
        path, cache_dir="./cache"
    )

    get_out_proj = lambda m, l: m.model.visual.blocks[l].attn.proj
    return model, processor, get_out_proj


def get_model_vars(model):
    blocks = model.model.visual.blocks
    depth = len(blocks)

    n_heads = model.model.visual.blocks[0].attn.num_heads
    layers = [x for x in range(depth) if x >= depth * 0.8]
    sae_head_dim = 36000 // n_heads
    return blocks, depth, n_heads, layers, sae_head_dim
