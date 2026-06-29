"""VLM family registry for the unified VLM entry point (``vlm_eval.py``).

Each family (Gemma / InternVL / Qwen) differs only in how its vision-language
model is loaded and where the vision encoder, attention out-projection and
patch geometry live. ``load_vlm(family, size_idx, device)`` hides those
differences behind a single ``SimpleNamespace`` so the rest of the pipeline is
family-agnostic.

The returned bundle exposes:
    model, processor, model_path       - the HF model + processor + repo id
    get_out_proj(model, layer)         - attn out-projection for head ablation
    vision_encoder                     - module fed to the attention hook
    attn_fn(image, grid_thw)           - family attention-score hook (returns
                                         xs, qs, ks, vs, attention_scores)
    grad_fn(analytical_gradient)       - per-family gradient fixup before scoring
    preprocess(pil)                    - image -> (input_tensor, grid_thw)
    blocks, depth, n_heads, head_dim   - vision-encoder geometry
    layers                             - candidate layers for circuit mining
    image_size, patch_size            - token-grid geometry for mask building

Model sizes stay an index into the family's path list (``--size``), preserving
the original ``sys.argv[1]`` selection behaviour.
"""

import os
from types import SimpleNamespace

import torch
from torchvision import transforms

from vlm_common import (
    get_attention_scores_gemma,
    get_attention_scores_ivl,
    get_attention_scores_qwen,
    init_qwen_model,
)

GEMMA_PATHS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
]
IVL_PATHS = [
    "OpenGVLab/InternVL3_5-8B-HF",
    "OpenGVLab/InternVL3_5-4B-HF",
    "OpenGVLab/InternVL3_5-14B-HF",
]
QWEN_PATHS = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
]

FAMILY_PATHS = {"gemma": GEMMA_PATHS, "ivl": IVL_PATHS, "qwen": QWEN_PATHS}


def _make_preprocess(processor, resize_and_crop):
    def preprocess(data):
        data = resize_and_crop(data)
        input_tensor, grid_thw = processor.image_processor(data, return_tensors="pt").values()
        return input_tensor, grid_thw
    return preprocess


def _load_gemma(size_idx):
    from huggingface_hub import login
    from transformers import AutoProcessor, AutoModelForImageTextToText

    login(token=os.environ.get("HF_TOKEN", ""))
    model_path = GEMMA_PATHS[size_idx]
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, device_map="cuda", cache_dir="./cache").eval()
    processor = AutoProcessor.from_pretrained(model_path, cache_dir="./cache")

    vision_encoder = model.model.vision_tower.vision_model
    blocks = vision_encoder.encoder.layers
    depth = len(blocks)
    resize_and_crop = transforms.Compose([
        transforms.Resize((224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    return SimpleNamespace(
        model=model, processor=processor, model_path=model_path,
        get_out_proj=lambda m, l: m.model.vision_tower.vision_model.encoder.layers[l].self_attn.out_proj,
        vision_encoder=vision_encoder,
        attn_fn=lambda image, grid_thw: get_attention_scores_gemma(vision_encoder, image, layer=-1),
        grad_fn=lambda g: g,
        preprocess=_make_preprocess(processor, resize_and_crop),
        blocks=blocks, depth=depth,
        n_heads=blocks[0].self_attn.num_heads,
        head_dim=blocks[0].self_attn.head_dim,
        layers=[x for x in range(depth) if x >= depth * 0.5],
        image_size=896,  # SigLIP processes at 896 regardless of the 224 pre-crop
        patch_size=vision_encoder.embeddings.patch_embedding.kernel_size[0],
    )


def _load_ivl(size_idx):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_path = IVL_PATHS[size_idx]
    processor = AutoProcessor.from_pretrained(
        model_path, cache_dir="./cache", trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, device_map="cuda", torch_dtype=torch.bfloat16,
        cache_dir="./cache", trust_remote_code=True).eval()

    vision_encoder = model.model.vision_tower
    blocks = vision_encoder.encoder.layer
    depth = len(blocks)
    resize_and_crop = transforms.Compose([
        transforms.Resize((448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
    ])
    return SimpleNamespace(
        model=model, processor=processor, model_path=model_path,
        get_out_proj=lambda m, l: m.model.vision_tower.encoder.layer[l].attention.projection_layer,
        vision_encoder=vision_encoder,
        attn_fn=lambda image, grid_thw: get_attention_scores_ivl(vision_encoder, image, layer=-1),
        grad_fn=lambda g: g[:, 1:],  # InternVL: drop CLS token before scoring
        preprocess=_make_preprocess(processor, resize_and_crop),
        blocks=blocks, depth=depth,
        n_heads=blocks[0].attention.num_heads,
        head_dim=blocks[0].attention.head_dim,
        layers=[x for x in range(depth) if x >= depth * 0.5],
        image_size=448,
        patch_size=vision_encoder.embeddings.patch_embeddings.patch_size[0],
    )


def _load_qwen(size_idx):
    model_path = QWEN_PATHS[size_idx]
    model, processor, get_out_proj = init_qwen_model(model_path)

    vision_encoder = model.model.visual
    blocks = vision_encoder.blocks
    depth = len(blocks)
    resize_and_crop = transforms.Compose([
        transforms.Resize((256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
    ])
    return SimpleNamespace(
        model=model, processor=processor, model_path=model_path,
        get_out_proj=get_out_proj,
        vision_encoder=vision_encoder,
        attn_fn=lambda image, grid_thw: get_attention_scores_qwen(vision_encoder, image, grid_thw),
        grad_fn=lambda g: g,
        preprocess=_make_preprocess(processor, resize_and_crop),
        blocks=blocks, depth=depth,
        n_heads=blocks[0].attn.num_heads,
        head_dim=blocks[0].attn.head_dim,
        layers=[x for x in range(depth) if x >= depth * 0.8],
        image_size=256,
        patch_size=vision_encoder.patch_embed.patch_size,
    )


_LOADERS = {"gemma": _load_gemma, "ivl": _load_ivl, "qwen": _load_qwen}


def load_vlm(family, size_idx):
    """Load a VLM family at the given size index. See module docstring for the
    returned bundle's attributes."""
    if family not in _LOADERS:
        raise ValueError(f"Unknown family '{family}'. Choices: {sorted(_LOADERS)}")
    paths = FAMILY_PATHS[family]
    if not (0 <= size_idx < len(paths)):
        raise ValueError(f"--size {size_idx} out of range for '{family}' (have {len(paths)}: {paths})")
    return _LOADERS[family](size_idx)
