"""Shared building blocks for the CLIP typographic-attack classification scripts.

Both classification entry modes (CV threshold-optimisation and z-sigma sweep)
share the same model/SAE loading + circuit-mining prefix and the same per-dataset
evaluation inner loops. Those are factored out here so the unified ``clip_eval.py``
holds only the per-mode orchestration.

IMPORTANT — accuracy accumulation: the original ``typo_a.py`` (CV) accumulates
batch accuracy as ``mean()*N`` while ``typo_a_zsweep.py`` uses ``sum()``. These
are numerically near-identical but NOT bit-identical (float rounding). The
``accum`` argument ("mean" or "sum") preserves each mode's exact behaviour.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from model_training.sae import TopKSAE, ReLUSAE
from eval_utils import (
    create_masks, calcualte_score, get_attention_scores, get_attribution_map,
    fix_attn_head_list,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYER_DICT = {"vit_b16": 11, "vit_l14": 23, "vit_h14": 31, "vit_g14": 39, "vit_big_g14": 47}
BASE_DIR = "/p/realai/bohan/headsae/attributing-clip/configs/train_sae/imagenet"
CONFIG_SUFFIX = '_seed123_topk128'


# --- z-threshold head spec --------------------------------------------------

def build_layer_spec(head_scores, layers, n_heads, sigma):
    """Heads with score > layer_mean + sigma * layer_std (per layer, z-threshold)."""
    layer_spec = {}
    for layer in layers:
        layer_spec[layer] = []
        layer_scores = np.array([head_scores[(layer, head)] for head in range(n_heads)])
        layer_mean, layer_std = layer_scores.mean(), layer_scores.std()
        for head in range(n_heads):
            if head_scores[(layer, head)] > layer_mean + layer_std * sigma:
                layer_spec[layer].append(head)
    return layer_spec


def fmt_dt(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# --- random per-head SAE scaffolding (CLIP) --------------------------------

def build_random_sae_list(model, layers, config, config_name, config_name_suffix, device):
    sae_list = {}
    curr_config = config.copy()
    for layer in layers:
        curr_config['sae_layer_idx'] = layer
        curr_config['sae_k'] = int(config_name_suffix.split('topk')[-1])
        sae = TopKSAE.from_config(curr_config, model)
        curr_wo = model.transformer.resblocks[layer].attn.out_proj.weight.data
        curr_bo = model.transformer.resblocks[layer].attn.out_proj.bias.data
        curr_wo = curr_wo.reshape(model.transformer.resblocks[layer].attn.num_heads, -1, model.transformer.resblocks[layer].attn.head_dim)
        curr_ls = model.transformer.resblocks[layer].ls_1

        for i, encoder in enumerate(sae.encoders):
            wo = curr_wo[i]
            encoder[0].weight.data = torch.randn_like(encoder[0].weight.data).to(device)
            encoder[0].bias.data = torch.randn_like(encoder[0].bias.data).to(device)
            encoder[0].weight.data = F.normalize(encoder[0].weight.data, dim=1, p=2)
        sae.per_head_recon = False
        sae_list[layer] = sae.to(device)
    return sae_list


# --- model + SAE loading ----------------------------------------------------

def load_clip_and_sae(model_name, config_fn, args):
    """Load the CLIP model and attach a (randomly-initialised) per-head SAE.

    Returns a dict with everything the mining + eval stages need.
    """
    import yaml
    from models import get_fn_model_loader

    layer = 0
    for key in LAYER_DICT:
        if key in model_name:
            layer = LAYER_DICT[key]
            break

    config_file = os.path.join(BASE_DIR, f'{model_name}', config_fn)
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    config_name = os.path.basename(config_file)[:-5]
    config["config_file"] = config_file

    print(f"Loading model: {model_name}")
    model = get_fn_model_loader(model_name)()
    n_heads = model.transformer.resblocks[0].attn.num_heads

    config['sae_n_heads'] = n_heads
    config['sae_layer_idx'] = layer
    config['sae_hidden_dim'] = args.expand_ratio

    if 'relu' in CONFIG_SUFFIX:
        config['sae_k'] = 0.001
        sae = ReLUSAE.from_config(config, model=model)
    else:
        sae = TopKSAE.from_config(config, model=model)

    model.add_sae(sae)
    model.eval()
    model = model.to(DEVICE)
    model.sae.per_head_recon = False
    preprocess = model.preprocess

    depth = len(model.transformer.resblocks)
    layers = [x for x in range(depth) if x >= depth * 0.8]
    sae_head_dim = model.sae.hidden_dim // n_heads
    print(f"Calculating head clarity scores for layers: {layers} with SAE head dim {sae_head_dim} and {n_heads} heads per layer.")

    return dict(model=model, config=config, config_name=config_name, layer=layer,
                n_heads=n_heads, layers=layers, sae_head_dim=sae_head_dim,
                preprocess=preprocess, depth=depth)


# --- circuit mining ---------------------------------------------------------

def mine_head_scores(model, dataset_val, config, config_name, n_heads, layers,
                     preprocess, args, model_name, seed, timings=None):
    """Run the random-SAE circuit-mining loop, returning (head_scores, output_dir)."""
    transform = RandomTextBorderTransform(
        [x.split(',')[0] for x in dataset_val.class_names], border_ratio=0.2, post_transform=preprocess)
    dataset_val.transform = transform
    subset = torch.utils.data.Subset(dataset_val, list(range(0, len(dataset_val), 1000)))
    inv_norm = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711])
    sample_img = inv_norm(subset[0][0][0]).permute(1, 2, 0).cpu().numpy()
    sample_img = Image.fromarray((sample_img * 255).astype(np.uint8))
    imgsave_path = os.path.join("data_cache", "attribution", f"{model_name}_sample_text_border.jpg")
    os.makedirs(os.path.dirname(imgsave_path), exist_ok=True)
    sample_img.save(imgsave_path)
    print(f"Saved sample text border image to {imgsave_path}")

    dl = DataLoader(subset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    image_size = model.preprocess.transforms[0].size
    patch_size = model.patch_size[0] if type(model.patch_size) is tuple else model.patch_size
    token_shape = (image_size // patch_size) ** 2
    head_latent_dim = model.sae.hidden_dim // n_heads
    print(f"Image size: {image_size}, Patch size: {patch_size}, Token shape: {token_shape}, SAE head latent dim: {head_latent_dim}")
    all_masks = create_masks(token_shape, text_depth=int(np.ceil(image_size * 0.2 / patch_size))).to(DEVICE)

    sae_list = build_random_sae_list(model, layers=layers, config=config,
                                     config_name=config_name, config_name_suffix=CONFIG_SUFFIX, device=DEVICE)

    with torch.inference_mode() and torch.autocast(device_type='cuda', enabled=args.autocast):
        head_scores = {}
        total_samples = 0
        for step, batch in enumerate(tqdm(dl, total=len(dl), desc="Circuit mining (head scores)")):
            image_data, image_label = batch
            images, text_loc = image_data
            images = images.to(DEVICE)
            total_samples += images.shape[0]
            _, _, _, vs, attention_scores = get_attention_scores(model, images)
            for layer in layers:
                sae = sae_list[layer]
                for head in range(n_heads):
                    sae_w = sae.encoders[head][0].weight_v.data
                    sae_b = sae.encoders[head][0].bias.data
                    cls_attention_softmax, analytical_gradient = get_attribution_map(
                        attention_scores, vs, sae_w, sae_b, layer=layer, head=head)
                    final_scores = calcualte_score(text_loc, analytical_gradient, all_masks).sum(0)
                    if (layer, head) not in head_scores:
                        head_scores[(layer, head)] = torch.zeros(head_latent_dim).to(DEVICE)
                    head_scores[(layer, head)] += final_scores

    output_dir = f"typo_results_{args.expand_ratio}_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    layer_head_score_path = os.path.join(output_dir, f"{model_name}_head_scores.pt")
    torch.save(head_scores, layer_head_score_path)

    for i in layers:
        for head in range(n_heads):
            head_scores[(i, head)] = head_scores[(i, head)].mean().item() / len(subset)

    return head_scores, output_dir


# --- text encoder + class embeddings ---------------------------------------

def load_text_encoder(model_name):
    from models import get_fn_model_loader
    text_model = get_fn_model_loader(model_name)(return_clip_model=True)
    del text_model.visual
    text_model = text_model.to(DEVICE).eval()
    return text_model


def encode_class_labels(text_model, class_labels, skip_none=True):
    """Encode "a photo of a <label>." prompts into normalised CLIP text embeddings."""
    class_embeddings = []
    for i in range(0, len(class_labels), 64):
        batch_labels = class_labels[i:i + 64]
        if skip_none:
            batch_labels = ["a photo of a " + label + "." for label in batch_labels if label != 'none']
        else:
            batch_labels = ["a photo of a " + label + '.' for label in batch_labels]
        with torch.no_grad():
            batch_emb = text_model.encode_text(text_model.tokenizer(batch_labels).to(DEVICE))
            batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
            class_embeddings.append(batch_emb)
    return torch.cat(class_embeddings, dim=0).to(DEVICE)


# --- shared evaluation inner loops -----------------------------------------

def _add(cum, correct_tensor, n, accum):
    """Accumulate batch correctness, preserving the per-mode float behaviour."""
    if accum == "mean":
        return cum + correct_tensor.float().mean().item() * n
    return cum + correct_tensor.float().sum().item()


def evaluate_attack_dataset(model, ds, dl, class_embeddings, layer_spec, args, accum):
    """Object/attack accuracy on an attack dataset under a fixed head spec."""
    cum_acc = 0.0
    cum_attack_acc = 0.0
    total_samples = 0
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=args.autocast):
        model.use_sae = False
        for batch in tqdm(dl, total=len(dl), desc=f"Evaluating {ds.ds_name}"):
            images = batch['image'].to(DEVICE)
            with fix_attn_head_list(model, layer_spec, input_data=None):
                image_features = model(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                sims = image_features @ class_embeddings.t()
                preds = sims.argmax(dim=-1)
                labels = [ds.class_names.index(ol) for ol in batch['object_label']]
                attack_labels = [ds.class_names.index(aw) for aw in batch['attack_word']]
                cum_acc = _add(cum_acc, preds.cpu() == torch.tensor(labels), images.shape[0], accum)
                cum_attack_acc = _add(cum_attack_acc, preds.cpu() == torch.tensor(attack_labels), images.shape[0], accum)
                total_samples += images.shape[0]
    return cum_acc / total_samples, cum_attack_acc / total_samples, total_samples


def evaluate_imagenet(model, imnet_dl, class_embeddings, args, accum, layer_spec=None):
    """ImageNet-100 accuracy, optionally under a fixed head spec (alpha=1.0)."""
    from contextlib import nullcontext
    cum_acc = 0.0
    total_samples = 0
    ctx = fix_attn_head_list(model, layer_spec, alpha=1.0) if layer_spec is not None else nullcontext()
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=args.autocast), ctx:
        for batch in tqdm(imnet_dl, total=len(imnet_dl)):
            images = batch[0].to(DEVICE)
            image_features = model(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = image_features @ class_embeddings.t()
            preds = sims.argmax(dim=-1)
            cum_acc = _add(cum_acc, preds.cpu() == batch[1], images.shape[0], accum)
            total_samples += images.shape[0]
    return cum_acc / total_samples, total_samples


# RandomTextBorderTransform lives in proj_utils; re-export so mine_head_scores
# can use it without each caller importing it.
from proj_utils import RandomTextBorderTransform  # noqa: E402
