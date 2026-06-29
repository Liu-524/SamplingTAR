import logging
import os
import random
import time
import yaml
import argparse
import json
from functools import partial
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import re

try:
    from dataset_utils import get_dataset
    from model_training.sae import TopKSAE, ReLUSAE
    from models import get_fn_model_loader
    from proj_utils import get_ckpt_path_typo
    from experiments.attribution import get_top_activating_samples
    LAYER_DICT = {"vit_b16": 11, "vit_l14": 23, "vit_h14": 31, "vit_g14": 39, "vit_big_g14": 47}
    BASE_DIR = "/p/realai/bohan/headsae/attributing-clip/configs/train_sae/imagenet"
    DEFAULT_MODEL = 'clip_vit_l14_laion2b_s32b_b82k'
    CONFIG_SUFFIX = '_seed123_topk128'
    EXPAND_RATIO = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError as e:
    print(f"Warning: Project specific imports failed. Ensure your python path is correct. {e}")
# --- Utils: Model & Math ---

def get_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--config_fn", type=str, default="cfg-aux+transcode-siam.yaml")
    parser.add_argument("--cv_dataset", type=str, default="kaggle_imnet100_text")
    parser.add_argument("--autocast", action='store_true', help="Use autocast during evaluation")
    parser.add_argument("--seed", type=int, default=4222)
    parser.add_argument("--expand_ratio", type=int, default=16)
    return parser
def prepare_torch():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')


def get_last_modified_time(path):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))

def interpolate_positional_embedding(pos_emb, old_grid_size, new_grid_size):
    """
    Interpolate positional embeddings for different image sizes.
    """
    if old_grid_size == new_grid_size:
        return pos_emb

    class_token, pos_tokens = pos_emb[:1], pos_emb[1:]
    dim = pos_tokens.shape[-1]
    pos_tokens = pos_tokens.reshape(1, old_grid_size, old_grid_size, dim).permute(0, 3, 1, 2)

    pos_tokens = F.interpolate(
        pos_tokens, size=(new_grid_size, new_grid_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, dim)
    new_pos_emb = torch.cat((class_token.unsqueeze(0), pos_tokens), dim=1)
    return new_pos_emb

def resize_model(model, new_img_size=224):
    preprocess = model.preprocess
    preprocess.transforms[0].size = new_img_size
    preprocess.transforms[1].size = (new_img_size, new_img_size)

    if hasattr(model, "positional_embedding"):
        pos_emb = model.positional_embedding
        old_grid_size = int((pos_emb.shape[0] - 1) ** 0.5)
        new_grid_size = int(new_img_size / model.patch_size[0])
        print(f"Resizing positional embedding from {old_grid_size} to {new_grid_size}")
        new_pos_emb = interpolate_positional_embedding(pos_emb, old_grid_size, new_grid_size)
        model.positional_embedding = nn.Parameter(new_pos_emb)
    elif hasattr(model, "pos_embed"):
        pos_emb = model.pos_embed
        old_grid_size = int((pos_emb.shape[1] - 1) ** 0.5)
        new_grid_size = int(new_img_size / model.patch_size)
        new_pos_emb = interpolate_positional_embedding(pos_emb, old_grid_size, new_grid_size)
        model.pos_embed = nn.Parameter(new_pos_emb)
    else:
        print("No positional embedding found.")
    return model

@torch.inference_mode()
def clarity_score(V):
    # V.shape = (n_neurons) x n_samples x n_features
    V_nrmed = torch.nn.functional.normalize(V, dim=-1)
    clarity = ((V_nrmed.mean(-2).pow(2).sum((-1))) - 1 / V.shape[-2]) / (V.shape[-2] - 1) * V.shape[-2]
    return clarity

def get_save_path(base_dir, model_name, config_name, config_suffix, layer):
    model_dir = f"{model_name}_{config_name}-{layer}{config_suffix}"
    return os.path.join(base_dir, model_dir)

# --- Dataset Classes ---

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Get all image files in the directory
        
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

        self.class_names = sorted(list(set([fname.split('_')[0].split('=')[1] for fname in self.img_files] + [re.split(r"[\.=]", fname.split('_')[1])[1] for fname in self.img_files])))
        self.class_names = [x for x in self.class_names if len(x) > 0] + ['none']

        with open(os.path.join(img_dir, 'labels.txt'), 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        self.ds_name = os.path.basename(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB") # Ensure 3 channels
        
        if self.transform:
            image = self.transform(image)
        label = self.img_files[idx][:-4]  # Example: use filename (without extension) as label
        if len(label.split('_')) == 2:
            l, a = label.split('_')
        else:
            info = label.split('_')
            l, a = info[0], info[1]
        l = l.split('=')[1]
        a = a.split('=')[1]
        return {
            'image': image,
            'group': self.ds_name,
            'object_label': l,
            'attack_word': a if len(a) > 0 else 'none',
            'area_pct': -1.0,
            'img_id': self.img_files[idx]
        }


def collate_fn(batch, preprocess_fn):
    images = []
    groups = []
    object_labels = []
    attack_words = []
    area_pcts = []
    img_ids = []

    for item in batch:
        image, group, object_label, attack_word, area_pct, img_id = item.values()
        images.append(preprocess_fn(image))
        groups.append(group)
        object_labels.append(object_label)
        attack_words.append(attack_word)
        area_pcts.append(area_pct)
        img_ids.append(img_id)
    images = torch.stack(images, dim=0)
    return {
        'image': images,
        'group': groups,
        'object_label': object_labels,
        'attack_word': attack_words,
        'area_pct': area_pcts,
        'img_id': img_ids
    }

# --- Attribution & Clarity Logic ---

def get_head_clarity(model, features, dataset_val, model_name, config_name, expand_ratio, topk=20):
    layers = len(model.transformer.resblocks)
    n_heads = model.transformer.resblocks[0].attn.num_heads
    all_layer_scores = {}
    
    print("Calculating head clarity scores...")
    for i in range(layers):
        all_scores = {}
        neurons = []
        image_lists = []
        read_path = get_save_path(f"./results/res{expand_ratio}", model_name, config_name,CONFIG_SUFFIX, i)
        
        if not os.path.exists(read_path) or not os.path.exists(read_path + '/activation_to_top_activating_sample.pkl'):
            # print(f"Path {read_path} does not exist, skip layer {i}.")
            continue
            
        activation_to_top_activating_sample, _ = get_top_activating_samples(model, dataset_val, read_path, False)    
        
        for neuron_index, data in activation_to_top_activating_sample.items():
            if len(data) < topk:
                continue
            image_indices = [img_idx for _, img_idx in data[: topk]]
            neurons.append(neuron_index)
            image_lists.append(image_indices)
            
        for neuron, image_indices in zip(neurons, image_lists):
            neuron_head = neuron // (model.sae.hidden_dim // n_heads)
            if features.dim() == 2:
                cls_features = features[image_indices, :]
            else:
                cls_features = features[image_indices, 0, :]
            
            clarity = clarity_score(cls_features)
            if neuron_head not in all_scores:
                all_scores[neuron_head] = []
            all_scores[neuron_head].append(clarity.item())
        all_layer_scores[i] = all_scores
    return all_layer_scores

# --- Hooks & Context Managers ---

def dual_forward(model, images):
    data = {}
    def get_layer_inputs(module, input, output, i):
        data[i] = input[0].detach()

    mlp_data={}
    def get_mlp_layer_inputs(module, input, output, i):
        mlp_data[i] = output.detach()

    hooks = []
    for i in range(len(model.transformer.resblocks)):
        hook = model.transformer.resblocks[i].register_forward_hook(partial(get_layer_inputs, i=i))
        mlp_hook = model.transformer.resblocks[i].mlp.register_forward_hook(partial(get_mlp_layer_inputs, i=i))
        hooks.append(hook)
        hooks.append(mlp_hook)
    
    _ = model(images)
    
    for hook in hooks:
        hook.remove()
    return data, mlp_data


@contextmanager
def fix_attn_head_list(model, layer_spec, input_data=None, alpha=1.0):
    # layer_spec: dict of {layer_idx: [head_indices]}
    hooks = []
    
    def hook_fn(module, input, output, layer, heads):
        B, L, C = output.shape
        # Recompute attention internals
        q, k, v = F.linear(input[0], module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(B, L, module.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, L, module.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, L, module.num_heads, -1).transpose(1, 2)
        q = module.ln_q(q)
        k = module.ln_k(k)

        att = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        att = att.softmax(dim=-1)

        # factors = att[:,:,:,1:].sum(dim=-1, keepdim=True)
        # for head in heads:
        #     att[:,head,:,0] = alpha
        #     att[:,head,:,1:] = att[:,head,:,1:] * (1 - alpha) / (factors[:,head,:,:] + 1e-6)          
        factors = att[:,:,:1,1:].sum(dim=-1, keepdim=True)
        for head in heads:
            att[:,head,:1,0] = alpha
            att[:,head,:1,1:] = att[:,head,:1,1:] * (1 - alpha) / (factors[:,head,:1,:] + 1e-6)          
            
        v = module.attn_drop(att @ v)
        x = v.transpose(1, 2).reshape(B, L, C)
        x = module.out_proj(x)
        x = module.out_drop(x)
        return x

    for layer in layer_spec:
        heads = layer_spec[layer]
        hook = model.transformer.resblocks[layer].attn.register_forward_hook(partial(hook_fn, layer=layer, heads=heads))
        hooks.append(hook)
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
@contextmanager
def ablate_attn_head_list(model, layer_spec, input_data=None):
    # head_list: list of (layer, head)
    hooks = []
    def hook_fn(module, input, output, layer, heads):
        if input_data is not None:
            input = [input_data[layer]]
        B, N, C = output.shape
        head_dim = C // model.sae.n_heads
        x = input[0].view(B, N, model.sae.n_heads, head_dim)
        for head in heads:
            x[:, :1, head, :] = 0.0
        x = x.view(B, N, C)
        return F.linear(x, module.weight, module.bias)

    for layer in layer_spec:
        heads = layer_spec[layer]
        hook = model.transformer.resblocks[layer].attn.out_proj.register_forward_hook(partial(hook_fn, layer=layer, heads=heads))
        hooks.append(hook)
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()

@contextmanager
def empty_context():
    yield

# --- Visualization (Optional) ---
def plot_sae_weights(model, n_heads=1):
    try:
        weight_g = model.sae.encoders[0][0].weight_g.data
        weight_v = model.sae.encoders[0][0].weight_v.data
        weight = weight_v * weight_g
        # Simple visualization logic
        weight = F.interpolate(weight.unsqueeze(0).unsqueeze(0), 
                             size=(weight.shape[1], weight.shape[1]), 
                             mode='bilinear', align_corners=False).squeeze()
        weight = weight.cpu()
        plt.figure()
        plt.imshow(weight, cmap='viridis')
        plt.colorbar()
        plt.title('SAE Weight Matrix')
        plt.show()
    except Exception as e:
        print(f"Skipping visualization: {e}")

# --- Main Logic ---


def get_attention_scores(model, input_tensor, layer=-1):
    # This function remains unchanged
    qs, ks, vs, xs = {}, {}, {}, {}
    def get_qkv_hook(i, module, input, output):
        x = input[0]
        xs[i] = x
        N, L, C = x.shape
        attn = module.attn
        x = module.ln_1(x)
        q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        k = k.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        v = v.reshape(N, L, attn.num_heads, -1).transpose(1, 2)
        q = attn.ln_q(q)
        k = attn.ln_k(k)
        qs[i], ks[i], vs[i] = q, k, v
        return output

    hooks = []
    for i, block in enumerate(model.transformer.resblocks):
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

def get_attribution_map(pre_softmax_attention, vs, sae_w, sae_b, layer, head):
    # This function remains unchanged
    cls_token_index = 0
    original_scores_head = pre_softmax_attention[layer][:, head]
    values_head = vs[layer][:, head]
    sae_neuron_weight = sae_w.T
    cls_scores = original_scores_head[:, cls_token_index, :]
    cls_attention_softmax = F.softmax(cls_scores, dim=-1)
    value_contributions = torch.matmul(values_head, sae_neuron_weight)

    avg_value_contribution = torch.sum(cls_attention_softmax.unsqueeze(-1) * value_contributions, dim=1, keepdim=True)

    analytical_gradient = cls_attention_softmax.unsqueeze(-1) * (value_contributions - avg_value_contribution)
    return cls_attention_softmax, analytical_gradient



def build_sae_list(model, layers, config, config_name, config_name_suffix, device, expand_ratio=16):
    sae_list = {}
    curr_config = config.copy()
    for layer in layers:
        curr_config['sae_layer_idx'] = layer
        curr_config['sae_k'] = int(config_name_suffix.split('topk')[-1])
        sae_ckpt_path = get_ckpt_path_typo(config, config_name, config_name_suffix, layer=layer, base_dir=f'typo_checkpoints_topk_{expand_ratio}')
        curr_config['sae_ckpt_path'] = sae_ckpt_path

        sae = TopKSAE.from_config(curr_config, model)
        sae.per_head_recon = False
        # sae.add_residual = False
        sae_list[layer] = sae.to(device)
    return sae_list

def create_masks(token_shape, text_depth=2):
    LOCATION_MAP = {0: "Top", 1: "Bottom", 2: "Left", 3: "Right"}
    d = torch.floor(torch.sqrt(torch.tensor(token_shape))).int()
    masks = []
    for i in range(4):
        mask = torch.zeros(d, d)
        if i == 0:  # Top
            mask[:text_depth, :] = 1
        elif i == 1:  # Bottom
            mask[-text_depth:, :] = 1
        elif i == 2:  # Left
            mask[:, :text_depth] = 1
        elif i == 3:  # Right
            mask[:, -text_depth:] = 1
        masks.append(mask.flatten())
    return torch.stack(masks)


def calcualte_score(text_loc, score_maps, all_masks):
    masks = all_masks[text_loc]  # (B, 196)
    score_maps = score_maps[:, 1:]  # remove cls token
    score_maps = torch.clamp(score_maps, min=0.0, max=1.0)
    score_maps = score_maps.permute(0, 2, 1)  # (B, C, 196)
    pos_scores = (score_maps * masks.unsqueeze(1)).sum(-1)  # (B, C)
    neg_scores = (score_maps * (1 - masks).unsqueeze(1)).sum(-1)  # (B, C)
    final_scores = pos_scores / (pos_scores + neg_scores + 1e-6)
    return final_scores  # (B, C)

def process_head_scores(head_scores, sae_head_dim, n_heads, model, dataset_val, model_name, config_name, config_suffix, layers, expand_ratio=16):
    for i in layers:
        read_path = get_save_path(f"./results/res{expand_ratio}", model_name, config_name, config_suffix, i)
        if not os.path.exists(read_path) or not os.path.exists(read_path + '/activation_to_top_activating_sample.pkl'):
            # print(f"Path {read_path} does not exist, skip layer {i}.")
            continue                
        activation_to_top_activating_sample, _ = get_top_activating_samples(model, dataset_val, read_path, False)  
        
        valid_neurons = [neuron for neuron in activation_to_top_activating_sample if len(activation_to_top_activating_sample[neuron]) >= 20 and activation_to_top_activating_sample[neuron][19][0] >= 0.5]
        valid_neurons = set(valid_neurons)
        for head in range(n_heads):

            head_indices = [neuron % sae_head_dim for neuron in valid_neurons if (neuron // (model.sae.hidden_dim // n_heads)) == head]

            if len(head_indices) == 0:
                head_scores[(i, head)] = 0.0
                continue
            head_scores[(i, head)] = head_scores[(i, head)][head_indices].mean().item() / len(dataset_val)
    return head_scores


imagenet_100_classes = [
    "cock",
    "tailed frog",
    "green snake",
    "barn spider",
    "bee eater",
    "snail",
    "limpkin",
    "hen",
    "loggerhead",
    "king snake",
    "garden spider",
    "hornbill",
    "sea slug",
    "American coot",
    "goldfinch",
    "leatherback turtle",
    "garter snake",
    "black widow",
    "hummingbird",
    "chiton",
    "bustard",
    "indigo bunting",
    "mud turtle",
    "vine snake",
    "tarantula",
    "toucan",
    "chambered nautilus",
    "red-backed sandpiper",
    "bulbul",
    "terrapin",
    "night snake",
    "wolf spider",
    "drake",
    "Dungeness crab",
    "redshank",
    "magpie",
    "banded gecko",
    "boa constrictor",
    "tick",
    "goose",
    "rock crab",
    "oystercatcher",
    "chickadee",
    "common iguana",
    "green mamba",
    "black grouse",
    "black swan",
    "spiny lobster",
    "pelican",
    "tench",
    "water ouzel",
    "whiptail",
    "sea snake",
    "ptarmigan",
    "wallaby",
    "crayfish",
    "albatross",
    "goldfish",
    "kite",
    "agama",
    "horned viper",
    "prairie chicken",
    "wombat",
    "hermit crab",
    "sea lion",
    "great white shark",
    "bald eagle",
    "green lizard",
    "diamondback",
    "peacock",
    "jellyfish",
    "white stork",
    "tiger shark",
    "great grey owl",
    "Komodo dragon",
    "sidewinder",
    "macaw",
    "sea anemone",
    "spoonbill",
    "hammerhead",
    "common newt",
    "American alligator",
    "harvestman",
    "sulphur-crested cockatoo",
    "flatworm",
    "flamingo",
    "electric ray",
    "spotted salamander",
    "thunder snake",
    "scorpion",
    "lorikeet",
    "nematode",
    "bittern",
    "stingray",
    "axolotl",
    "hognose snake",
    "black and gold garden spider",
    "coucal",
    "conch",
    "crane",
]
