import logging
import os
import pickle
import random
import sys
import time
from argparse import ArgumentParser
from functools import partial
from heapq import heappop, heappush
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch import nn
torch.set_float32_matmul_precision('high')

from dataset_utils import get_dataset
from model_training.sae import TopKSAE, ReLUSAE
from models import get_fn_model_loader
from proj_utils import resize_model, get_wanted_size
from proj_utils.eval_utils import get_top_activation_and_images, neuron_dedup

# --- Constants ---
HEAP_MAX_SIZE = 1000
STEP_SIZE = 1000

CLARITY_FEATURE_MODEL = 'clip_vit_h14_dfn5b'

from torch.utils.data import Dataset, DataLoader, BatchSampler
class ListBatchSampler(BatchSampler):
    def __init__(self, index_batches):
        self.index_batches = index_batches
    
    def __iter__(self):
        # Simply yield one batch (list of indices) at a time
        for batch in self.index_batches:
            yield batch
            
    def __len__(self):
        # This tells the DataLoader how many batches to expect
        return len(self.index_batches)

# --- Helper Functions ---

def setup_seeds(seed: int):
    """Set seeds for reproducibility."""
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def check_latest_ckpt(ckpt_root: str, only_last: bool = False) -> str | None:
    """Finds the most recently modified checkpoint file in a directory."""
    if not os.path.exists(ckpt_root):
        return None
    ckpts = [f for f in os.listdir(ckpt_root) if f.endswith('.ckpt')]
    if only_last:
        ckpts = [f for f in ckpts if 'last' in f]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_root, x)), reverse=True)
    return ckpts[0]


def get_top_activating_samples(model: torch.nn.Module, dataset_val, output_dir: str, force_recompute: bool = False):
    """
    Computes or loads the top activating samples for each SAE neuron.
    """

    GLOBAL_SAE_ACTIVATIONS = {}
    def sae_forward_hook(module, input, output):
        """A forward hook to capture SAE activations."""
        x_recon, z, z_bar = output
        GLOBAL_SAE_ACTIVATIONS['x_recon'] = x_recon.detach()
        GLOBAL_SAE_ACTIVATIONS['z'] = z.detach()
        GLOBAL_SAE_ACTIVATIONS['z_bar'] = z_bar.detach()
        return output
    if output_dir is not None:
        activation_samples_path = os.path.join(output_dir, 'activation_to_top_activating_sample.pkl')
        activation_counts_path = os.path.join(output_dir, 'activation_counts.pkl')

        if not force_recompute and os.path.exists(activation_samples_path): # TODO: FORCE RE-COMPUTE DISABLED FOR DEBUGGING
            logging.info(f'Loading cached activation data from {output_dir}')
            with open(activation_samples_path, 'rb') as f:
                activation_to_top_activating_sample = pickle.load(f)
            with open(activation_counts_path, 'rb') as f:
                activation_counts = pickle.load(f)
            return activation_to_top_activating_sample, activation_counts

    logging.info(f'No cached data found at {output_dir}. Computing from scratch...')
    device = next(model.parameters()).device
    sae_hidden_dim = model.sae.hidden_dim

    hook = model.sae.register_forward_hook(sae_forward_hook)
    
    original_token_type = model.sae.token_type
    if original_token_type == 'spatial':
        model.sae.token_type = 'cls'
    
    activation_counts = torch.zeros(sae_hidden_dim, dtype=torch.int32).to(device)
    activation_to_top_activating_sample = {idx: [] for idx in range(sae_hidden_dim)}
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    for i, (images, _) in enumerate(tqdm(dataloader_val, desc="Finding Top Activations")):
        with torch.no_grad() and torch.autocast(device_type=device.type, enabled=True):
            _ = model(images.to(device))
        
        z_bar = GLOBAL_SAE_ACTIVATIONS['z_bar']
        z_bar_cls = z_bar[:, 0, :].relu()

        for b in range(z_bar_cls.shape[0]):
            image_idx = i * dataloader_val.batch_size + b
            indices = z_bar_cls[b].nonzero(as_tuple=False)
            values = z_bar_cls[b, indices].squeeze(-1)
            activation_counts.index_add_(0, indices.flatten(), torch.ones_like(indices.flatten(), dtype=torch.int32))        
            for act_val, idx in zip(values, indices):
                heappush(activation_to_top_activating_sample[idx.item()], (act_val.item(), image_idx))
                if len(activation_to_top_activating_sample[idx.item()]) > HEAP_MAX_SIZE:
                    heappop(activation_to_top_activating_sample[idx.item()])
    
    hook.remove()
    model.sae.token_type = original_token_type
    
    activation_counts = activation_counts.cpu().numpy()
    activation_to_top_activating_sample = {k: sorted(v, reverse=True, key=lambda x: x[0]) for k, v in activation_to_top_activating_sample.items()}
    if output_dir is not None:
        with open(activation_samples_path, 'wb') as f:
            pickle.dump(activation_to_top_activating_sample, f)
        with open(activation_counts_path, 'wb') as f:
            pickle.dump(activation_counts, f)
        
    return activation_to_top_activating_sample, activation_counts

# --- Attribution and Clarity Score Calculation ---

@torch.inference_mode()
def clarity_score(V):
    """Calculates the clarity of neuron activations."""
    V_nrmed = F.normalize(V, dim=-1)
    clarity_val = ((V_nrmed.mean(-2).pow(2).sum((-1))) - 1 / V.shape[-2]) / (V.shape[-2] - 1) * V.shape[-2]
    return clarity_val

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
USE_IG = False

if not USE_IG:
    
    def get_attribution_map_basic(pre_softmax_attention, vs, sae_w, sae_b, layer, head, neuron_index):
        # This function remains unchanged
        B, n_heads, L, d_head = vs[layer].shape
        cls_token_index = 0
        if head == -1:
            sae_neuron_weight_full = sae_w[neuron_index]
            sae_neuron_weights_per_head = sae_neuron_weight_full.view(n_heads, d_head)
            scores_layer = pre_softmax_attention[layer]
            values_layer = vs[layer]
            cls_scores_all_heads = scores_layer[:, :, cls_token_index, :]
            cls_attention_softmax_all_heads = F.softmax(cls_scores_all_heads, dim=-1)
            value_contributions_all_heads = torch.einsum('bhld,hd->bhl', values_layer, sae_neuron_weights_per_head)
            avg_value_contribution_all_heads = torch.sum(cls_attention_softmax_all_heads * value_contributions_all_heads, dim=-1, keepdim=True)
            analytical_gradients_all_heads = cls_attention_softmax_all_heads * (value_contributions_all_heads - avg_value_contribution_all_heads)
            total_gradient = analytical_gradients_all_heads.sum(dim=1)
            avg_softmax = cls_attention_softmax_all_heads.mean(dim=1)
            return avg_softmax, total_gradient
        else:
            scores_head = pre_softmax_attention[layer][:, head]
            values_head = vs[layer][:, head]
            sae_neuron_weight_full = sae_w[neuron_index]
            start_idx, end_idx = head * d_head, (head + 1) * d_head
            sae_neuron_weight_slice = sae_neuron_weight_full[start_idx:end_idx]
            cls_scores = scores_head[:, cls_token_index, :]
            cls_attention_softmax = F.softmax(cls_scores, dim=-1)
            value_contributions = torch.matmul(values_head, sae_neuron_weight_slice)
            avg_value_contribution = torch.sum(cls_attention_softmax * value_contributions, dim=-1, keepdim=True)
            analytical_gradient = cls_attention_softmax * (value_contributions - avg_value_contribution)
            return cls_attention_softmax, analytical_gradient

    def get_attribution_map(pre_softmax_attention, vs, sae_w, sae_b, layer, head, neuron_index):
        # This function remains unchanged
        cls_token_index = 0
        original_scores_head = pre_softmax_attention[layer][:, head]
        values_head = vs[layer][:, head]
        sae_neuron_weight = sae_w[neuron_index % sae_w.shape[0]]
        cls_scores = original_scores_head[:, cls_token_index, :]
        cls_attention_softmax = F.softmax(cls_scores, dim=-1)
        value_contributions = torch.matmul(values_head, sae_neuron_weight)
        avg_value_contribution = torch.sum(cls_attention_softmax * value_contributions, dim=-1, keepdim=True)
        analytical_gradient = cls_attention_softmax * (value_contributions - avg_value_contribution)
        return cls_attention_softmax, analytical_gradient
else:
    def _calculate_analytical_gradient(cls_scores, values, sae_neuron_weights, head):
        """
        Internal helper to calculate the analytical gradient of neuron pre-activation
        w.r.t. the (potentially interpolated) cls_scores.
        
        This function is now used by both the standard and IG methods.
        
        Args:
            cls_scores (torch.Tensor): Pre-softmax attention scores.
                Shape [B, L] for single head or [B, H, L] for all heads.
            values (torch.Tensor): Value vectors.
                Shape [B, L, d_head] for single head or [B, H, L, d_head] for all heads.
            sae_neuron_weights (torch.Tensor): Relevant slice of the SAE neuron weight.
                Shape [d_head] for single head or [H, d_head] for all heads.
            head (int): The head index, or -1 for all heads.
        
        Returns:
            torch.Tensor: The analytical gradient w.r.t. cls_scores.
                Shape [B, L] or [B, H, L].
        """
        if head == -1:
            # scores: [B, H, L], values: [B, H, L, d_head], weights: [H, d_head]
            cls_attention_softmax_all_heads = F.softmax(cls_scores, dim=-1)
            
            # [B, H, L, d_head] @ [H, d_head] -> [B, H, L]
            value_contributions_all_heads = torch.einsum('bhld,hd->bhl', values, sae_neuron_weights)
            
            # [B, H, L] * [B, H, L] -> sum(dim=-1) -> [B, H, 1]
            avg_value_contribution_all_heads = torch.sum(cls_attention_softmax_all_heads * value_contributions_all_heads, dim=-1, keepdim=True)
            
            # [B, H, L] * ([B, H, L] - [B, H, 1])
            analytical_gradients_all_heads = cls_attention_softmax_all_heads * (value_contributions_all_heads - avg_value_contribution_all_heads)
            
            return analytical_gradients_all_heads # [B, H, L]
        else:
            # scores: [B, L], values: [B, L, d_head], weights: [d_head]
            cls_attention_softmax = F.softmax(cls_scores, dim=-1)
            
            # [B, L, d_head] @ [d_head] -> [B, L]
            value_contributions = torch.matmul(values, sae_neuron_weights)
            
            # [B, L] * [B, L] -> sum(dim=-1) -> [B, 1]
            avg_value_contribution = torch.sum(cls_attention_softmax * value_contributions, dim=-1, keepdim=True)
            
            # [B, L] * ([B, L] - [B, 1])
            analytical_gradient = cls_attention_softmax * (value_contributions - avg_value_contribution)
            
            return analytical_gradient # [B, L]
        
    def get_attribution_map(pre_softmax_attention, vs, sae_w, sae_b, layer, head, neuron_index, steps=50): # OVERLOAD
        """
        Calculates the Integrated Gradients attribution of the SAE neuron's
        pre-activation w.r.t. the pre-softmax attention scores.
        """
        B, n_heads, L, d_head = vs[layer].shape
        cls_token_index = 0
        
        sae_neuron_weight_full = sae_w[neuron_index % sae_w.shape[0]] # [d_model]
        # --- Specific Head ---
            # 1. Get constants and original input 'x'
        values_head = vs[layer][:, head] # [B, L, d_head]
        
        start_idx, end_idx = head * d_head, (head + 1) * d_head
        sae_neuron_weight_slice = sae_neuron_weight_full # [d_head]
        
        # This is 'x', the input we are attributing to.
        cls_scores = pre_softmax_attention[layer][:, head, cls_token_index, :].clone() # [B, L]

        # 2. Define baseline 'x''
        baseline_scores = torch.zeros_like(cls_scores)
        baseline_scores = cls_scores * 0.8 # Using 80% of original scores as baseline
        diff = cls_scores - baseline_scores
        
        # 3. Loop for integral approximation
        integrated_grads_sum = torch.zeros_like(cls_scores)

        for k in range(1, steps + 1):
            alpha = k / float(steps)
            # x(alpha) = x' + alpha * (x - x')
            interpolated_scores = (baseline_scores + alpha * diff)
            # NOTE: No .clone().requires_grad_() needed!

            # --- Calculate ∇F(x(α)) analytically ---
            analytical_grad_step = _calculate_analytical_gradient(
                interpolated_scores,
                values_head,
                sae_neuron_weight_slice,
                head
            )
            # NOTE: No .backward() needed!
            
            # 5. Accumulate the gradient
            integrated_grads_sum += analytical_grad_step
                
        # 6. Final IG calculation
        # IG = (x - x') * (1/m) * Σ ∇F(x(α))
        ig_attributions = (diff / steps) * integrated_grads_sum
        
        # Also return the original softmax for comparison
        cls_attention_softmax_original = F.softmax(cls_scores.detach(), dim=-1)
        
        return cls_attention_softmax_original, ig_attributions



    def get_attribution_map_basic(pre_softmax_attention, vs, sae_w, sae_b, layer, head, neuron_index, steps=50):
        """
        Calculates the Integrated Gradients attribution using the
        analytical gradient function in a loop (no autodiff).
        
        Handles two cases:
        1. head == -1: Attributes the total neuron activation across all heads.
        2. head != -1: Attributes the neuron activation from a single, specific head.
        """
        B, n_heads, L, d_head = vs[layer].shape
        cls_token_index = 0
        
        sae_neuron_weight_full = sae_w[neuron_index % sae_w.shape[0]] # [d_model]
        # We still need the bias for the 'all heads' case, but it's not used in the helper.
        # The bias gradient w.r.t. scores is 0, so it's implicitly correct.
        
        if head == -1:
            # --- All Heads ---
            # 1. Get constants and original input 'x'
            sae_neuron_weights_per_head = sae_neuron_weight_full.view(n_heads, d_head) # [n_heads, d_head]
            values_layer = vs[layer] # [B, n_heads, L, d_head]

            # This is 'x', the input we are attributing to.
            cls_scores_all_heads = pre_softmax_attention[layer][:, :, cls_token_index, :].clone() # [B, n_heads, L]
            
            # 2. Define baseline 'x''
            baseline_scores = torch.zeros_like(cls_scores_all_heads)
            baseline_scores = cls_scores_all_heads * 0.8 # Using 80% of original scores as baseline
            diff = cls_scores_all_heads - baseline_scores
            
            # 3. Loop for integral approximation
            integrated_grads_sum = torch.zeros_like(cls_scores_all_heads)
            
            for k in range(1, steps + 1):
                alpha = k / float(steps)
                # x(alpha) = x' + alpha * (x - x')
                interpolated_scores = (baseline_scores + alpha * diff)
                # NOTE: No .clone().requires_grad_() needed!
                
                # --- Calculate ∇F(x(α)) analytically ---
                analytical_grad_step = _calculate_analytical_gradient(
                    interpolated_scores,
                    values_layer,
                    sae_neuron_weights_per_head,
                    head
                )
                # NOTE: No .backward() needed!
                
                # 5. Accumulate the gradient
                integrated_grads_sum += analytical_grad_step
                
            # 6. Final IG calculation: IG = (x - x') * (1/m) * Σ ∇F(x(α))
            ig_attributions_all_heads = (diff / steps) * integrated_grads_sum # [B, n_heads, L]
            
            # Sum attributions over all heads to match analytical function's output shape
            total_ig_attributions = ig_attributions_all_heads.sum(dim=1) # [B, L]
            
            # Also return the original avg softmax for comparison
            avg_softmax = F.softmax(cls_scores_all_heads.detach(), dim=-1).mean(dim=1)
            
            return avg_softmax, total_ig_attributions
        else:
            # --- Specific Head ---
            # 1. Get constants and original input 'x'
            values_head = vs[layer][:, head] # [B, L, d_head]
            
            start_idx, end_idx = head * d_head, (head + 1) * d_head
            sae_neuron_weight_slice = sae_neuron_weight_full[start_idx:end_idx] # [d_head]
            
            # This is 'x', the input we are attributing to.
            cls_scores = pre_softmax_attention[layer][:, head, cls_token_index, :].clone() # [B, L]

            # 2. Define baseline 'x''
            baseline_scores = torch.zeros_like(cls_scores)
            baseline_scores = cls_scores * 0.8 # Using 80% of original scores as baseline
            diff = cls_scores - baseline_scores
            
            # 3. Loop for integral approximation
            integrated_grads_sum = torch.zeros_like(cls_scores)

            for k in range(1, steps + 1):
                alpha = k / float(steps)
                # x(alpha) = x' + alpha * (x - x')
                interpolated_scores = (baseline_scores + alpha * diff)
                # NOTE: No .clone().requires_grad_() needed!

                # --- Calculate ∇F(x(α)) analytically ---
                analytical_grad_step = _calculate_analytical_gradient(
                    interpolated_scores,
                    values_head,
                    sae_neuron_weight_slice,
                    head
                )
                # NOTE: No .backward() needed!
                
                # 5. Accumulate the gradient
                integrated_grads_sum += analytical_grad_step
                    
            # 6. Final IG calculation
            ig_attributions = (diff / steps) * integrated_grads_sum # [B, L]
            
            # Also return the original softmax for comparison
            cls_attention_softmax_original = F.softmax(cls_scores.detach(), dim=-1)
            
            return cls_attention_softmax_original, ig_attributions
        

def get_attribution_map_helper(model, images, head, neuron_index, layer, return_attention_scores=False):
    # This function remains unchanged
    device = next(model.parameters()).device
    images = images.to(device)
    xs, qs, ks, vs, attention_scores = get_attention_scores(model, images, layer)
    if model.sae.use_basic or model.sae.siamese_encoder:
        sae_w = model.sae.encoders[0][0].weight.data
        sae_b = model.sae.encoders[0][0].bias.data
        amap, grad = get_attribution_map_basic(attention_scores, vs, sae_w, sae_b, layer, head, neuron_index)
    else:
        sae_w = model.sae.encoders[head][0].weight.data
        sae_b = model.sae.encoders[head][0].bias.data
        amap, grad = get_attribution_map(attention_scores, vs, sae_w, sae_b, layer, head, neuron_index)
    if return_attention_scores:
        return images, amap, grad, attention_scores
    return images, amap, grad


def batch_centered_transform(x: torch.Tensor, vcenter: float = 0.0, eps: float = 1e-8):
    """
    Applies CenteredNorm to a batch of tensors.
    Assumes dim 0 is the batch dimension.
    
    - If x is (B, ...): Normalizes each sample b independently using its own range.
    - If x is (B,): Normalizes the entire 1D vector as a single group (global norm).
    """

    centered = x - vcenter
    
    # Case 1: 1D Tensor (Batch of scalars treated as a single distribution)
    if x.ndim == 1:
        halfrange = centered.abs().max()
        
    # Case 2: Multi-dim Tensor (Batch of vectors/images)
    else:
        # Flatten all dims except batch (dim 0) to find per-sample max
        # x.size(0) is B. -1 flattens the rest.
        flat_centered = centered.view(x.size(0), -1)
        
        # Calculate max per sample
        halfrange = flat_centered.abs().max(dim=1).values
        
        # Reshape halfrange to (B, 1, 1, ...) to allow broadcasting back to x
        # We append 1s for every dimension in x after the first
        view_shape = [x.size(0)] + [1] * (x.ndim - 1)
        halfrange = halfrange.view(*view_shape)

    # Apply formula: result = 0.5 + (centered / 2 * halfrange)
    normalized = centered / (2.0 * halfrange + eps)
    
    return normalized


def get_batch_gradient_scores_from_images(model, images, head, neuron_index, layer, do_center=True, ignore_negative=True):
    # This function remains unchanged
    device = next(model.parameters()).device
    all_amaps, all_gradient_scores = [], []

    with torch.inference_mode() and torch.autocast(device_type=device.type, enabled=True):
        _, all_amaps, all_gradient_scores, ascores = \
             get_attribution_map_helper(model, images, head, neuron_index, layer, return_attention_scores=True)
    if ignore_negative:
        all_gradient_scores = all_gradient_scores.clamp(min=0.0)
    if do_center:
        all_gradient_scores = batch_centered_transform(all_gradient_scores, vcenter=0.0)

    return all_amaps, all_gradient_scores

def get_batch_gradient_scores_list(model, images, neuron_indices, layer, do_ixg=False):

    # always uncentered

    device = next(model.parameters()).device
    images = images.to(device)
    xs, qs, ks, vs, attention_scores = get_attention_scores(model, images, layer)
    all_amaps = []
    all_gradient_scores = []
    with torch.inference_mode() and torch.autocast(device_type=device.type, enabled=True):
        for i, neuron_index in enumerate(neuron_indices):
            head = -1 if model.sae.use_basic else neuron_index // (model.sae.hidden_dim // model.sae.n_heads) 
            if model.sae.use_basic or model.sae.siamese_encoder:
                sae_w = model.sae.encoders[0][0].weight.data
                sae_b = model.sae.encoders[0][0].bias.data
                amap, grad = get_attribution_map_basic(attention_scores, vs, sae_w, sae_b, layer, head, neuron_index)
            else:
                sae_w = model.sae.encoders[head][0].weight.data
                sae_b = model.sae.encoders[head][0].bias.data
                amap, grad = get_attribution_map(attention_scores, vs, sae_w, sae_b, layer, head, neuron_index)
            if do_ixg:
                grad = grad * attention_scores[layer][i, head, 0, :]
            all_amaps.append(amap)
            all_gradient_scores.append(grad)
    return all_amaps, all_gradient_scores

def extract_clip_patch_features(dataset, model, output_path, batch_size=32):
    # This function remains unchanged
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    sample_image = next(iter(dataloader))[0].to(device)
    with torch.inference_mode():
        last_hidden_state = model.visual(sample_image)
        num_patches, embedding_dim = last_hidden_state.shape[1], last_hidden_state.shape[2]
    logging.info(f"Detected {num_patches} patches with an embedding dimension of {embedding_dim}.")
    shape = (len(dataset), num_patches, embedding_dim)
    patch_features_mmap = np.zeros(shape, dtype=np.float16)
    for i, (images, _) in enumerate(tqdm(dataloader, desc="Extracting Patch Features")):
        with torch.inference_mode() and torch.autocast(device_type=device.type, enabled=True):
            patch_tokens_full = model.visual(images.to(device))
            patch_tokens = patch_tokens_full / patch_tokens_full.norm(dim=-1, keepdim=True)
            start_idx, end_idx = i * batch_size, i * batch_size + images.shape[0]
            patch_features_mmap[start_idx:end_idx] = patch_tokens.detach().cpu().numpy()
    # np.save(output_path, patch_features_mmap) ## TODO: DISABLE SAVING FOR DEBUGGING ##
    logging.info("Patch feature extraction complete.")
    return patch_features_mmap

def make_threshold_plot(scores, output_path: str, do_patch: bool):
    plt.figure(figsize=(8,6))
    threshold = torch.arange(0, 1.01, 0.01)
    cs = []
    for t in threshold:
        cs.append((scores > t).sum().item())
    
    plt.plot(threshold, cs, label=labels[i], marker='o' if 'basic' in labels[i] else 's')
    plt.xlabel('Clarity Score Threshold')
    plt.ylabel('Number of Neurons above Threshold')
    title_text = ("CLS Clarity Scores" if not do_patch else "Patch Clarity Scores")
    plt.title(title_text)
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def load_model_and_sae(config: dict, sae_ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Loads the base model and attaches the trained SAE from a specific checkpoint path."""
    def update_config(config: dict) -> dict:

        config['sae_hidden_dim'] = config.get('sae_hidden_dim')
        config['sae_ckpt_path'] = sae_ckpt_path

        if 'relu' in config.get('config_name_suffix', ''):
            config['sae_k'] = float(config.get('config_name_suffix').split('relu')[-1])
        elif 'topk' in config.get('config_name_suffix', ''):
            config['sae_k'] = int(config.get('config_name_suffix').split('topk')[-1])
        return config
    
    if 'dino' in config["model_name"]:
        model = get_fn_model_loader(config["model_name"])(load_as_clip=True)
    else:
        model = get_fn_model_loader(config["model_name"])()
    config = update_config(config)
    if 'relu' in config.get('config_name_suffix', ''):
        sae = ReLUSAE.from_config(config=config, model=model)
    else:
        sae = TopKSAE.from_config(config=config, model=model)
    
    modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(sae_ckpt_path)))
    logging.info(f'Loaded SAE from {sae_ckpt_path}. Last modified: {modified_time}')
    model.add_sae(sae)
    model.to(device)
    model.eval()
    model.sae.layer_idx = config['layer']
    return model


def find_checkpoint_dir(args):
    config_name_suffix = args.config_name_suffix
    if 'relu' in config_name_suffix:
        return f'checkpoints_relu_{args.sae_hidden_dim}'
    if 'topk' in config_name_suffix:
        return f'checkpoints_topk_{args.sae_hidden_dim}'
    return 'checkpoints'



def get_images_from_class(dataset, class_idx):
    image_indices = [i for i in range(len(dataset)) if dataset.imgs[i][1] == class_idx]
    return image_indices


# --- Main Execution ---

def main(args):
    """Main function to run the analysis pipeline."""
    # 1. Construct all paths
    config_file_path = os.path.join(args.base_dir, args.model_name, args.config_filename)
    config_name_no_ext = os.path.splitext(args.config_filename)[0]
    experiment_name = f"{args.model_name}_{config_name_no_ext}-{args.layer}{args.config_name_suffix}"
    final_output_dir = os.path.join(args.output_dir,("ixg_" if args.use_ixg else "") + f'text_imnet{args.sae_hidden_dim}', experiment_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    sae_ckpt_dir = f'{find_checkpoint_dir(args)}/{args.model_name}_{config_name_no_ext}{args.config_name_suffix}-{args.layer}'

    if args.ckpt_name:
        sae_ckpt_path = os.path.join(sae_ckpt_dir, args.ckpt_name)
    else:
        latest_ckpt = check_latest_ckpt(sae_ckpt_dir, only_last=True)
        if latest_ckpt is None:
            # No logging is set up yet, so use print for this critical error.
            print(f"FATAL: No checkpoint found in {sae_ckpt_dir}. Please specify with --ckpt_name.")
            return
        sae_ckpt_path = os.path.join(sae_ckpt_dir, latest_ckpt)

    # 2. Handle the dry run flag (prints to console only)
    if args.dry_run:
        print("--- Dry Run Mode ---")
        print("The following paths will be used for the experiment:")
        print(f"\n[INPUTS]\n  - Config file: {config_file_path}\n  - SAE Checkpoint: {sae_ckpt_path}")
        print(f"\n[OUTPUTS]\n  - Experiment Directory: {final_output_dir}")
        print("\n--- End of Dry Run ---")
        return

    # 3. Set up logging for a real run
    if args.dedup_threshold == 1:
        log_file_path = os.path.join(final_output_dir, 'attribution.log')
        assert not os.path.exists(log_file_path), f"Log file {log_file_path} already exists. To avoid overwriting, please specify a different dedup_threshold."
    else:
        log_file_path = os.path.join(final_output_dir, f'attribution_{args.dedup_threshold}.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='w'),
                            logging.StreamHandler(sys.stdout)
                        ])

    # 4. Proceed with normal execution
    logging.info(f"Loading configuration from: {config_file_path}")
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    config.update(vars(args))
    
    logging.info(f"All outputs for this run will be saved to: {final_output_dir}")
    setup_seeds(config.get("random_seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Running on device: {device}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

    model = load_model_and_sae(config, sae_ckpt_path, device)

    clip_model = get_fn_model_loader(config['model_name'])(return_clip_model=True).eval().to(device)

    
    dataset_class = get_dataset(config["dataset_name"])
    dataset_val = dataset_class(data_path=config["data_path"], normalize_data=True, split="test", **config.get("dataset_kwargs", {}))
    dataset_val.transform = model.preprocess

    sae_hidden_dim = model.sae.hidden_dim
    n_heads = model.transformer.resblocks[0].attn.num_heads

    
    if os.path.exists("/localtmp/clip_patch_features.npy"):
        logging.info("Using local cached CLIP patch features...")
        features_path = "/localtmp/clip_patch_features.npy"
        patch_features = np.load(features_path).astype(np.float16)
        clip_config = get_fn_model_loader(CLARITY_FEATURE_MODEL)(config_only=True)
        clip_model_patch_size = clip_config['vision_cfg']['patch_size']
        clip_image_size = clip_config['vision_cfg']['image_size']
    else:
        # global_path = os.path.join("/localtmp/headsae/", CLARITY_FEATURE_MODEL)
        global_path = os.path.join(os.path.dirname(final_output_dir), 'global', CLARITY_FEATURE_MODEL)
        os.makedirs(global_path, exist_ok=True)
        logging.info("\n--- Loading/Extracting CLIP Patch Features ---")
        features_path = os.path.join(global_path, "clip_patch_features.npy")
        if not os.path.exists(features_path): ## TODO: TO FORCE RE-EXTRACTION ##        
            clip_model = get_fn_model_loader(CLARITY_FEATURE_MODEL)(return_clip_model=True).eval()
            clip_model_patch_size = clip_model.visual.patch_size[0]
            clip_image_size = clip_model.preprocess.transforms[0].size
            clip_model.visual.pool_type = 'none'
            clip_model.to(device)
            dataset_for_clip = dataset_class(data_path=config["data_path"], normalize_data=True, split="test", **config.get("dataset_kwargs", {}))
            dataset_for_clip.transform = clip_model.preprocess
            patch_features = extract_clip_patch_features(dataset_for_clip, clip_model, features_path)
            np.save(features_path, patch_features)
            del clip_model
        else:
            clip_config = get_fn_model_loader(CLARITY_FEATURE_MODEL)(config_only=True)
            clip_model_patch_size = clip_config['vision_cfg']['patch_size']
            clip_image_size = clip_config['vision_cfg']['image_size']
            patch_features = np.load(features_path)
    patch_features = torch.tensor(patch_features, dtype=torch.float16, device=device)
    patch_features -= patch_features.mean(dim=(0,1), keepdim=True)
   

    def get_inverse_attn_mask_from_attribution(scores, threshold=1):
        # scores: (B, H, W)

        flat = scores.reshape(scores.shape[0], -1)
        flat = (flat - flat.mean(dim=-1, keepdim=True)) / (flat.std(dim=-1, keepdim=True) + 1e-8)
        mask = (flat > threshold).float()
        mask[:, 0] = 0.0 # never use CLS token
        zero_masks = mask.sum(dim=-1) == 0
        if (zero_masks).any():
            mask[zero_masks] = (flat[zero_masks] > 0.0).float()
        return mask.reshape(scores.shape)


    #TODO: quick fix for dino_vitb8
    patch_size = model.patch_size if isinstance(model.patch_size, int) else model.patch_size[0]
    image_size = clip_image_size
    model = resize_model(model, new_img_size=get_wanted_size(image_size, clip_model_patch_size, patch_size))
    
    logging.info(f"MODEL PREPROCESS: {model.preprocess.transforms}")
    dataset_val.transform = model.preprocess
    
    image_indices_list = [get_images_from_class(dataset_val, class_idx) for class_idx in range(1000)]
    sampler = ListBatchSampler(image_indices_list)

    dataloader = DataLoader(dataset_val, batch_sampler=sampler, pin_memory=True, num_workers=4, persistent_workers=True)

    def neuron_filter(model, neurons):
        if model.sae.use_basic:
            return neurons[:1]
        neuron_head = neurons // (model.sae.hidden_dim // n_heads)
        selected_neurons = []
        for h in range(n_heads):
            if (neuron_head == h).any():
                selected_neurons.append(neurons[neuron_head == h][0])
        return np.array(selected_neurons)
    
    @torch.inference_mode()
    def neuron_select(model, sae, cls, to_select=12):
        device=next(model.parameters()).device
        class_repr = model.encode_text(model.tokenizer([cls]).to(device))
        dec = sae.decoder.weight.to(device)
        neuron_repr = dec.T @ model.visual.proj
        
        similarities = F.cosine_similarity(class_repr.unsqueeze(1), neuron_repr.unsqueeze(0), dim=-1).detach()
        values, indices = similarities.topk(to_select, dim=-1)
        return indices[0].cpu().numpy(), values[0].cpu().numpy()
    
    cs = []
    neuron_selects = []
    for (batch, image_indices) in tqdm(zip(dataloader, image_indices_list), total=len(dataloader), desc="Calculating Class Clarity Scores"):
        images, labels = batch
        label_text = dataset_val.classes[labels[0].item()]
        indices, values = neuron_select(clip_model, model.sae, label_text, to_select=max(1, int(n_heads * args.dedup_threshold)))
        
        neurons = neuron_filter(model, indices)
        neuron_selects.append(neurons)
        # for neuron in neurons:
        #     head = neuron // (model.sae.hidden_dim // n_heads)
        #     if model.sae.use_basic:
        #         head = -1
        
        #     amap, gradient_scores = get_batch_gradient_scores_from_images(model, images, head, neuron, layer=args.layer, do_center=False)

        #     all_gradient_scores.append(gradient_scores)
        if args.use_ixg:
            all_amaps, all_gradient_scores = get_batch_gradient_scores_list(model, images, neurons, args.layer, do_ixg=True)
        else:
            all_amaps, all_gradient_scores = get_batch_gradient_scores_list(model, images, neurons, args.layer)

        gradient_scores = torch.stack(all_gradient_scores, dim=0).mean(dim=0)
        gradient_scores = gradient_scores.clamp(min=0.0)  ## TODO: check if clamping is needed
        gradient_scores = batch_centered_transform(gradient_scores, vcenter=0)
        masks = get_inverse_attn_mask_from_attribution(gradient_scores, threshold=0.5)
        image_features = patch_features[image_indices] * masks.unsqueeze(-1).to(device)
        class_feature = image_features.mean(dim=1)
        class_score = clarity_score(class_feature)
        # print(f'Class {labels[0].item()} Clarity Score: {class_score.item():.4f}')
        # print("Top 6 neurons:", indices)
        # print("Top 6 values:", values)
        cs.append((labels[0].item(), class_score.item(), indices, values))

    
    logging.info("Clarity score calculation complete.")
    logging.info(f'average clarity score: {np.mean([c[1] for c in cs]):.4f}')
    # Save results
    if args.dedup_threshold == 1:
        results_path = os.path.join(final_output_dir, "imnet_attribution.pkl")
    else:
        results_path = os.path.join(final_output_dir, f"imnet_attribution_{args.dedup_threshold}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({'scores': cs, 'neuron_selects': neuron_selects}, f)
    logging.info(f"Clarity scores saved to {results_path}")
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze a trained Sparse Autoencoder on a Vision Transformer.")
    parser.add_argument("--base_dir", type=str, default="configs/train_sae/imagenet", help="Base directory where model configuration files are stored.")
    parser.add_argument("--output_dir", type=str, default="/bigtemp2/qzp4ta/headsae", help="Base directory to store experiment outputs.")
    parser.add_argument("--config_filename", type=str, required=True, help="Name of the YAML configuration file.")
    parser.add_argument("--model_name", type=str, default='clip_vit_b16_datacomp_xl_s13b_b90k', help="Name of the model.")
    parser.add_argument("--sae_hidden_dim", type=int, required=True, help="Hidden dimension size of the SAE.")
    parser.add_argument("--layer", type=int, default=11, help="The transformer layer where the SAE is attached.")
    parser.add_argument("--config_name_suffix", type=str, default="", help="Suffix for the config name to locate checkpoints.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recomputation of cached data.")
    parser.add_argument("--dry_run", action='store_true', help="Perform a dry run, printing paths without executing.")
    parser.add_argument("--ckpt_name", type=str, default=None, help="Specific checkpoint filename to load. Defaults to 'last.ckpt'.")
    parser.add_argument("--num_neurons", type=int, default=2000, help="Number of neurons to analyze for clarity scores.")
    parser.add_argument("--num_image_per_neuron", type=int, default=20, help="Number of top activating images per neuron to use.")
    parser.add_argument("--dedup_threshold", type=float, default=0., help="OVERLOADED: as num_head multiplier for number of neurons to select per class.")
    parser.add_argument("--use_ixg", action='store_true', help="Use Input X Gradient for attribution instead of analytical gradients.")
    args = parser.parse_args()
    main(args)

