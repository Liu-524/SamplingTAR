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

from experiments.attribution import clarity_score, get_top_activating_samples, get_attribution_map_helper
from experiments.attribution import (
    batch_centered_transform,
    get_attribution_map_helper,
)
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

    n_heads = model.transformer.resblocks[0].attn.num_heads
    config['sae_hidden_dim'] = args.sae_hidden_dim

    config['sae_n_heads'] = n_heads
    sae = TopKSAE.from_config(config, model)

    
    modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(sae_ckpt_path)))
    logging.info(f'Loaded SAE from {sae_ckpt_path}. Last modified: {modified_time}')
    model.add_sae(sae)
    model.to(device)
    model.eval()
    model.sae.layer_idx = config['layer']
    model.sae.per_head_recon = False
    return model


def find_checkpoint_dir(args):
    config_name_suffix = args.config_name_suffix
    if 'relu' in config_name_suffix:
        return f'checkpoints_relu_{args.sae_hidden_dim}'
    if 'topk' in config_name_suffix:
        return f'typo_checkpoints_topk_{args.sae_hidden_dim}'
    return 'checkpoints'



# --- Main Execution ---

def main(args):
    """Main function to run the analysis pipeline."""
    # 1. Construct all paths
    config_file_path = os.path.join(args.base_dir, args.model_name, args.config_filename)
    config_name_no_ext = os.path.splitext(args.config_filename)[0]
    experiment_name = f"{args.model_name}_{config_name_no_ext}-{args.layer}{args.config_name_suffix}"
    final_output_dir = os.path.join(args.output_dir, f'res{args.sae_hidden_dim}', experiment_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    sae_ckpt_dir = f'{find_checkpoint_dir(args)}/{args.model_name}_{config_name_no_ext}_layer{args.layer}{args.config_name_suffix}'

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
    log_file_path = os.path.join(final_output_dir, f'concept_{args.num_image_per_neuron}.log')
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
    
    dataset_class = get_dataset(config["dataset_name"])
    dataset_val = dataset_class(data_path=config["data_path"], normalize_data=True, split="test", **config.get("dataset_kwargs", {}))
    dataset_val.transform = model.preprocess
    
    with torch.no_grad() and torch.autocast(device_type=device.type, enabled=True):
        activation_to_top_activating_sample, activation_counts = get_top_activating_samples(
            model, dataset_val, final_output_dir, force_recompute=False
            # None, None, final_output_dir.replace('concept', 'expan'), force_recompute=False
        )
    


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
    parser.add_argument("--dedup_threshold", type=float, default=0.2, help="Similarity threshold for neuron deduplication.")
    args = parser.parse_args()
    main(args)

