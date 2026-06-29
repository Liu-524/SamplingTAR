"""Unified CLIP typographic-attack classification entry point.

Merges the former ``typo_a.py`` (CV threshold-optimisation) and
``typo_a_zsweep.py`` (z-sigma sweep) into one script; the strategy is selected
with ``--mode`` instead of living in the filename:

    python clip_eval.py --model_name <clip> --config_fn <cfg>.yaml --mode cv     --gpu 2,3
    python clip_eval.py --model_name <clip> --config_fn <cfg>.yaml --mode zsweep --gpu 2,3

  --mode cv      cross-validated z-threshold selection on --cv_dataset, then a
                 single evaluation with the best spec (writes
                 {model}_{cv_dataset}_attribution.json)
  --mode zsweep  evaluate every z-threshold in CANDIDATES, one block per sigma,
                 with wall-clock timing (writes {model}_zsweep.json)
  --gpu  CUDA device ids, e.g. "2,3" -> CUDA_VISIBLE_DEVICES (set before torch)

Model/SAE loading + circuit mining are shared (see clip_common). The two modes
preserve their original accuracy accumulation exactly (CV: mean*N, sweep: sum).
"""

import argparse
import os

# Candidate z-thresholds differ per mode, matching the original scripts.
CV_CANDIDATES = [0.5, 1.0, 2.0]
ZSWEEP_CANDIDATES = [0.0, 0.5, 1.0, 2.0]


def __getattr__(name):
    """Lazily re-export the CLIP ``build_random_sae_list`` for eccv_supp.ipynb,
    which does ``from clip_eval import build_random_sae_list``. Kept lazy (PEP 562)
    so importing this module for the CLI does not pull torch before --gpu sets
    CUDA_VISIBLE_DEVICES."""
    if name == "build_random_sae_list":
        from clip_common import build_random_sae_list
        return build_random_sae_list
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _preparse_gpu_mode():
    """Parse only --gpu/--mode so CUDA_VISIBLE_DEVICES is set before torch import."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", default=None)
    pre.add_argument("--mode", default="cv", choices=["cv", "zsweep"])
    args, _ = pre.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args.mode


# --- CV mode helpers (faithful to original typo_a.py) ----------------------

def evaluate_fold(model, ds, class_embeddings, indices, layer_spec, args):
    """Inference on the subset 'indices' with 'layer_spec' interventions (CV fold)."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from clip_common import DEVICE
    from eval_utils import fix_attn_head_list

    subset = torch.utils.data.Subset(ds, indices)
    dl = DataLoader(subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

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
                acc = (preds.cpu() == torch.tensor(labels)).float().mean().item()
                attack_acc = (preds.cpu() == torch.tensor(attack_labels)).float().mean().item()
                cum_acc += acc * images.shape[0]
                cum_attack_acc += attack_acc * images.shape[0]
                total_samples += images.shape[0]
    print(f"   Total Samples Evaluated: {total_samples}")
    print(f"   Object Accuracy: {cum_acc / total_samples:.4f}")
    print(f"   Attack Word Accuracy: {cum_attack_acc / total_samples:.4f}")
    if total_samples > 0:
        return cum_acc / total_samples
    return 0.0


def optimize_threshold(model, text_model, dataset, head_scores, layers, n_heads, device, args):
    """Pick the z-threshold (sigma) that maximises object accuracy on the CV set."""
    import numpy as np
    from clip_common import encode_class_labels

    print(f"Starting CV on {len(dataset)} samples with cached scores...")
    class_embeddings = encode_class_labels(text_model, list(dataset.class_names), skip_none=True)
    best_sigma = None
    best_acc = 0.0
    best_spec = None
    indices = np.arange(len(dataset))

    for sigma in CV_CANDIDATES:
        layer_spec = {}
        for layer in layers:
            layer_spec[layer] = []
            layer_scores = np.array([head_scores[(layer, head)] for head in range(n_heads)])
            layer_mean, layer_std = layer_scores.mean(), layer_scores.std()
            for head in range(n_heads):
                score = head_scores[(layer, head)]
                if score > layer_mean + layer_std * sigma:
                    layer_spec[layer].append(head)

        val_acc = evaluate_fold(model, dataset, class_embeddings, indices, layer_spec, args)
        print(f"   Sigma {sigma} ::: {layer_spec} -> Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_sigma = sigma
            best_acc = val_acc
            best_spec = layer_spec
    return best_sigma, best_spec, []


def run_cv(model, head_scores, output_dir, layers, n_heads, preprocess, model_name, args):
    import json
    from functools import partial
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from clip_common import (
        DEVICE, load_text_encoder, encode_class_labels,
        evaluate_attack_dataset, evaluate_imagenet,
    )
    from eval_utils import CustomImageDataset, collate_fn

    dataset_cv = CustomImageDataset(img_dir=f'./data/{args.cv_dataset}', transform=preprocess)
    print(f"Optimizing threshold on CV dataset: {args.cv_dataset} with {len(dataset_cv)} samples")
    print("Loading Text Encoder")
    text_model = load_text_encoder(model_name)
    best_sigma, layer_spec, _ = optimize_threshold(
        model, text_model, dataset_cv, head_scores, layers, n_heads, DEVICE, args)
    print(f"Selected heads for ablation based on attribution scores: {layer_spec}")

    image_dirs = ['./data/rta100', './data/disentangle', './data/paintds', './data/imagenet_100_text']
    all_results = {'sigma': best_sigma, 'layer_spec': layer_spec}

    print("Starting Evaluation Loop...")
    for image_dir in image_dirs:
        if not os.path.exists(image_dir):
            print(f"Skipping {image_dir}, path not found.")
            continue
        ds = CustomImageDataset(img_dir=image_dir)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=partial(collate_fn, preprocess_fn=preprocess))
        class_embeddings = encode_class_labels(text_model, list(ds.class_names), skip_none=True)
        obj_acc, att_acc, total_samples = evaluate_attack_dataset(
            model, ds, dl, class_embeddings, layer_spec, args, accum="mean")
        print(f"Results for dataset: {ds.ds_name}")
        print(f"  Object Accuracy: {obj_acc:.4f}")
        print(f"  Attack Word Accuracy: {att_acc:.4f}")
        all_results[ds.ds_name] = {
            "object_accuracy": obj_acc,
            "attack_word_accuracy": att_acc,
            "total_samples": total_samples,
        }

    imnet_100 = ImageFolder(root='./data/kaggle_imnet100/val', transform=preprocess)
    imnet_dl = DataLoader(imnet_100, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    id_to_class = json.load(open('./data/kaggle_imnet100/Labels.json', 'r'))
    imnet_labels = [id_to_class[x].split(",")[0] for x in imnet_100.classes]
    imnet_emb = encode_class_labels(text_model, imnet_labels, skip_none=False)

    acc, total_samples = evaluate_imagenet(model, imnet_dl, imnet_emb, args, accum="mean", layer_spec=layer_spec)
    print(f"Clarity SAE Accuracy on ImageNet: {acc:.4f}")
    all_results['imagenet'] = {"object_accuracy": acc, "total_samples": total_samples}

    acc, total_samples = evaluate_imagenet(model, imnet_dl, imnet_emb, args, accum="mean", layer_spec=None)
    print(f"Clarity SAE Accuracy on ImageNet: {acc:.4f}")
    all_results['imagenet_full_model'] = {"object_accuracy": acc, "total_samples": total_samples}

    output_path = os.path.join(output_dir, f"{model_name}_{args.cv_dataset}_attribution.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"All results saved to {output_path}")


def run_zsweep(model, head_scores, output_dir, layers, n_heads, preprocess, model_name, args,
               timings, wall_start):
    import json
    import time
    from functools import partial
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from clip_common import (
        load_text_encoder, encode_class_labels, evaluate_attack_dataset,
        evaluate_imagenet, build_layer_spec, fmt_dt,
    )
    from eval_utils import CustomImageDataset, collate_fn

    print("Loading Text Encoder")
    text_model = load_text_encoder(model_name)

    image_dirs = ['./data/rta100', './data/disentangle', './data/paintds', './data/imagenet_100_text']
    eval_datasets = []
    for image_dir in image_dirs:
        if not os.path.exists(image_dir):
            print(f"Skipping {image_dir}, path not found.")
            continue
        ds = CustomImageDataset(img_dir=image_dir)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=partial(collate_fn, preprocess_fn=preprocess))
        ds_class_emb = encode_class_labels(text_model, list(ds.class_names), skip_none=True)
        eval_datasets.append((ds, dl, ds_class_emb))

    imnet_100 = ImageFolder(root='./data/kaggle_imnet100/val', transform=preprocess)
    imnet_dl = DataLoader(imnet_100, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    id_to_class = json.load(open('./data/kaggle_imnet100/Labels.json', 'r'))
    imnet_labels = [id_to_class[x].split(",")[0] for x in imnet_100.classes]
    imnet_class_emb = encode_class_labels(text_model, imnet_labels, skip_none=False)

    imnet_full_model_acc, total_samples = evaluate_imagenet(
        model, imnet_dl, imnet_class_emb, args, accum="sum", layer_spec=None)
    print(f"ImageNet (no intervention) Accuracy: {imnet_full_model_acc:.4f}")

    sweep_results = {
        "candidates": ZSWEEP_CANDIDATES,
        "imagenet_full_model": {"object_accuracy": imnet_full_model_acc, "total_samples": total_samples},
        "by_sigma": {},
    }
    timings["per_sigma_sec"] = {}
    sweep_start = time.time()

    for sigma in ZSWEEP_CANDIDATES:
        sigma_start = time.time()
        layer_spec = build_layer_spec(head_scores, layers, n_heads, sigma)
        print(f"\n===== Sigma {sigma} =====")
        print(f"layer_spec: {layer_spec}")
        sigma_results = {"layer_spec": layer_spec}

        for ds, dl, ds_class_emb in eval_datasets:
            obj_acc, att_acc, total_samples = evaluate_attack_dataset(
                model, ds, dl, ds_class_emb, layer_spec, args, accum="sum")
            print(f"  {ds.ds_name}: obj={obj_acc:.4f} attack={att_acc:.4f}")
            sigma_results[ds.ds_name] = {
                "object_accuracy": obj_acc,
                "attack_word_accuracy": att_acc,
                "total_samples": total_samples,
            }

        imnet_acc, total_samples = evaluate_imagenet(
            model, imnet_dl, imnet_class_emb, args, accum="sum", layer_spec=layer_spec)
        print(f"  ImageNet (σ={sigma}): {imnet_acc:.4f}")
        sigma_results["imagenet"] = {"object_accuracy": imnet_acc, "total_samples": total_samples}

        sigma_elapsed = time.time() - sigma_start
        timings["per_sigma_sec"][str(sigma)] = sigma_elapsed
        sigma_results["wall_clock_sec"] = sigma_elapsed
        print(f"[time] σ={sigma}: {fmt_dt(sigma_elapsed)} ({sigma_elapsed:.1f}s)")
        sweep_results["by_sigma"][str(sigma)] = sigma_results

    timings["sweep_total_sec"] = time.time() - sweep_start
    timings["total_sec"] = time.time() - wall_start
    timings["end_iso"] = time.strftime('%Y-%m-%d %H:%M:%S')
    sweep_results["timings"] = timings

    print("\n[time] ===== Wall-clock summary =====")
    if "circuit_mining_sec" in timings:
        print(f"[time] circuit mining: {fmt_dt(timings['circuit_mining_sec'])}")
    for s, t in timings["per_sigma_sec"].items():
        print(f"[time] σ={s}: {fmt_dt(t)}")
    print(f"[time] sweep total: {fmt_dt(timings['sweep_total_sec'])}")
    print(f"[time] grand total: {fmt_dt(timings['total_sec'])}  ({timings['start_iso']} → {timings['end_iso']})")

    output_path = os.path.join(output_dir, f"{model_name}_zsweep.json")
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=4)
    print(f"All sweep results saved to {output_path}")


def main():
    mode = _preparse_gpu_mode()

    import time
    from pytorch_lightning import seed_everything
    from dataset_utils import get_dataset
    from eval_utils import get_eval_parser, prepare_torch
    from clip_common import load_clip_and_sae, mine_head_scores, fmt_dt

    parser = get_eval_parser()
    parser.add_argument("--gpu", default=None, help='CUDA device ids -> CUDA_VISIBLE_DEVICES')
    parser.add_argument("--mode", default="cv", choices=["cv", "zsweep"])
    args = parser.parse_args()
    print("Arguments:", args)

    wall_start = time.time()
    timings = {"start_iso": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(wall_start))}
    if mode == "zsweep":
        print(f"[time] start: {timings['start_iso']}")

    seed = args.seed
    seed_everything(seed)
    prepare_torch()

    bundle = load_clip_and_sae(args.model_name, args.config_fn, args)
    model = bundle["model"]

    dataset_cls = get_dataset(bundle["config"]["dataset_name"])
    dataset_val = dataset_cls(
        data_path=bundle["config"]["data_path"], normalize_data=True, split="train",
        **bundle["config"].get("dataset_kwargs", {}))

    mining_start = time.time()
    if mode == "zsweep":
        print("[time] circuit mining: start")
    head_scores, output_dir = mine_head_scores(
        model, dataset_val, bundle["config"], bundle["config_name"],
        bundle["n_heads"], bundle["layers"], bundle["preprocess"], args, args.model_name, seed)
    if mode == "zsweep":
        mining_elapsed = time.time() - mining_start
        timings["circuit_mining_sec"] = mining_elapsed
        print(f"[time] circuit mining: done in {fmt_dt(mining_elapsed)} ({mining_elapsed:.1f}s)")

    print("Head Scores (Layer, Head) -> Score:")
    for key in head_scores:
        print(f"  Layer {key[0]}, Head {key[1]}: {head_scores[key]:.4f}")

    if mode == "cv":
        run_cv(model, head_scores, output_dir, bundle["layers"], bundle["n_heads"],
               bundle["preprocess"], args.model_name, args)
    else:
        run_zsweep(model, head_scores, output_dir, bundle["layers"], bundle["n_heads"],
                   bundle["preprocess"], args.model_name, args, timings, wall_start)


if __name__ == "__main__":
    main()
