# ECCV Supplementary — Typographic-Attack Attribution for CLIP & VLMs

Code release for typographic-attack circuit attribution: we mine
the attention heads responsible for reading text in an image (via per-head concept
attribution) and ablate them to recover the true object. Two evaluation
pipelines share the same attribution core:

- **CLIP** zero-shot classification (`clip_eval.py`)
- **Vision-language models** — Gemma / InternVL / Qwen — generative VQA (`vlm_eval.py`)

Every local dependency is vendored into this directory, so it runs without the
surrounding research repository.

## Layout

```
eccv/
├── pyproject.toml / requirements.txt   # dependencies
│
├── clip_eval.py            # CLIP classification eval     — entry point (--mode cv|zsweep)
├── clip_common.py          #   └─ shared load / mining / eval helpers
├── vlm_eval.py             # VLM (Gemma/InternVL/Qwen) eval — entry point (--model, --mode)
├── vlm_common.py           #   └─ shared utils (SAE, attention hooks, scoring, ablation)
├── vlm_models.py           #   └─ per-family model registry (load_vlm)
├── vlm_results_table.py    # aggregate VLM result JSONs -> camera-ready LaTeX table
├── internvl_fuse_moe.py    # one-off: fuse InternVL3.5-30B-A3B MoE expert weights
├── eccv_supp.ipynb         # supplementary figures / analysis notebook
│
├── eval_utils.py           # shared eval helpers (datasets, head ablation, scoring)
├── proj_utils/             # checkpoint paths, text-overlay transforms, text utils
├── dataset_utils/          # ImageNet dataset wrappers (+ imagenet21k_labels.txt)
├── model_training/         # Contains SAE definition, unused in this work
├── models/                 # CLIP / DINO / DINOv2 ViT loaders (get_fn_model_loader)
└── experiments/            # attribution maps + concept loading helpers
```

Both `clip_eval` and `vlm_eval` collapse what used to be one script per
model/mode — model family and evaluation mode are now command-line arguments.

## Install

```bash
pip install -r requirements.txt          # or:  pip install -e .
```

`models/dinov2_vit.py` lazily imports `dinov2` (Facebook's repo) only when a
DINOv2 backbone is requested; install it separately if you need those models.

## Entry points

### CLIP classification — `clip_eval.py`

```bash
python clip_eval.py --model_name clip_vit_l14_datacomp_xl_s13b_b90k \
                    --config_fn <config>.yaml --mode cv     --gpu 2,3
python clip_eval.py --model_name clip_vit_l14_datacomp_xl_s13b_b90k \
                    --config_fn <config>.yaml --mode zsweep --gpu 2,3
```

- `--mode cv` — cross-validate the z-threshold on `--cv_dataset`, then evaluate
  with the best head spec → `{model}_{cv_dataset}_attribution.json`
- `--mode zsweep` — evaluate every z-threshold in `CANDIDATES`, with timing →
  `{model}_zsweep.json`

### VLM evaluation — `vlm_eval.py`

```bash
python vlm_eval.py --model {gemma,ivl,qwen} --mode {base,prompt,zsweep,prompt_zsweep} \
                   --size N --subset N [--clean] --gpu 2,3
```

- `--model` selects the family; `--size` indexes that family's model list
  (see `vlm_models.py`)
- `--mode`: `base` (single z=1 spec) · `prompt` (+ "ignore the textual shapes"
  instruction) · `zsweep` (sigma sweep, one JSONL per sigma) · `prompt_zsweep`
- writes JSONL under `vlm_results/` (or `vlm_zsweep_results/`);
  `vlm_results_table.py` aggregates the clean-set results into a LaTeX table

`--gpu 2,3` sets `CUDA_VISIBLE_DEVICES` before torch initialises (so it composes
with `device_map="auto"`).

## Runtime notes

- **Working directory.** Run from this `eccv/` directory: `dataset_utils/imagenet.py`
  loads `dataset_utils/imagenet21k_labels.txt` via a path relative to the CWD, and
  the scripts reference `./data`, `./cache`, `./configs` relatively. Place (or
  symlink) those alongside the code here.
- **External data / checkpoints are not vendored.** The pipelines expect dataset
  folders (`./data/rta100`, `./data/disentangle`, `./data/paintds`,
  `./data/imagenet_100_text`, `./data/kaggle_imnet100`, and RIO-Bench under
  `./data/hub` for the VLMs), model config YAMLs under `configs/train_sae/imagenet/...`,
  and model weights under `./cache`.

## Acknowledgements

Built on the [Attributing-CLIP](https://github.com/maxdreyer/attributing-clip) codebase.