import os
import glob
import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import snapshot_download

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3_5-30B-A3B-HF"
OUTPUT_DIR = "./cache/OpenGVLab/InternVL3_5-30B-A3B-HF"
NUM_EXPERTS = 128
DTYPE = torch.float32
REVERSE_SWIGLU_ORDER = False # Flip to True ONLY if the saved model still outputs gibberish

print(f"1. Locating original weights for {MODEL_ID} in local cache...")
# This will not redownload the model if it's already in your huggingface cache
cache_dir = snapshot_download(MODEL_ID, allow_patterns="*.safetensors", cache_dir='./data/hub')
shard_files = glob.glob(os.path.join(cache_dir, "*.safetensors"))

print("\n2. Loading model skeleton (this may take a moment)...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir='./data/hub')
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=DTYPE,
    trust_remote_code=True,
    cache_dir='./data/hub',
)

NUM_LAYERS = len(model.model.language_model.layers)
expected_total_injections = NUM_LAYERS * NUM_EXPERTS
injected_registry = set()

print("\n3. Injecting fused MoE weights directly into VRAM...")
with torch.no_grad():
    for shard_path in shard_files:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            shard_keys = f.keys()
            
            layers_in_shard = set()
            for key in shard_keys:
                if "mlp.experts" in key and "gate_proj" in key:
                    layer_idx = int(key.split("layers.")[1].split(".")[0])
                    layers_in_shard.add(layer_idx)
                    
            for L in layers_in_shard:
                print(f"  -> Fusing and injecting Layer {L}...")
                
                target_gate_up = model.model.language_model.layers[L].mlp.experts.gate_up_proj
                target_down = model.model.language_model.layers[L].mlp.experts.down_proj
                t_device = target_gate_up.device
                
                for E in range(NUM_EXPERTS):
                    gate_key = f"language_model.model.layers.{L}.mlp.experts.{E}.gate_proj.weight"
                    up_key   = f"language_model.model.layers.{L}.mlp.experts.{E}.up_proj.weight"
                    down_key = f"language_model.model.layers.{L}.mlp.experts.{E}.down_proj.weight"
                    
                    if gate_key in shard_keys:
                        # --- VERIFICATION CHECK 1: NO DUPLICATES ---
                        if (L, E) in injected_registry:
                            raise RuntimeError(f"FATAL: Expert {E} in Layer {L} is being injected twice! Check your shards.")
                        
                        gate = f.get_tensor(gate_key).to(device=t_device, dtype=DTYPE)
                        up = f.get_tensor(up_key).to(device=t_device, dtype=DTYPE)
                        down = f.get_tensor(down_key).to(device=t_device, dtype=DTYPE)
                        
                        out_dim = gate.shape[0]
                        
                        if REVERSE_SWIGLU_ORDER:
                            target_gate_up.data[E, :out_dim, :] = up
                            target_gate_up.data[E, out_dim:, :] = gate
                        else:
                            target_gate_up.data[E, :out_dim, :] = gate
                            target_gate_up.data[E, out_dim:, :] = up
                            
                        target_down.data[E, :, :] = down
                        
                        # Register the successful injection
                        injected_registry.add((L, E))

print("\n--- Running Injection Verification ---")
# --- VERIFICATION CHECK 2: NO MISSING WEIGHTS ---
if len(injected_registry) != expected_total_injections:
    missing_count = expected_total_injections - len(injected_registry)
    raise RuntimeError(f"FATAL: Missing {missing_count} expert injections! Expected {expected_total_injections}, got {len(injected_registry)}.")
print(f"Success! All {expected_total_injections} experts safely injected exactly once.")

print(f"\n4. Saving fused model and processor to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.model.save_pretrained(
    OUTPUT_DIR, 
    safe_serialization=True, 
    max_shard_size="15GB"
)
processor.save_pretrained(OUTPUT_DIR)

print(f"\n--- Done! You can now load the model locally from {OUTPUT_DIR} ---")