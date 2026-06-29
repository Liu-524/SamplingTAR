import torch
import torch.nn as nn
import open_clip
from open_clip.model import VisionTransformer
from functools import partial

# --- Refactored Adaptation Function ---
def adapt_dino_to_open_clip(dino_model: nn.Module) -> VisionTransformer:
    """
    Adapts a DINO ViT model to the open_clip ViT architecture by first creating a
    compatible configuration and then initializing a new open_clip model.

    This function handles all necessary architectural modifications and weight transfers,
    making the returned open_clip model a functional equivalent of the DINO backbone.
    """
    print("--- Starting DINO to open_clip Adaptation (Config-First Approach) ---")

    # --- Step 1: Infer configuration from the DINO model ---
    print("Step 1: Inferring model configuration from DINO model...")
    dino_config = {
        'image_size': 224,
        'patch_size': dino_model.patch_embed.patch_size,
        'width': dino_model.embed_dim,
        'layers': len(dino_model.blocks),
        'heads': dino_model.blocks[0].attn.num_heads,
        # DINO's LayerNorm has eps=1e-6, which is crucial for numerical equivalence.
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'mlp_ratio': dino_model.blocks[0].mlp.fc1.bias.shape[0] // dino_model.blocks[0].mlp.fc2.bias.shape[0],
        'output_dim': dino_model.embed_dim # Not strictly necessary for backbone but good practice
    }
    print(f" -> Inferred Config: {dino_config}")

    # --- Step 2: Initialize open_clip model with the DINO-compatible config ---
    print("\nStep 2: Initializing new open_clip VisionTransformer with custom config...")
    adapted_model = VisionTransformer(**dino_config)
    adapted_model.hidden_dim = dino_model.embed_dim  # Ensure hidden_dim is set correctly
    device = next(dino_model.parameters()).device
    adapted_model.to(device)
    

    # --- Step 3: Perform necessary post-initialization architectural tweaks ---
    print("\nStep 3: Performing post-initialization architectural tweaks...")

    # FIX 1: DINO's patch embedding conv layer has a bias. Replace the default one.
    dino_patch_embed = dino_model.patch_embed.proj
    new_conv1 = nn.Conv2d(
        in_channels=dino_patch_embed.in_channels,
        out_channels=dino_patch_embed.out_channels,
        kernel_size=dino_patch_embed.kernel_size,
        stride=dino_patch_embed.stride,
        bias=True  # Ensure bias is enabled to match DINO
    ).to(device)
    adapted_model.conv1 = new_conv1
    print(" -> Replaced open_clip's conv1 with a DINO-compatible Conv2d layer (with bias).")

    # FIX 2: DINO does not have a pre-transformer LayerNorm. Replace with Identity.
    adapted_model.ln_pre = nn.Identity()
    print(" -> Replaced open_clip's ln_pre with nn.Identity to match DINO architecture.")
    
    # FIX 3: Initialize the projection layer head, which DINO doesn't have.
    # We initialize it as an identity matrix as it's not part of the DINO backbone weights.
    adapted_model.proj = nn.Parameter(torch.eye(dino_model.embed_dim, device=device))
    print(" -> Initialized the final projection layer ('proj') as an identity matrix.")


    # --- Step 4: Map and load state dictionary ---
    print("\nStep 4: Mapping and loading weights from DINO to the new model...")
    dino_state_dict = dino_model.state_dict()
    mapped_weights = {}

    for dino_key, dino_weight in dino_state_dict.items():
        new_key = dino_key

        # --- Handle key remapping and shape adjustments ---
        if dino_key == 'cls_token':
            new_key = 'class_embedding'
            # DINO [1, 1, 384] vs OpenCLIP [1, 384]
            dino_weight = dino_weight.squeeze()
        elif dino_key == 'pos_embed':
            new_key = 'positional_embedding'
            # DINO [1, 197, 384] vs OpenCLIP [197, 384]
            dino_weight = dino_weight.squeeze(0)
        elif dino_key.startswith('patch_embed.proj'):
            new_key = dino_key.replace('patch_embed.proj', 'conv1')
        elif dino_key.startswith('blocks.'):
            new_key = new_key.replace('blocks.', 'transformer.resblocks.')
            new_key = new_key.replace('norm1', 'ln_1')
            new_key = new_key.replace('norm2', 'ln_2')
            new_key = new_key.replace('attn.qkv.', 'attn.in_proj_')
            new_key = new_key.replace('attn.proj.', 'attn.out_proj.')
            new_key = new_key.replace('mlp.fc1.', 'mlp.c_fc.')
            new_key = new_key.replace('mlp.fc2.', 'mlp.c_proj.')
        elif dino_key.startswith('norm.'):
            new_key = new_key.replace('norm.', 'ln_post.')
        elif dino_key.startswith('head.'):
            # We are only adapting the backbone, so we skip DINO's head.
            continue

        mapped_weights[new_key] = dino_weight

    msg = adapted_model.load_state_dict(mapped_weights, strict=False)
    print(" -> Weight loading message:", msg)

    # All weights should be loaded except for the `proj` parameter we initialized manually.
    expected_missing = ['proj']
    unexpectedly_missing = [k for k in msg.missing_keys if k not in expected_missing]

    assert not unexpectedly_missing, f"Unexpected missing keys found: {unexpectedly_missing}"
    assert not msg.unexpected_keys, f"Unexpected keys found in checkpoint: {msg.unexpected_keys}"

    print("\n -> Assertion Passed: All matching weights loaded successfully.")
    print("--- Adaptation Complete ---")

    return adapted_model


def adapt_dinov2_to_open_clip(dino_model: nn.Module) -> VisionTransformer:
    """
    Adapts a DINO ViT model to the open_clip ViT architecture by first creating a
    compatible configuration and then initializing a new open_clip model.

    This function handles all necessary architectural modifications and weight transfers,
    making the returned open_clip model a functional equivalent of the DINO backbone.
    """
    print("--- Starting DINO to open_clip Adaptation (Config-First Approach) ---")

    # --- Step 1: Infer configuration from the DINO model ---
    print("Step 1: Inferring model configuration from DINO model...")
    dino_config = {
        'image_size': 224,
        'patch_size': dino_model.patch_embed.patch_size,
        'width': dino_model.embed_dim,
        'layers': len(dino_model.blocks),
        'heads': dino_model.blocks[0].attn.num_heads,
        # DINO's LayerNorm has eps=1e-6, which is crucial for numerical equivalence.
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'mlp_ratio': dino_model.blocks[0].mlp.fc1.bias.shape[0] // dino_model.blocks[0].mlp.fc2.bias.shape[0],
        'output_dim': dino_model.embed_dim # Not strictly necessary for backbone but good practice
    }
    print(f" -> Inferred Config: {dino_config}")

    # --- Step 2: Initialize open_clip model with the DINO-compatible config ---
    print("\nStep 2: Initializing new open_clip VisionTransformer with custom config...")
    adapted_model = VisionTransformer(**dino_config)
    device = next(dino_model.parameters()).device
    adapted_model.to(device)

    # --- Step 3: Perform necessary post-initialization architectural tweaks ---
    print("\nStep 3: Performing post-initialization architectural tweaks...")

    adapted_model.positional_embedding = nn.Parameter(dino_model.pos_embed[0])

    for i, block in enumerate(adapted_model.transformer.resblocks):
        block.ls_1 = dino_model.blocks[i].ls1
        block.ls_2 = dino_model.blocks[i].ls2

    # FIX 1: DINO's patch embedding conv layer has a bias. Replace the default one.
    dino_patch_embed = dino_model.patch_embed.proj
    new_conv1 = nn.Conv2d(
        in_channels=dino_patch_embed.in_channels,
        out_channels=dino_patch_embed.out_channels,
        kernel_size=dino_patch_embed.kernel_size,
        stride=dino_patch_embed.stride,
        bias=True  # Ensure bias is enabled to match DINO
    ).to(device)
    adapted_model.conv1 = new_conv1
    print(" -> Replaced open_clip's conv1 with a DINO-compatible Conv2d layer (with bias).")

    # FIX 2: DINO does not have a pre-transformer LayerNorm. Replace with Identity.
    adapted_model.ln_pre = nn.Identity()
    print(" -> Replaced open_clip's ln_pre with nn.Identity to match DINO architecture.")

    # FIX 3: Initialize the projection layer head, which DINO doesn't have.
    # We initialize it as an identity matrix as it's not part of the DINO backbone weights.
    adapted_model.proj = nn.Parameter(torch.eye(dino_model.embed_dim, device=device))
    if hasattr(dino_model, 'register_tokens') and dino_model.register_tokens is not None:
        adapted_model.register_tokens = nn.Parameter(torch.zeros_like(dino_model.register_tokens))
        
    print(" -> Initialized the final projection layer ('proj') as an identity matrix.")


    # --- Step 4: Map and load state dictionary ---
    print("\nStep 4: Mapping and loading weights from DINO to the new model...")
    dino_state_dict = dino_model.state_dict()
    mapped_weights = {}

    for dino_key, dino_weight in dino_state_dict.items():
        new_key = dino_key

        # --- Handle key remapping and shape adjustments ---
        if dino_key == 'cls_token':
            new_key = 'class_embedding'
            # DINO [1, 1, 384] vs OpenCLIP [1, 384]
            dino_weight = dino_weight.squeeze()
        elif dino_key == 'pos_embed':
            new_key = 'positional_embedding'
            # DINO [1, 197, 384] vs OpenCLIP [197, 384]
            dino_weight = dino_weight.squeeze(0)
        elif dino_key.startswith('patch_embed.proj'):
            new_key = dino_key.replace('patch_embed.proj', 'conv1')
        elif dino_key.startswith('blocks.'):
            new_key = new_key.replace('blocks.', 'transformer.resblocks.')
            new_key = new_key.replace('norm1', 'ln_1')
            new_key = new_key.replace('norm2', 'ln_2')
            new_key = new_key.replace('ls1', 'ls_1')
            new_key = new_key.replace('ls2', 'ls_2')
            new_key = new_key.replace('attn.qkv.', 'attn.in_proj_')
            new_key = new_key.replace('attn.proj.', 'attn.out_proj.')
            new_key = new_key.replace('mlp.fc1.', 'mlp.c_fc.')
            new_key = new_key.replace('mlp.fc2.', 'mlp.c_proj.')
        elif dino_key.startswith('norm.'):
            new_key = new_key.replace('norm.', 'ln_post.')
        elif dino_key.startswith('head.'):
            # We are only adapting the backbone, so we skip DINO's head.
            continue

        mapped_weights[new_key] = dino_weight

    msg = adapted_model.load_state_dict(mapped_weights, strict=False)
    print(" -> Weight loading message:", msg)

    # All weights should be loaded except for the `proj` parameter we initialized manually.
    expected_missing = ['proj']
    expected_unloaded = ['mask_token']
    unexpectedly_missing = [k for k in msg.missing_keys if k not in expected_missing]
    unexpectedly_present = [k for k in msg.unexpected_keys if k not in expected_unloaded]

    assert not unexpectedly_missing, f"Unexpected missing keys found: {unexpectedly_missing}"
    assert not unexpectedly_present, f"Unexpected keys found in checkpoint: {unexpectedly_present}"
    if hasattr(adapted_model, 'register_tokens'):
        adapted_model.class_embedding = nn.Parameter(torch.cat((adapted_model.class_embedding.unsqueeze(0).unsqueeze(1), adapted_model.register_tokens.data), dim=1))
        adapted_model.num_registers = adapted_model.class_embedding.shape[1] - 1
    else:
        adapted_model.num_registers = 0
    print("\n -> Assertion Passed: All matching weights loaded successfully.")
    print("--- Adaptation Complete ---")

    return adapted_model