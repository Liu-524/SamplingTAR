from typing import Optional

import open_clip
import torch.hub
torch.hub.set_dir('/bigtemp2/qzp4ta/torch_hub')

from torch.utils.checkpoint import checkpoint
from torchvision import transforms as pth_transforms
from models.dino_adapter import adapt_dino_to_open_clip

from torch.nn.functional import scaled_dot_product_attention

from models.clip_vit import forward_transformer as forward_transformer_clip

def get_dino_val_transform(image_size=224):
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return val_transform
def flash_attn_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
    attn_output = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False, scale=self.scale)
    attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)
    return attn_output

def block_forward(self, x: torch.Tensor, return_attention: bool = False):
    B, N, C = x.shape
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

def get_dino_as_clip(
        name: str = "dino_vitb8",
        *args,
        **kwargs):
    
    dino_model = torch.hub.load('facebookresearch/dino:main', name, force_reload=False)
    clip_model = adapt_dino_to_open_clip(dino_model)

    clip_model.preprocess = get_dino_val_transform()
    def add_sae(self, sae):
        # convert negative layer index to positive
        sae.layer_idx = sae.layer_idx if sae.layer_idx >= 0 else len(clip_model.transformer.resblocks) + sae.layer_idx
        self.sae = sae
        self.transformer.forward = forward_transformer_clip.__get__(self)
        self.use_sae = True

    setattr(clip_model, "add_sae", add_sae.__get__(clip_model))
    # clip_model.transformer = clip_model
    return clip_model

def get_dino_vit(
        name: str = "dino_vitb8",
        load_as_clip: bool = False,
        *args,
        **kwargs):
    if load_as_clip:
        return get_dino_as_clip(name, *args, **kwargs)
    dino_model = torch.hub.load('facebookresearch/dino:main', name, force_reload=False)
    
    dino_model.preprocess = get_dino_val_transform()

    # unifying the forward pass into one single branch for inference
    for module in dino_model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()

    
    dino_model.hidden_dim = dino_model.embed_dim

    for block in dino_model.blocks:
        block.forward = block_forward.__get__(block)
        block.attn.forward = flash_attn_forward.__get__(block.attn)

    def add_sae(self, sae):
        # convert negative layer index to positive
        sae.layer_idx = sae.layer_idx if sae.layer_idx >= 0 else len(dino_model.blocks) + sae.layer_idx
        self.sae = sae
        self.forward = forward_transformer.__get__(self)
        self.use_sae = True
        self.grad_checkpointing = False if not hasattr(self, 'grad_checkpointing') else self.grad_checkpointing

    setattr(dino_model, "add_sae", add_sae.__get__(dino_model))

    return dino_model



def forward_resblock(block, x: torch.Tensor):
    
    N, L, C = x.shape
    hook_output = {}
    def forward_pre_hook(module, input, output):
        hook_output['head_output'] = input[0].clone()
        hook_output['proj_output'] = output.clone()
        return output
    
    handle = block.attn.proj.register_forward_hook(forward_pre_hook)
    
    _ =  block(x)
    head_output = hook_output.get('head_output', None)
    proj_output = hook_output.get('proj_output', None)
    if head_output is None or proj_output is None:
        raise RuntimeError("Forward hook did not capture expected outputs.")
    head_output = head_output.reshape(N, L , block.attn.num_heads, -1)
    handle.remove()
    return x, head_output, proj_output

def continue_forward_resblock(block, x: torch.Tensor, sae_out: torch.Tensor, transcode: bool = False, attn_mask: Optional[torch.Tensor] = None):
    if not transcode:
        sae_out = block.attn.proj(sae_out)
        sae_out = block.attn.proj_drop(sae_out)
    sae_out = block.drop_path((sae_out))
    sae_out = x + sae_out
    x = sae_out + block.drop_path(block.mlp(block.norm2(sae_out)))
    return x


def forward_transformer(self, x: torch.Tensor):
    x = self.prepare_tokens(x)
    for i, r in enumerate(self.blocks):
        if self.use_sae and i == self.sae.layer_idx:
            # Apply SAE to each token embedding (e.g., shape [B, T, D])
            x, pre, post = forward_resblock(r, x)
            if self.sae.transcode:
                if self.sae.token_type == "cls":
                    post_recon, _, _ = self.sae(x[:, :1], pre[:, :1], post[:, :1])  # Apply SAE to the CLS token
                    post_ori = post[:, 1:]
                    post_recon = post_recon[:, :1]
                    post = torch.cat([post_recon, post_ori], dim=1)
                elif self.sae.token_type == "spatial":
                    post_recon, _, _ = self.sae(x[:, 1:], pre[:, 1:], post[:, 1:])  # Apply SAE to the spatial tokens
                    post_ori = post[:, :1]
                    post = torch.cat([post_ori, post_recon], dim=1)
                x = continue_forward_resblock(r, x, post, self.sae.transcode)
            else:
                if self.sae.token_type == "cls":
                    pre_recon, _, _ = self.sae(x[:, :1], pre[:, :1], post[:, :1])  # Apply SAE to the CLS token
                    pre_ori = pre[:, 1:]
                    pre_recon = pre_recon[:, :1]
                    pre = torch.cat([pre_recon, pre_ori], dim=1)
                elif self.sae.token_type == "spatial":
                    pre_recon, _, _ = self.sae(x[:, 1:], pre[:, 1:], post[:, 1:])  # Apply SAE to the spatial tokens
                    pre_ori = pre[:, :1]
                    pre = torch.cat([pre_ori, pre_recon], dim=1)
                x = continue_forward_resblock(r, x, pre, self.sae.transcode)
        else:
            x = checkpoint(r, x, None, None) if self.grad_checkpointing and not torch.jit.is_scripting() else r(x)
    
    x = self.norm(x)
    return x


if __name__ == "__main__":
    vit = get_dino_vit("dino_vitb8", pretrained="dino_vitb8")
