from typing import Optional

import open_clip
import torch.hub
import einops

from torch.utils.checkpoint import checkpoint


def get_clip_vit(
        name,
        pretrained: str = "datacomp_xl_s13b_b90k",
        return_clip_model: bool = False,
        *args,
        **kwargs):
    if kwargs.get("config_only", False):
        return open_clip.get_model_config(name)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, cache_dir='/bigtemp2/qzp4ta/open_clip')

    # unifying the forward pass into one single branch for inference
    for module in clip_model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()

    vision_model = clip_model.visual
    vision_model.preprocess = clip_model.preprocess = preprocess
    vision_model.hidden_dim = vision_model.transformer.width

    def add_sae(self, sae):
        # convert negative layer index to positive
        sae.layer_idx = sae.layer_idx if sae.layer_idx >= 0 else len(vision_model.transformer.resblocks) + sae.layer_idx
        self.sae = sae
        self.transformer.forward = forward_transformer.__get__(self)
        self.use_sae = True

    setattr(vision_model, "add_sae", add_sae.__get__(vision_model))

    # if hasattr(vision_model, "transformer"):
    #     vision_model.transformer.forward = forward_transformer.__get__(vision_model)
    # else:
    #     print("Warning: No transformer found in the vision model. No SAE will be applied.")

    if return_clip_model:
        clip_model.tokenizer = open_clip.get_tokenizer(name)
        return clip_model

    return vision_model

def per_head_recon_handler(block, proj_in, proj_out, per_head_recon):
    if not per_head_recon:
        return proj_out
    n, l, h, c = proj_in.shape
    M = block.attn.out_proj.weight.data.clone()
    b = block.attn.out_proj.bias.data.clone()
    M = M.view(M.shape[0], h, c)
    head_contribution = torch.einsum('nlhc,ohc->nlho', proj_in, M) + b.unsqueeze(0).unsqueeze(1).unsqueeze(2) / h
    return torch.cat([proj_out.unsqueeze(2), head_contribution], dim=2)




def forward_resblock(block, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, per_head_recon: bool = False):
    
    N, L, C = x.shape
    hook_output = {}
    def forward_pre_hook(module, input, output):
        hook_output['head_output'] = input[0].clone()
        hook_output['proj_output'] = output.clone()
        return output
    
    handle = block.attn.out_proj.register_forward_hook(forward_pre_hook)
    
    _ =  block(x, attn_mask=attn_mask)
    head_output = hook_output.get('head_output', None)
    proj_output = hook_output.get('proj_output', None)
    proj_output = block.ls_1(block.ln_attn(proj_output))
    if head_output is None or proj_output is None:
        raise RuntimeError("Forward hook did not capture expected outputs.")
    head_output = head_output.reshape(N, L , block.attn.num_heads, -1)
    handle.remove()
    proj_output = per_head_recon_handler(block, head_output, proj_output, per_head_recon)
    return x, head_output, proj_output

def continue_forward_resblock(block, x: torch.Tensor, sae_out: torch.Tensor, transcode: bool = False, attn_mask: Optional[torch.Tensor] = None):
    if not transcode:
        sae_out = block.attn.out_proj(sae_out)
        sae_out = block.attn.out_drop(sae_out)
        sae_out = block.ls_1(block.ln_attn(sae_out))
    sae_out = x + sae_out
    x = sae_out + block.ls_2(block.mlp(block.ln_2(sae_out)))
    return x

def process_sae_output(post_ori, post_recon, token_type, per_head_recon):
    if per_head_recon:
        post_ori_ = post_ori[:, :, 0]
    else:
        post_ori_ = post_ori
    if token_type == "cls":
        return torch.cat([post_recon[:, :1], post_ori_[:, 1:]], dim=1)
    elif token_type == "spatial":
        return torch.cat([post_ori_[:, :1], post_recon[:, 1:]], dim=1)
    else:
        raise ValueError(f"Unknown token_type: {token_type}")
        

def forward_transformer(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    if not self.transformer.batch_first:
        x = x.transpose(0, 1).contiguous()
    # if self.sae.training:
    #     x_l, pre_l, post_l = [], [], []
    #     for i, r in enumerate(self.transformer.resblocks):
    #         if self.use_sae and i >= self.sae.layer_idx:
    #             x, pre, post = forward_resblock(r, x, attn_mask)
    #             # recon, _, _ = self.sae(x, pre, post)
    #             x = continue_forward_resblock(r, x, post + x, True, attn_mask)
    #             x_l.append(x)
    #             pre_l.append(pre)
    #             post_l.append(post)
    #         else:
    #             x = checkpoint(r, x, None, None, attn_mask) if self.transformer.grad_checkpointing and not torch.jit.is_scripting() else r(x, attn_mask=attn_mask)
    #     x = torch.cat(x_l, dim=0)
    #     pre = torch.cat(pre_l, dim=0)
    #     post = torch.cat(post_l, dim=0)
    #     recon, _, _ = self.sae(x, pre, post)
    #     return x
    
    for i, r in enumerate(self.transformer.resblocks):
        if self.use_sae and i == self.sae.layer_idx:
            # Apply SAE to each token embedding (e.g., shape [B, T, D])
            x, pre, post = forward_resblock(r, x, attn_mask, self.sae.per_head_recon)
            if self.sae.transcode:
                if self.sae.token_type == "cls":
                    post_recon, _, _ = self.sae(x[:, :1], pre[:, :1], post[:, :1])  # Apply SAE to the CLS token
                elif self.sae.token_type == "spatial":
                    post_recon, _, _ = self.sae(x[:, 1:], pre[:, 1:], post[:, 1:])  # Apply SAE to the spatial tokens

                post = process_sae_output(post, post_recon, self.sae.token_type, self.sae.per_head_recon)
                x = continue_forward_resblock(r, x, post, self.sae.transcode, attn_mask)
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
                x = continue_forward_resblock(r, x, pre, self.sae.transcode, attn_mask)
        else:
            x = checkpoint(r, x, None, None, attn_mask) if self.transformer.grad_checkpointing and not torch.jit.is_scripting() else r(x, attn_mask=attn_mask)
        
    if not self.transformer.batch_first:
        x = x.transpose(0, 1)
    return x


if __name__ == "__main__":
    vit = get_clip_vit()
