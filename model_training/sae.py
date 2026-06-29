from abc import abstractmethod

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class BatchedLinear(nn.Module):
    """
    A custom PyTorch layer for batched linear projections with weight normalization.

    This module performs `n_heads` independent linear transformations on the last
    dimension of an input tensor. It is designed to project the outputs of
    multiple attention heads simultaneously using an efficient `torch.bmm`
    operation. Weight normalization is applied to each head's projection matrix
    independently.

    Args:
        n_heads (int): The number of independent projections (e.g., number of attention heads).
        in_features (int): The size of each input sample (e.g., head_dim).
        out_features (int): The size of each output sample (e.g., proj_dim).
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
    """
    def __init__(self, n_heads: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = out_features

        # Register the batched weight as a parameter.
        # The shape is (n_heads, out_features, in_features) which follows the
        # convention of nn.Linear (out, in).
        self.weight = nn.Parameter(torch.empty(n_heads, out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(n_heads, out_features))
        else:
            # If no bias, register it as None
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the weights and biases for each head."""
        # Kaiming uniform initialization is a good default for layers followed by ReLU
        for i in range(self.n_heads):
            nn.init.kaiming_uniform_(self.weight[i], a=torch.sqrt(5))
            
        if self.bias is not None:
            # Calculate fan-in for each head's weight matrix to initialize bias
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            if fan_in != 0:
                bound = 1 / torch.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the batched linear projection.

        Args:
            x (torch.Tensor): Input tensor from multi-head attention.
                              Shape: (batch_size, seq_len, n_heads, in_features).

        Returns:
            torch.Tensor: The projected output tensor.
                          Shape: (batch_size, seq_len, n_heads, out_features).
        """
        batch_size, seq_len, _, _ = x.shape
        
        # --- Prepare Tensors for Batched Matrix Multiplication (bmm) ---
        # `torch.bmm` expects 3D tensors of shape (b, n, m) and (b, m, p).
        # We treat `n_heads` as the batch dimension 'b' for the matmul.
        
        # Reshape input x: (batch, seq, n_heads, in_features) -> (n_heads, batch * seq, in_features)
        x_reshaped = x.permute(2, 0, 1, 3).reshape(self.n_heads, batch_size * seq_len, self.in_features)

        # The linear transformation is y = xW^T + b.
        # self.weight shape: (n_heads, out_features, in_features)
        # We need to transpose the last two dimensions to get W^T.
        # weight_t shape: (n_heads, in_features, out_features)
        weight_t = self.weight.transpose(1, 2)

        # --- Perform Batched Matrix Multiplication ---
        # (n_heads, batch * seq, in_features) @ (n_heads, in_features, out_features)
        # Result shape: (n_heads, batch * seq, out_features)
        output = torch.bmm(x_reshaped, weight_t)

        # --- Reshape Output and Add Bias ---
        # Reshape output back to a more intuitive format.
        # (n_heads, batch * seq, out_features) -> (n_heads, batch, seq, out_features)
        output = output.view(self.n_heads, batch_size, seq_len, self.out_features)
        # Permute to bring batch_size and seq_len to the front.
        # -> (batch, seq, n_heads, out_features)
        output = output.permute(1, 2, 0, 3)

        if self.bias is not None:
            # self.bias shape is (n_heads, out_features).
            # PyTorch's broadcasting adds it to the last two dimensions of `output`,
            # which is exactly what we want.
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        """Provides a string representation of the layer's configuration."""
        return (f'n_heads={self.n_heads}, in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None}')

class TopKSAE(nn.Module):
    def __init__(self,
                 input_dim=768,
                 n_heads=12,
                 hidden_dim=30000,
                 k=64,
                 layer_idx=-1,
                 token="cls",
                 use_basic=False,
                 transcode=False,
                 norm_activations=False,
                 siamese_encoder=False,
                 use_aux=False,
                 *args,
                 **kwargs):
        """
        Top-k Sparse Autoencoder (SAE) with Weight Normalization.

        Args:
        - input_dim (int): Input feature dimension.
        - hidden_dim (int): Latent space dimension.
        - k (int): Number of neurons allowed to be active.
        - layer_idx (int): Index of the layer where SAE is applied. Negative index counts from the end.
        - sae_token (str): Token type for SAE, e.g., "cls" for classification token or 'spatial' for spatial tokens.
        """
        super(TopKSAE, self).__init__()
        print(f"TopKSAE: input_dim={input_dim}, hidden_dim={hidden_dim}, k={k}, layer_idx={layer_idx}, token={token}")
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.token_type = token
        self.hook_pre_sae = nn.Identity()
        self.hook_hidden_post = nn.Identity()
        self.hook_post_sae = nn.Identity()
        self.hook_aux_sae = nn.Identity()
        self.add_residual = True  # Whether to add residual connections in SAE
        self.n_heads = n_heads
        
        self.decoder = weight_norm(nn.Linear(hidden_dim, input_dim, bias=True), dim=1)  # Apply weight norm
        self._fix_weight_norm()
        self.use_basic = use_basic
        self.transcode = transcode
        self.norm_activations = norm_activations
        self.siamese_encoder = siamese_encoder
        self.use_aux = use_aux
        self.dead_neurons = None
        self.use_block_loss = False
        self.block_coeff = 0.1
        self.per_head_recon = self.use_basic == False # Only use per-head recon with head-decomposed SAE
        
        print("Using transcoder target." if self.transcode else "Using reconstruction target.")
        print("Using normalized activations." if self.norm_activations else "Using activations as is.")
        if use_basic:
            self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim, hidden_dim)))])
        else:
            if siamese_encoder:
                print("Using siamese encoder for SAE.")
                self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim // n_heads, hidden_dim // n_heads))) for _ in range(1)])
            else:
                print("Using independent encoders for SAE.")
                self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim // n_heads, hidden_dim // n_heads))) for _ in range(n_heads)])
        assert not use_basic or not (self.norm_activations or self.siamese_encoder), "Basic SAE cannot be combined with normalized activations or siamese encoder."
        print(f'SAE encoder dim: {self.encoders[0][0].weight.shape}')

    def _fix_weight_norm(self):
        """ Set the magnitude parameter (g) to 1 and remove it from train_sae. """
        for layer in [self.decoder]:  # Access weight-normed layers
            layer.weight_g.data.fill_(1.0)  # Set scale to 1
            layer.weight_g.requires_grad = False  # Freeze scale parameter

    def top_k_masking(self, z, k=None):

        """ top-k activation masking. """
        k = self.k if k is None else k
        values, indices = torch.topk(z, k, dim=1)  # Get top-k activations
        mask = torch.zeros_like(z).scatter(1, indices, 1.0)  # Create binary mask
        return z * mask
    def forward_encoder(self, x):
        if self.siamese_encoder:
            z = self.encoders[0](x)
            z = z.reshape(x.shape[0], x.shape[1], -1)
            return z
        zs = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(x[:, i, :])  # Apply each encoder to the corresponding head
            zs.append(z.unsqueeze(1))  # Unsqueeze to add head dimension
        z = torch.stack(zs, dim=1)  # Stack the outputs along the head dimension
        z = z.reshape(x.shape[0], x.shape[1], -1)
        return z
    
    def forward_basic_encoder(self, x):
        z = self.encoders[0](x)
        return z
    
    def alternative_forward(self, x, pre, post):
        N, L, nh, hc = pre.shape
        post= post.permute(2,0,1,3)
        post_all, post_head = post[0], post[1:]
        x_res = post_all
        
        z = self.forward_encoder(pre.reshape(N*L, nh, hc)) # (N * L, nh, nc)
        z_all = z.reshape(N* L, -1)
        z_bar_all = self.top_k_masking(z_all)
        
        z_head = z.reshape(N*L*nh, -1)
        z_bar_head = self.top_k_masking(z_head, k = self.k // nh)
        z_bar_head = z_bar_head.reshape(N* L, nh, -1)


        x_recon_all = self.decoder(z_bar_all).reshape(N, L, -1)
        W_dec_reshaped = self.decoder.weight.view(nh, z_bar_head.shape[-1], -1)
        x_recon_head_flat = torch.einsum('bhk,hkm->bhm', z_bar_head, W_dec_reshaped)
        x_recon_head = x_recon_head_flat.reshape(N, L, nh, -1)



        x_recon = torch.cat([x_recon_all.unsqueeze(2), x_recon_head], dim=2).permute(2,0,1,3)

        self.hook_pre_sae(post)
        self.hook_hidden_post(z_bar_all.reshape(N, L, -1))
        self.hook_post_sae(x_recon)
        if self.add_residual:
            x_recon_all = x_recon_all + (x_res - x_recon_all).detach()
        return x_recon_all, z_all, z_bar_all
        

    def forward(self, x, pre, post):
        if self.per_head_recon and not self.use_basic:
            return self.alternative_forward(x, pre, post)
        N, L, nh, hc = pre.shape
        
        if self.transcode:
            # hack pre SAE hook to "reconstruct" the projected attention output
            post = self.hook_pre_sae(post)
            x_res = post
        else:
            pre = pre.reshape(N, L, nh * hc)
            self.hook_pre_sae(pre)
            x_res = pre
        batch_size = x.shape[0]
        token_dim = x.shape[1]
        if self.use_basic:
            if self.transcode:
                # z = self.forward_basic_encoder(post.reshape(batch_size * token_dim, nh * hc))
                z = self.forward_basic_encoder(pre.reshape(batch_size * token_dim, nh * hc))
            else:
                z = self.forward_basic_encoder(pre.reshape(batch_size * token_dim, nh * hc))
            z_bar = self.top_k_masking(z)
            z_bar = self.hook_hidden_post(z_bar.reshape(batch_size, token_dim, -1))
            x_recon = self.decoder(z_bar.reshape(batch_size * token_dim, -1))
            x_recon = self.hook_post_sae(x_recon.reshape(batch_size, token_dim, -1))
            if self.add_residual:
                x_recon = x_recon + (x_res - x_recon).detach()
            return x_recon, z, z_bar
        else:
            z = self.forward_encoder(pre.reshape(batch_size * token_dim, nh, hc))
            z = z.reshape(batch_size * token_dim, -1)
            z_bar = self.top_k_masking(z)
            z_bar = self.hook_hidden_post(z_bar.reshape(batch_size, token_dim, -1))
            x_recon = self.decoder(z_bar.reshape(batch_size * token_dim, -1))
            x_recon = self.hook_post_sae(x_recon.reshape(batch_size, token_dim, -1))
            
        if self.use_aux and self.dead_neurons is not None:
            z_aux = z + ~self.dead_neurons.unsqueeze(0) * -1e6 # mask out non-dead neurons
            z_aux = self.top_k_masking(z_aux, self.k * 4)
            x_aux = self.decoder(z_aux.reshape(batch_size * token_dim, -1))
            x_aux = x_aux.reshape(batch_size, token_dim, -1)
            x_aux = self.hook_aux_sae(x_aux)
        if self.add_residual:
            x_recon = x_recon + (x_res - x_recon).detach()
        return x_recon, z, z_bar
    @classmethod
    def from_config(cls, config: dict, model: nn.Module = None, model_dim=30000):
        """Instantiate TopKSAE from a config dictionary."""
        sae_keys = ["sae_input_dim", "sae_hidden_dim", "sae_k", "sae_layer_idx", "sae_token", "sae_ckpt_path",
                    "sae_n_heads", "sae_transcode", "sae_siamese_encoder", "sae_use_basic", 'sae_use_aux']
        
        sae_kwargs = {key.replace('sae_', ''): config[key] for key in sae_keys if key in config}
        if model is not None:
            sae_kwargs['input_dim'] = model.hidden_dim
            if config.get('sae_hidden_dim', 30000) < model.hidden_dim:
                sae_kwargs['hidden_dim'] = model.hidden_dim * config.get('sae_hidden_dim', None)
        else:
            sae_kwargs['input_dim'] = model_dim
            if config.get('sae_hidden_dim', 30000) < model_dim:
                sae_kwargs['hidden_dim'] = model_dim * config.get('sae_hidden_dim', None)
        print(sae_kwargs)
        sae = cls(**sae_kwargs)
        if "ckpt_path" in sae_kwargs:
            try:
                print(f"Loading SAE weights from {sae_kwargs['ckpt_path']}")
                state_dict = torch.load(sae_kwargs["ckpt_path"], map_location='cpu')["state_dict"]
                for key in list(state_dict.keys()):
                    if key.startswith('sae.'):
                        new_key = key[len('sae.'):]
                        state_dict[new_key] = state_dict.pop(key)
                state_dict.pop('activation_counts')
                sae.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading SAE checkpoint, using default weights.\n{e}")
        return sae



class ReLUSAE(nn.Module):
    def __init__(self,
                 input_dim=768,
                 n_heads=12,
                 hidden_dim=30000,
                 k=64,
                 layer_idx=-1,
                 token="cls",
                 use_basic=False,
                 transcode=False,
                 norm_activations=False,
                 siamese_encoder=False,
                 use_aux=False,
                 *args,
                 **kwargs):
        """
        Top-k Sparse Autoencoder (SAE) with Weight Normalization.

        Args:
        - input_dim (int): Input feature dimension.
        - hidden_dim (int): Latent space dimension.
        - k (int): Number of neurons allowed to be active.
        - layer_idx (int): Index of the layer where SAE is applied. Negative index counts from the end.
        - sae_token (str): Token type for SAE, e.g., "cls" for classification token or 'spatial' for spatial tokens.
        """
        super(ReLUSAE, self).__init__()
        print(f"ReLUSAE: input_dim={input_dim}, hidden_dim={hidden_dim}, k={k}, layer_idx={layer_idx}, token={token}")
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.token_type = token
        self.hook_pre_sae = nn.Identity()
        self.hook_hidden_post = nn.Identity()
        self.hook_post_sae = nn.Identity()
        self.hook_aux_sae = nn.Identity()
        self.add_residual = True  # Whether to add residual connections in SAE
        self.n_heads = n_heads
        
        self.decoder = weight_norm(nn.Linear(hidden_dim, input_dim, bias=True), dim=1)  # Apply weight norm
        self._fix_weight_norm()
        self.use_basic = use_basic
        self.transcode = transcode
        self.norm_activations = norm_activations
        self.siamese_encoder = siamese_encoder
        self.use_aux = use_aux
        self.dead_neurons = None
        self.use_block_loss = False
        self.block_coeff = 0.1
        self.per_head_recon = True

        if use_aux:
            print("Using auxiliary loss on encoder output.")
        
        print("Using transcoder target." if self.transcode else "Using reconstruction target.")
        print("Using normalized activations." if self.norm_activations else "Using activations as is.")
        if use_basic:
            self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim, hidden_dim)))])
        else:
            if siamese_encoder:
                print("Using siamese encoder for SAE.")
                self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim // n_heads, hidden_dim // n_heads))) for _ in range(1)])
            else:
                print("Using independent encoders for SAE.")
                self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim // n_heads, hidden_dim // n_heads))) for _ in range(n_heads)])
        assert not use_basic or not (self.norm_activations or self.siamese_encoder), "Basic SAE cannot be combined with normalized activations or siamese encoder."
        print(f'SAE encoder dim: {self.encoders[0][0].weight.shape}')
    def zero_encoder_biases(self):
        """ Utility function to zero out biases of all encoder layers. """
        for encoder in self.encoders:
            for layer in encoder:
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    def _fix_weight_norm(self):
        """ Set the magnitude parameter (g) to 1 and remove it from train_sae. """
        for layer in [self.decoder]:  # Access weight-normed layers
            layer.weight_g.data.fill_(1.0)  # Set scale to 1
            layer.weight_g.requires_grad = False  # Freeze scale parameter

    def top_k_masking(self, z, k=None):

        """ top-k activation masking. """
        return z.relu()
    
    def forward_encoder(self, x):
        if self.siamese_encoder:
            z = self.encoders[0](x)
            z = z.reshape(x.shape[0], x.shape[1], -1)
            return z
        zs = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(x[:, i, :])  # Apply each encoder to the corresponding head
            zs.append(z.unsqueeze(1))  # Unsqueeze to add head dimension
        z = torch.stack(zs, dim=1)  # Stack the outputs along the head dimension
        z = z.reshape(x.shape[0], x.shape[1], -1)
        return z
    
    def forward_basic_encoder(self, x):
        z = self.encoders[0](x)
        return z
    def forward(self, x, pre, post):
        N, L, nh, hc = pre.shape
        
        if self.transcode:
            # hack pre SAE hook to "reconstruct" the projected attention output
            post = self.hook_pre_sae(post)
            x_res = post
        else:
            pre = pre.reshape(N, L, nh * hc)
            self.hook_pre_sae(pre)
            x_res = pre
        batch_size = x.shape[0]
        token_dim = x.shape[1]
        if self.use_basic:
            if self.transcode:
                # z = self.forward_basic_encoder(post.reshape(batch_size * token_dim, nh * hc))
                z = self.forward_basic_encoder(pre.reshape(batch_size * token_dim, nh * hc))
            else:
                z = self.forward_basic_encoder(pre.reshape(batch_size * token_dim, nh * hc))
            z_bar = self.top_k_masking(z)
            z_bar = self.hook_hidden_post(z_bar.reshape(batch_size, token_dim, -1))
            x_recon = self.decoder(z_bar.reshape(batch_size * token_dim, -1))
            x_recon = self.hook_post_sae(x_recon.reshape(batch_size, token_dim, -1))
            if self.add_residual:
                x_recon = x_recon + (x_res - x_recon).detach()
            return x_recon, z, z_bar
        else:
            z = self.forward_encoder(pre.reshape(batch_size * token_dim, nh, hc))
            z = z.reshape(batch_size * token_dim, -1)
            z_bar = self.top_k_masking(z)
            z_bar = self.hook_hidden_post(z_bar.reshape(batch_size, token_dim, -1))
            x_recon = self.decoder(z_bar.reshape(batch_size * token_dim, -1))
            x_recon = self.hook_post_sae(x_recon.reshape(batch_size, token_dim, -1))
            
        if self.use_aux and self.dead_neurons is not None:
            z_aux = z + ~self.dead_neurons.unsqueeze(0) * -1e6 # mask out non-dead neurons
            z_aux = self.top_k_masking(z_aux, self.k * 4)
            x_aux = self.decoder(z_aux.reshape(batch_size * token_dim, -1))
            x_aux = x_aux.reshape(batch_size, token_dim, -1)
            x_aux = self.hook_aux_sae(x_aux)
        if self.add_residual:
            x_recon = x_recon + (x_res - x_recon).detach()
        return x_recon, z, z_bar

    @classmethod
    def from_config(cls, config: dict, model: nn.Module):
        """Instantiate TopKSAE from a config dictionary."""
        sae_keys = ["sae_input_dim", "sae_hidden_dim", "sae_k", "sae_layer_idx", "sae_token", "sae_ckpt_path",
                    "sae_n_heads", "sae_transcode", "sae_siamese_encoder", "sae_use_basic", 'sae_use_aux']
        sae_kwargs = {key.replace('sae_', ''): config[key] for key in sae_keys if key in config}
        sae_kwargs['input_dim'] = model.hidden_dim
        if config.get('sae_hidden_dim', 30000) < model.hidden_dim:
            sae_kwargs['hidden_dim'] = model.hidden_dim * config.get('sae_hidden_dim', None)
        print(sae_kwargs)
        sae = cls(**sae_kwargs)
        if "ckpt_path" in sae_kwargs:
            try:
                print(f"Loading SAE weights from {sae_kwargs['ckpt_path']}")
                sae.load_state_dict(torch.load(sae_kwargs["ckpt_path"], map_location='cpu')["state_dict"])
            except Exception as e:
                print(f"Error loading SAE checkpoint, using default weights.\n{e}")
        return sae


class JumpReLUFunction(torch.autograd.Function):
    """
    The JumpReLU activation function with a Rectangle Estimator for the backward pass.
    
    Forward:
        f(x, threshold) = x if x > threshold else 0
        
    Backward (w.r.t x):
        Pass-through if x > threshold, else 0.
        
    Backward (w.r.t threshold):
        Approximated using a Rectangle Estimator (Uniform PDF approximation of Dirac Delta).
        gradients flow if |x - threshold| < bandwidth / 2.
        The gradient magnitude is scaled by (1 / bandwidth) to effectively approximate the delta function.
    """
    @staticmethod
    def forward(ctx, x, threshold, bandwidth=0.1):
        # Create the hard mask
        mask = (x > threshold).float()
        
        # Save context for backward pass
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        
        # Apply mask
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # 1. Gradient with respect to Input (x)
        # Standard ReLU-like gradient: 1 if active, 0 if inactive
        grad_x = grad_output * (x > threshold).float()

        # 2. Gradient with respect to Threshold
        # Theoretical derivative: -grad_output * x * delta(x - threshold)
        # Rectangle Approximation: delta(z) ≈ (1/bandwidth) * rect(z/bandwidth)
        
        diff = x - threshold
        # Check if inside window [-bandwidth/2, bandwidth/2]
        inside_window = (diff.abs() < (bandwidth / 2.0)).float()
        
        # The negative sign comes from chain rule d/dt H(x-t) = -delta(x-t)
        # We multiply by (1.0 / bandwidth) to maintain the area of 1 (PDF property)
        grad_threshold_full = -(grad_output * x * inside_window * (1.0 / bandwidth))
        
        # Reduction logic:
        # The threshold is typically shape [Hidden], while input is [Batch, Hidden].
        if threshold.dim() > 0:
            dims_to_sum = list(range(grad_threshold_full.dim() - threshold.dim()))
            if dims_to_sum:
                grad_threshold = grad_threshold_full.sum(dim=dims_to_sum)
            else:
                grad_threshold = grad_threshold_full
        else:
            grad_threshold = grad_threshold_full.sum()

        # Return gradients for: x, threshold, bandwidth (None)
        return grad_x, grad_threshold, None
    
class JumpReLUSAE(nn.Module):
    def __init__(self, 
                 k=64,
                 input_dim=768, 
                 n_heads=12, 
                 hidden_dim=30000, 
                 layer_idx=-1, 
                 token="cls",
                 use_basic=False, 
                 transcode=False, 
                 norm_activations=False,
                 siamese_encoder=False,
                 use_aux=False,
                 init_threshold=1.0, # CHANGED: Default increased to 1.0 to prevent initial collapse
                 bandwidth=0.1,
                 *args, 
                 **kwargs):
        """
        JumpReLU Sparse Autoencoder (SAE).
        """
        super(JumpReLUSAE, self).__init__()
        print(f"JumpReLUSAE: input_dim={input_dim}, hidden_dim={hidden_dim}, layer_idx={layer_idx}")
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.token_type = token
        self.n_heads = n_heads
        self.use_basic = use_basic
        self.transcode = transcode
        self.norm_activations = norm_activations
        self.siamese_encoder = siamese_encoder
        self.use_aux = use_aux
        self.add_residual = True

        self.dead_neurons = None
        self.use_block_loss = False
        self.block_coeff = 0.1
        self.per_head_recon = True
        
        
        # JumpReLU specific parameters
        # We initialize with a log parameter to ensure positivity during training updates if optimizer allows,
        # though strict positivity isn't enforced by exp() alone if we update log_threshold directly.
        # However, exp() ensures the *value* used is always positive.
        self.log_threshold = nn.Parameter(torch.zeros(hidden_dim) + torch.log(torch.tensor(init_threshold)))
        self.bandwidth = bandwidth 

        # Hooks
        self.hook_pre_sae = nn.Identity()
        self.hook_hidden_post = nn.Identity()
        self.hook_post_sae = nn.Identity()
        self.hook_aux_sae = nn.Identity()

        # Decoder
        self.decoder = weight_norm(nn.Linear(hidden_dim, input_dim, bias=True), dim=1)
        self._fix_weight_norm()

        # Encoders
        if use_basic:
            self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim, hidden_dim)))])
        else:
            if siamese_encoder:
                print("Using siamese encoder for SAE.")
                self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim // n_heads, hidden_dim // n_heads))) for _ in range(1)])
            else:
                print("Using independent encoders for SAE.")
                self.encoders = nn.ModuleList([nn.Sequential(weight_norm(nn.Linear(input_dim // n_heads, hidden_dim // n_heads))) for _ in range(n_heads)])

        self.dead_neurons = None

    def _fix_weight_norm(self):
        """ Set the magnitude parameter (g) to 1 and freeze it. """
        for layer in [self.decoder]:
            layer.weight_g.data.fill_(1.0)
            layer.weight_g.requires_grad = False

    @property
    def threshold(self):
        """Return the actual threshold value."""
        return torch.exp(self.log_threshold)

    @torch.no_grad()
    def initialize_with_data(self, x, pre, post, target_l0=50):
        """
        Initialize thresholds based on actual data statistics.
        This is CRITICAL for JumpReLU to avoid "initialization collapse" where 
        everything starts active and the optimizer crushes weights to fix it.
        """
        print(f"Initializing thresholds with data (Target L0: {target_l0})...")
        # Run forward pass up to pre-activations
        _, z_pre, _ = self.forward(x, pre, post)
        
        # z_pre is [Batch*Seq, Hidden]
        z_pre = z_pre.detach()
        
        # Calculate the k-th largest value per batch element is computationally heavy for global stats.
        # Instead, we find the percentile that corresponds to target_l0 / hidden_dim.
        sparsity_level = target_l0 / self.hidden_dim
        percentile = 1.0 - sparsity_level
        
        # We want the threshold for each neuron such that it is active `sparsity_level` % of the time.
        # Sort along batch dimension
        kth_index = int(percentile * z_pre.shape[0])
        kth_values, _ = torch.kthvalue(z_pre, kth_index, dim=0)
        
        # Set log threshold. Clamp to avoid log(<=0).
        safe_values = torch.clamp(kth_values, min=1e-6)
        self.log_threshold.data = torch.log(safe_values)
        
        print(f"Thresholds initialized. Mean threshold: {safe_values.mean().item():.4f}")

    def jump_relu_activation(self, z):
        """ Applies JumpReLU activation using the custom autograd function. """
        return JumpReLUFunction.apply(z, self.threshold, self.bandwidth)

    def calculate_l0_norm(self, pre_activations):
        """
        Calculates the differentiable L0 norm approximation consistent with the Rectangle Estimator.
        
        Since the Backward pass assumes the derivative is a rectangle of width `bandwidth`,
        the Forward pass for the loss term must be the integral of that rectangle:
        a Piecewise Linear Ramp (Hard Sigmoid) from 0 to 1.
        
        L0 ≈ sum( HardSigmoid( (z - threshold) / bandwidth ) )
        """
        # 1. Normalize difference by bandwidth
        diff = (pre_activations - self.threshold) / self.bandwidth
        
        # 2. Apply Piecewise Linear Ramp (Hard Sigmoid)
        # Maps [-0.5, 0.5] to [0, 1]
        probs = torch.clamp(diff + 0.5, 0.0, 1.0)
        
        # 3. Sum probabilities to get expected L0
        return probs.sum()

    def forward_encoder(self, x):
        if self.siamese_encoder:
            z = self.encoders[0](x)
            z = z.reshape(x.shape[0], x.shape[1], -1)
            return z
            
        zs = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(x[:, i, :])
            zs.append(z.unsqueeze(1))
        z = torch.stack(zs, dim=1)
        z = z.reshape(x.shape[0], x.shape[1], -1)
        return z

    def forward_basic_encoder(self, x):
        z = self.encoders[0](x)
        return z

    def forward(self, x, pre, post):
        N, L, nh, hc = pre.shape

        if self.transcode:
            post = self.hook_pre_sae(post)
            x_res = post
        else:
            pre = pre.reshape(N, L, nh * hc)
            self.hook_pre_sae(pre)
            x_res = pre

        batch_size = x.shape[0]
        token_dim = x.shape[1]

        # 1. Encode (Get pre-activations)
        if self.use_basic:
            if self.transcode:
                z_pre = self.forward_basic_encoder(pre.reshape(batch_size * token_dim, nh * hc))
            else:
                z_pre = self.forward_basic_encoder(pre.reshape(batch_size * token_dim, nh * hc))
        else:
            z_pre = self.forward_encoder(pre.reshape(batch_size * token_dim, nh, hc))
            z_pre = z_pre.reshape(batch_size * token_dim, -1)

        # 2. Apply JumpReLU Activation
        z_bar = self.jump_relu_activation(z_pre)

        # Hooks & Reshaping
        _ = self.hook_hidden_post((z_pre.reshape(batch_size, token_dim, -1), z_bar.reshape(batch_size, token_dim, -1)))
        
        # 3. Decode
        x_recon = self.decoder(z_bar.reshape(batch_size * token_dim, -1))
        x_recon = self.hook_post_sae(x_recon.reshape(batch_size, token_dim, -1))

        # 4. Residual
        if self.add_residual:
            x_recon = x_recon + (x_res - x_recon).detach()

        # 5. Aux Loss
        if self.use_aux and self.dead_neurons is not None:
            dead_mask = self.dead_neurons.unsqueeze(0)
            z_aux = z_pre.relu() * dead_mask
            x_aux = self.decoder(z_aux.reshape(batch_size * token_dim, -1))
            x_aux = x_aux.reshape(batch_size, token_dim, -1)
            x_aux = self.hook_aux_sae(x_aux)

        # Return: Reconstruction, Pre-activations (for L0 calc), Post-activations
        return x_recon, z_pre, z_bar

    @classmethod
    def from_config(cls, config: dict, model: nn.Module):
        """Instantiate JumpReLUSAE from a config dictionary."""
        sae_keys = ["sae_input_dim", "sae_hidden_dim", "sae_layer_idx", "sae_token", "sae_ckpt_path", 
                    "sae_n_heads", "sae_transcode", "sae_siamese_encoder", "sae_use_basic", 'sae_use_aux',
                    "sae_init_threshold", "sae_bandwidth"]
        
        sae_kwargs = {key.replace('sae_', ''): config[key] for key in sae_keys if key in config}
        sae_kwargs['input_dim'] = model.hidden_dim
        
        if config.get('sae_hidden_dim', 30000) < model.hidden_dim:
            sae_kwargs['hidden_dim'] = model.hidden_dim * config.get('sae_hidden_dim', None)
            
        print(sae_kwargs)
        sae = cls(**sae_kwargs)
        
        if "ckpt_path" in sae_kwargs:
            try:
                print(f"Loading SAE weights from {sae_kwargs['ckpt_path']}")
                sae.load_state_dict(torch.load(sae_kwargs["ckpt_path"], map_location='cpu')["state_dict"])
            except Exception as e:
                print(f"Error loading SAE checkpoint, using default weights.\n{e}")
        
        return sae