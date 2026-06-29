import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import OneCycleLR, ChainedScheduler
from open_clip.model import VisionTransformer as CLIPVisionTransformer

from proj_utils.training_utils import get_loss, get_optimizer
from model_training.sae import ReLUSAE, TopKSAE, JumpReLUSAE

class LitClassifier(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()
        self.loss = None
        self.optim = None
        self.model = model
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return x

    def default_step(self, x, y, stage):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.default_step(x, y, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="test")

    def set_optimizer(self, optim_name, params, lr, weight_decay=0.0, norm_weight_decay=None):
        self.lr = lr
        self.optim = get_optimizer(optim_name, params, lr, weight_decay, norm_weight_decay, model=self.model)

    def set_loss(self, loss_name, weights=None):
        if loss_name == 'mse':
            def mse_loss(input, target):
                mse = torch.nn.functional.mse_loss(input, target, reduction='none')
                return mse.sum(dim=-1).mean()
            self.loss = mse_loss
        else:
            self.loss = get_loss(loss_name, weights)

    def configure_optimizers(self, milestones=None):
        milestones = self.config.get("milestones", milestones)
        if milestones is None:
            milestones = [5, 8]
        lr_scheduler = self.config.get("lr_scheduler", "custom")
        if lr_scheduler == "MultiStepLR":
            print(f"Using MultiStepLR with milestones: {milestones}")
            sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim,
                                                        milestones=milestones,
                                                        gamma=0.1)
        elif lr_scheduler == "CosineAnnealingLR":
            sche = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                                                                        T_0=2000,
                                                                        T_mult=1,)
            sche1 = OneCycleLR(self.optim, max_lr=1e-3, total_steps=4000)
            sche = ChainedScheduler([sche1, sche, sche1])
        elif lr_scheduler == "custom":
            warmup_epochs = self.config.get('warm_up_steps', 1000)
            total_epochs = 1000 + warmup_epochs
            cosine_epochs = total_epochs - warmup_epochs

            # Define the warm-up + cosine learning rate schedule
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / (warmup_epochs + 1)  # Linear warm-up
                else:
                    return 0.2 + 0.4 * (1 + torch.cos(
                        torch.tensor((epoch - warmup_epochs) / cosine_epochs * 3.1415926535)))  # Cosine decay

            # Create LR scheduler
            sche = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)


        else:
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
            "interval": "epoch",
        }

        return [self.optim], [scheduler]

    def state_dict(self, **kwargs):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict, assign)


class Vanilla(LitClassifier):
    def __init__(self, model, sae, config):
        super().__init__(model, config)
        self.sae = sae
        self.pre_activations = None
        self.post_activations = None
        self.hidden_activations = None
        self.aux_activations = None
        sae.hook_pre_sae.register_forward_hook(self.hook_pre_activations)
        sae.hook_post_sae.register_forward_hook(self.hook_post_activations)
        sae.hook_hidden_post.register_forward_hook(self.hook_hidden_activations)
        sae.hook_aux_sae.register_forward_hook(self.hook_aux_activations)

        self.activation_counts = torch.zeros(sae.hidden_dim)  # Track activations
        self.total_samples = 0  # Number of samples processed

        self.mask = None

    def hook_pre_activations(self, module, input, output):
        self.pre_activations = output.clone()

    def hook_post_activations(self, module, input, output):
        self.post_activations = output.clone()

    def hook_hidden_activations(self, module, input, output):
        if output.__class__ == tuple:
            self.hidden_activations = (output[0].clone(), output[1].clone())
        else:
            self.hidden_activations = output.clone()

    def hook_aux_activations(self, module, input, output):
        self.aux_activations = output.clone()
    
    def on_train_start(self) -> None:
        self.model.eval()
        if self.model.__class__ == CLIPVisionTransformer:
            for i, block in enumerate(self.model.transformer.resblocks):
                if i < self.sae.layer_idx:
                    for param in block.parameters():
                        param.requires_grad = False
        else:
            for i, block in enumerate(self.model.blocks):
                if i < self.sae.layer_idx:
                    for param in block.parameters():
                        param.requires_grad = False
            
    def on_train_epoch_start(self) -> None:
        self.model.eval()
        self.sae.train()
        self.mask = None
        self._get_off_block_diagonal_mask(self.sae.encoders[0][0].weight, n_blocks=self.sae.n_heads)
        for layer in [self.sae.decoder]:  # Access weight-normed layers
            layer.weight_g.requires_grad = False  # Freeze scale parameter

    def _get_off_block_diagonal_mask(self, weight, n_blocks):

        """Helper function to create the mask once."""
        if self.mask is not None:
            return self.mask
        d_out, d_in = weight.shape

        block_rows = d_out // n_blocks
        block_cols = d_in // n_blocks
        
        row_indices = torch.arange(d_out)
        col_indices = torch.arange(d_in)

        row_block_idx = row_indices // block_rows
        col_block_idx = col_indices // block_cols

        on_block_diagonal_mask = row_block_idx.unsqueeze(1) == col_block_idx.unsqueeze(0)
        
        # We return the inverse mask (mask for OFF-diagonal elements)
        self.mask = ~on_block_diagonal_mask.to(weight.device)

        return self.mask

    def cls_spa_mse_loss(self, act, rec):
        loss = torch.nn.functional.mse_loss(rec, act, reduction='none')  # (B, L, C)
        assert loss.isnan().sum() == 0, "NaN in loss computation"
        sum_loss = loss[:, 0, :].mean() + loss[:, 1:, :].mean()  # Scalar
        assert sum_loss.isnan().sum() == 0, f"NaN in summed loss : {act.shape}"
        return sum_loss
    
    def fvu(self, act, rec):
        variance = (act - rec).var(dim=0)
        total_variance = act.var(dim=0)

        return (variance / total_variance).mean().item()

    import torch

    def block_diagonal_regularization_loss(self, weight: torch.Tensor, n_blocks: int) -> torch.Tensor:
        """
        Calculates an L1 loss to encourage a weight matrix to be block-diagonal.

        Args:
            weight (torch.Tensor): The weight tensor of the linear layer.
                                For nn.Linear(d_in, d_out), this has shape (d_out, d_in).
            n_blocks (int): The number of diagonal blocks to enforce.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        mask = self._get_off_block_diagonal_mask(weight, n_blocks)
        # The loss is the L1 norm of the elements NOT on the block diagonal
        off_block_diagonal_weights = weight[mask]
        loss = off_block_diagonal_weights.abs().mean()
        return loss

    def default_step(self, x, y, stage):
        self(x)
        

        act = self.pre_activations
        rec = self.post_activations
        if self.sae.__class__ == JumpReLUSAE:
            z, z_masked = self.hidden_activations
        else:
            z_masked = self.hidden_activations
        if self.sae.per_head_recon:
            z_masked = z_masked.reshape(-1, z_masked.shape[-1])
        rec_aux = self.aux_activations if self.sae.use_aux else None
        if self.training:
            # loss = recon_loss = 100 * self.cls_spa_mse_loss(  torch.nn.functional.normalize(act, dim=-1),
            #                                             torch.nn.functional.normalize(rec, dim=-1), 
            #                                         )
            loss = recon_loss = None

        if len(act.shape) == 3:
            dim_0 = act.shape[0]
            dim_1 = act.shape[1]
            act = act.reshape(dim_0 * dim_1, -1)
            rec = rec.reshape(dim_0 * dim_1, -1)
            z_masked = z_masked.reshape(dim_0 * dim_1, -1)

        if not self.training or loss is None:
            if self.sae.__class__ == TopKSAE:
                if self.sae.per_head_recon:
                    loss = self.loss(rec[0], act[0]) + self.loss(rec[1:], act[1:])
                else:
                    loss = self.loss(rec, act)
                recon_loss = loss.item()
            
            if self.sae.__class__ == ReLUSAE:
                recon_loss = self.loss(rec, act)
                loss =  recon_loss + z_masked.abs().sum(-1).mean() * self.sae.k
            if self.sae.__class__ == JumpReLUSAE:
                recon_loss = self.loss(rec, act)
                loss =  recon_loss + self.sae.calculate_l0_norm(z).mean()  * self.sae.k
        
        if len(act.shape) == 4:
            rec = rec[0].detach()
            act = act[0].detach()
            
        self.activation_counts += (z_masked > 0).sum(dim=0).detach().cpu()
        self.total_samples += z_masked.size(0)

        dead_neurons = self.activation_counts / self.total_samples < (10 / 50000)
        if self.total_samples > 10000:
            self.activation_counts /= 2
            self.total_samples /= 2
        
        # loss = recon_loss = 100 * self.loss(torch.nn.functional.normalize(rec, dim=-1),
        #                        torch.nn.functional.normalize(act, dim=-1))
        
        cosine_loss = (1 - torch.nn.functional.cosine_similarity(act, rec).mean()).abs().item()
        aux_loss = None
        

        fraction_of_varience_unexplained = self.fvu(act, rec)

        self.sae.dead_neurons = dead_neurons.clone().to(act.device)
        log_dict = {}

        if self.sae.use_aux and rec_aux is not None:
            aux_loss = 3 * self.loss(torch.nn.functional.normalize(rec_aux, dim=-1),
                                     torch.nn.functional.normalize(act, dim=-1))
            loss += aux_loss
            log_dict[f"{stage}_aux_loss"] = aux_loss
        if self.sae.use_basic and self.sae.use_block_loss:
            block_loss = self.block_diagonal_regularization_loss(self.sae.encoders[0][0].weight, n_blocks=self.sae.n_heads)
            loss += self.sae.block_coeff * block_loss
            log_dict[f"{stage}_block_loss"] = block_loss
            with torch.no_grad():
                primary_weight_mean = self.sae.encoders[0][0].weight[~self.mask].abs().mean().item()
            log_dict[f"{stage}_primary_weight_mean"] = primary_weight_mean
            
        log_dict.update(
            {f"{stage}_loss": loss.item(),
             f"{stage}_recon": recon_loss,
             f"{stage}_cosine": cosine_loss,
             f"{stage}_fvu": fraction_of_varience_unexplained,
             f"{stage}_max_act": z_masked.amax(-1).mean().item(),
             f"{stage}_neg_act": z_masked.amin(-1).mean().item(),
             f"{stage}_dead_neurons": dead_neurons.sum().item()
             })
        if self.sae.__class__ == JumpReLUSAE:
            log_dict[f"{stage}_avg_threshold"] = self.sae.threshold.mean().item()
        log_dict[f"{stage}_L0"] = (z_masked.abs() > 1e-5).sum().item() / z_masked.size(0)
        
        self.log_dict(
            log_dict,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def state_dict(self, **kwargs):
        return self.sae.state_dict()

    def set_optimizer(self, optim_name, params, lr, weight_decay=0.0, norm_weight_decay=None):
        self.lr = lr
        self.optim = get_optimizer(optim_name, params, lr, weight_decay, norm_weight_decay, model=self.sae)
