from typing import Callable
from models.clip_vit import get_clip_vit
from models.dino_vit import get_dino_vit
from models.dinov2_vit import get_dinov2_vit

MODELS = {
    # CLIP ViT B/32
    "clip_vit_b32_datacomp_m_s128m_b4k": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='datacomp_m_s128m_b4k', **kwargs),
    "clip_vit_b32_laion400m_e32": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='laion400m_e32', **kwargs),
    "clip_vit_b32_laion2b_s34b_b79k": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='laion2b_s34b_b79k', **kwargs),
    "clip_vit_b32_datacomp_xl_s13b_b90k": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='datacomp_xl_s13b_b90k', **kwargs),

    # CLIP ViT B/16
    "clip_vit_b16_datacomp_xl_s13b_b90k": lambda **kwargs: get_clip_vit('ViT-B-16', pretrained='datacomp_xl_s13b_b90k', **kwargs),

    # CLIP ViT L/14
    "clip_vit_l14_datacomp_xl_s13b_b90k": lambda **kwargs: get_clip_vit('ViT-L-14', pretrained='datacomp_xl_s13b_b90k', **kwargs),
    "clip_vit_l14_336_openai": lambda **kwargs: get_clip_vit('ViT-L-14-336', pretrained='openai', **kwargs),

    # CLIP ViT H/14
    "clip_vit_h14_dfn5b": lambda **kwargs: get_clip_vit('ViT-H-14-quickgelu', pretrained='dfn5b', **kwargs),

    # CLIP ViT Mobile-S2
    "clip_vit_mobiles2_datacompdr": lambda **kwargs: get_clip_vit('MobileCLIP-S2', pretrained='datacompdr', **kwargs),

    # DINO ViT
    "dino_vitb8": lambda **kwargs: get_dino_vit("dino_vitb8", **kwargs),
    "dino_vitb16": lambda **kwargs: get_dino_vit("dino_vitb16", **kwargs),
    "dino_vits8": lambda **kwargs: get_dino_vit("dino_vits8", **kwargs),
    "dino_vits16": lambda **kwargs: get_dino_vit("dino_vits16", **kwargs),
    "dinov2_vitb14": lambda **kwargs: get_dinov2_vit("dinov2_vitb14", **kwargs),
    "dinov2_vitl14": lambda **kwargs: get_dinov2_vit("dinov2_vitl14", **kwargs),
    'dinov2_vitb14_reg': lambda **kwargs: get_dinov2_vit("dinov2_vitb14_reg", **kwargs),
    'dinov2_vitl14_reg': lambda **kwargs: get_dinov2_vit("dinov2_vitl14_reg", **kwargs),
    "clip_vit_b16_laion2b_s34b_b88k": lambda **kwargs: get_clip_vit('ViT-B-16', pretrained='laion2b_s34b_b88k', **kwargs),
    "clip_vit_l14_laion2b_s32b_b82k": lambda **kwargs: get_clip_vit('ViT-L-14', pretrained='laion2b_s32b_b82k', **kwargs),
    "clip_vit_h14_laion2b_s32b_b79k": lambda **kwargs: get_clip_vit('ViT-H-14', pretrained='laion2b_s32b_b79k', **kwargs),
    "clip_vit_g14_laion2b_s34b_b88k": lambda **kwargs: get_clip_vit('ViT-g-14', pretrained='laion2b_s34b_b88k', **kwargs),
    "clip_vit_big_g14_laion2b_s39b_b160k": lambda **kwargs: get_clip_vit('ViT-bigG-14', pretrained='laion2b_s39b_b160k', **kwargs),
    ###
    ###

    "clip_vit_l14_laion400m_e31": lambda **kwargs: get_clip_vit('ViT-L-14', pretrained='laion400m_e31', **kwargs),
    "clip_vit_b32_laion2b_s34b_b79k": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='laion2b_s34b_b79k', **kwargs),
    
}


def get_fn_model_loader(model_name: str) -> Callable:
    if model_name in MODELS:
        def model_getter(*args, **kwargs):
            model = MODELS[model_name](*args, **kwargs)
            return model

        fn_model_loader = model_getter
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")
