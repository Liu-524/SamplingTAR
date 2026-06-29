import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
import time 
import logging
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import random
import os

def check_latest_ckpt(ckpt_root, only_last=False):
    if not os.path.exists(ckpt_root):
        return None
    ckpts = [f for f in os.listdir(ckpt_root) if f.endswith('.ckpt') if not only_last or 'last' in f]
    if len(ckpts) == 0:
        return None
    ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join(ckpt_root, x)), reverse=True)
    logging.info(f'Found checkpoints: {ckpts}. Last modified: {time.ctime(os.path.getmtime(os.path.join(ckpt_root, ckpts[0])))}')
    return ckpts[0]


def get_ckpt_path(config, config_name, suffix='',base_dir='checkpoints', ckpt=None, layer=11):
    model_name = config['model_name']
    dir_name = f'{model_name}_{config_name+suffix}-{layer}'
    ckpt_root = os.path.join(base_dir, dir_name)
    assert os.path.exists(ckpt_root), f'Checkpoint root {ckpt_root} does not exist.'

    latest_ckpt = check_latest_ckpt(ckpt_root, only_last=True)
    
        
    assert latest_ckpt is not None, f'No checkpoint found in {ckpt_root}.'

    if ckpt is not None:
        assert os.path.exists(os.path.join(ckpt_root, ckpt)), f'Checkpoint {ckpt} does not exist in {ckpt_root}.'
        if ckpt != latest_ckpt:
            logging.warning(f'Using specified checkpoint {ckpt} instead of latest {latest_ckpt}.')
    else:
        ckpt = latest_ckpt
    return os.path.join(ckpt_root, ckpt)

def get_ckpt_path_typo(config, config_name, suffix='',base_dir='checkpoints', ckpt=None, layer=11):
    model_name = config['model_name']
    dir_name = f'{model_name}_{config_name}_layer{layer}{suffix}'
    ckpt_root = os.path.join(base_dir, dir_name)
    assert os.path.exists(ckpt_root), f'Checkpoint root {ckpt_root} does not exist.'

    latest_ckpt = check_latest_ckpt(ckpt_root, only_last=True)
    
        
    assert latest_ckpt is not None, f'No checkpoint found in {ckpt_root}.'

    if ckpt is not None:
        assert os.path.exists(os.path.join(ckpt_root, ckpt)), f'Checkpoint {ckpt} does not exist in {ckpt_root}.'
        if ckpt != latest_ckpt:
            logging.warning(f'Using specified checkpoint {ckpt} instead of latest {latest_ckpt}.')
    else:
        ckpt = latest_ckpt
    return os.path.join(ckpt_root, ckpt)

def interpolate_positional_embedding(pos_emb, old_grid_size, new_grid_size):
    """
    Interpolate the positional embeddings to accommodate for a different image size.
    Copied from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """
    # pos_emb = model.pos_embed  # (1, old_num_patches+1, dim)
    # old_grid_size = int((pos_emb.shape[1] - 1) ** 0.5)
    # new_grid_size = int((new_img_size // patch_size))
    if old_grid_size == new_grid_size:
        return pos_emb

    class_token, pos_tokens = pos_emb[:1], pos_emb[1:]
    dim = pos_tokens.shape[-1]
    pos_tokens = pos_tokens.reshape(1, old_grid_size, old_grid_size, dim).permute(0, 3, 1, 2)

    pos_tokens = F.interpolate(
        pos_tokens, size=(new_grid_size, new_grid_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, dim)
    new_pos_emb = torch.cat((class_token.unsqueeze(0), pos_tokens), dim=1)
    return new_pos_emb[0]

def resize_model(model, new_img_size=224):
    # resize positional embedding
    preprocess = model.preprocess # torchvision.transforms.Compose
    preprocess.transforms[0].size = new_img_size
    preprocess.transforms[1].size = (new_img_size, new_img_size)

    patch_size = model.patch_size if isinstance(model.patch_size, int) else model.patch_size[0] 

    if hasattr(model, "positional_embedding"):
        pos_emb = model.positional_embedding
        old_grid_size = int((pos_emb.shape[0] - 1) ** 0.5)
        new_grid_size = int(new_img_size / patch_size)
        if old_grid_size == new_grid_size:
            return model
        print(f"Resizing positional embedding from {old_grid_size} to {new_grid_size}")
        new_pos_emb = interpolate_positional_embedding(pos_emb, old_grid_size, new_grid_size)
        model.positional_embedding = nn.Parameter(new_pos_emb)
    elif hasattr(model, "pos_embed"):
        pos_emb = model.pos_embed
        old_grid_size = int((pos_emb.shape[1] - 1) ** 0.5)
        new_grid_size = int(new_img_size / patch_size)
        if old_grid_size == new_grid_size:
            return model
        new_pos_emb = interpolate_positional_embedding(pos_emb, old_grid_size, new_grid_size)
        model.pos_embed = nn.Parameter(new_pos_emb)
    else:
        print("No positional embedding found.")
    return model
# features = np.array(features)  # shape: (num_images, num_patches, embedding_dim)

def get_wanted_size(feature_image_size, feature_patch_size, model_patch_size):
    # feature_image_size: size of image used to extract features
    # feature_patch_size: patch size used in feature extraction model
    # model_patch_size: patch size used in target model
    print(f"Feature image size: {feature_image_size}, feature patch size: {feature_patch_size}, model patch size: {model_patch_size}")
    scale =  model_patch_size / feature_patch_size
    wanted_size = feature_image_size * scale
    print(f"Wanted size: {wanted_size}, scale: {scale}")
    assert wanted_size % model_patch_size == 0, "Wanted size must be multiple of model patch size"
    assert wanted_size.is_integer(), "Wanted size must be integer"
    return int(wanted_size)

def get_parser():
    parser = ArgumentParser(
        description="Train models.",
    )
    parser.add_argument(
        "--config_file",
        default="configs/train_sae/imagenet/local/clip_vit_b32_datacomp_xl_s13b_b90k_imagenet_topk-spatial-64-30000_lr_0.0001.yaml",
    )
    return parser

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager
import random
import os

# Mapping for visualization clarity
LOCATION_MAP = {0: "Top", 1: "Bottom", 2: "Left", 3: "Right"}

class RandomTextBorderTransform:
    def __init__(self, texts, target_size=(224, 224), border_ratio=0.15, post_transform=None):
        self.texts = texts
        self.target_size = target_size
        self.border_ratio = border_ratio
        self.post_transform = post_transform
        
        self.font_paths = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        if not self.font_paths:
            self.font_paths = [] # Fallback

    def _get_random_color(self):
        return (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))

    def _get_fitted_font(self, text, box_h):
        """Finds a font size that fits the HEIGHT of the border."""
        if not self.font_paths: return ImageFont.load_default()
        
        font_path = random.choice(self.font_paths)
        # Target 70% fill to leave nice whitespace
        target_h = int(box_h * 0.70) 
        fontsize = max(8, target_h)
        font = ImageFont.truetype(font_path, fontsize)
        
        # Shrink if too tall
        # We check "Aj" because it covers both ascenders and descenders
        for _ in range(10):
            dummy = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            if hasattr(dummy, 'textbbox'):
                bbox = dummy.textbbox((0, 0), "Aj", font=font)
                h = bbox[3] - bbox[1]
            else:
                _, h = dummy.textsize("Aj", font=font)
            
            if h <= target_h: return font
            
            fontsize = int(fontsize * 0.9)
            if fontsize < 8: break
            try: font = ImageFont.truetype(font_path, fontsize)
            except: return ImageFont.load_default()

        return font

    def _draw_text_filled(self, base_canvas, text, color, box_w, box_h, rotation=0, center_coords=None):
        # 1. Determine dimensions relative to text direction
        if rotation in [90, -90]:
            short_dim = box_w  # The "height" of the text line
            long_dim = box_h   # The length to fill
        else:
            short_dim = box_h
            long_dim = box_w

        # 2. Get Font
        font = self._get_fitted_font(text, short_dim)

        # 3. Measure one word PRECISELY
        dummy = ImageDraw.Draw(base_canvas)
        if hasattr(dummy, 'textbbox'):
            # "anchor='lt'" ensures we measure from top-left logic
            bbox = dummy.textbbox((0, 0), text + "  ", font=font, anchor='lt')
            word_w = bbox[2] - bbox[0]
            word_h = bbox[3] - bbox[1]
        else:
            word_w, word_h = dummy.textsize(text + "  ", font=font)

        if word_w == 0: word_w = 1

        # 4. Create Repeating String
        repeats = int(long_dim / word_w) + 3
        full_string = (text + "  ") * repeats

        # 5. Create Layer TIGHT to the text height
        # FIX: Removed the large +50 buffer that was causing the offset
        layer_w = word_w * repeats
        layer_h = word_h 
        
        layer = Image.new('RGBA', (layer_w, layer_h), (255 - color[0], 255 - color[1], 255 - color[2], 0))
        d = ImageDraw.Draw(layer)
        
        # Draw text at (0,0) of this tight layer
        d.text((0, 0), full_string, fill=color, font=font, anchor='lt')

        # 6. Random Offset (Marquee shift)
        offset_x = random.randint(0, word_w)
        
        # Crop exactly the length we need
        crop_box = (offset_x, 0, offset_x + long_dim, layer_h)
        final_strip = layer.crop(crop_box)

        # 7. Rotate
        if rotation != 0: 
            # expand=True changes dimensions, so we must rely on center point pasting
            final_strip = final_strip.rotate(rotation, expand=True, resample=Image.BICUBIC)
        
        # 8. Paste Centered
        fw, fh = final_strip.size
        cx, cy = center_coords
        
        # Math to find top-left corner based on center (cx, cy)
        paste_x = int(cx - (fw / 2))
        paste_y = int(cy - (fh / 2))
        
        base_canvas.paste(final_strip, (paste_x, paste_y), mask=final_strip)

    def __call__(self, input_data):
        # --- Handle Tuple Input ---
        if isinstance(input_data, (tuple, list)):
            img = input_data[0]
        else:
            img = input_data

        if not isinstance(img, Image.Image):
             if isinstance(img, torch.Tensor): img = transforms.ToPILImage()(img)
        
        # --- Geometry Logic ---
        target_w, target_h = self.target_size
        edge_idx = random.randint(0, 3) 
        
        border_px_h = int(target_h * self.border_ratio)
        border_px_w = int(target_w * self.border_ratio)
        
        canvas = Image.new('RGB', (target_w, target_h), (255, 255, 255))
        
        if edge_idx == 0: # Top
            img_resized = img.resize((target_w, target_h - border_px_h), Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (0, border_px_h))
            center = (target_w // 2, border_px_h // 2)
            box_w, box_h = target_w, border_px_h
            rot = 0
            
        elif edge_idx == 1: # Bottom
            img_resized = img.resize((target_w, target_h - border_px_h), Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (0, 0))
            center = (target_w // 2, target_h - (border_px_h // 2))
            box_w, box_h = target_w, border_px_h
            rot = 0
            
        elif edge_idx == 2: # Left
            img_resized = img.resize((target_w - border_px_w, target_h), Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (border_px_w, 0))
            center = (border_px_w // 2, target_h // 2)
            # Note: For side borders, box_w is the skinny dimension (the border width)
            box_w, box_h = border_px_w, target_h
            rot = 90
            
        elif edge_idx == 3: # Right
            img_resized = img.resize((target_w - border_px_w, target_h), Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (0, 0))
            center = (target_w - (border_px_w // 2), target_h // 2)
            box_w, box_h = border_px_w, target_h
            rot = -90

        # --- Draw Text ---
        self._draw_text_filled(
            canvas, 
            random.choice(self.texts), 
            self._get_random_color(), 
            box_w, box_h, rot, center
        )

        # --- Post Transforms ---
        if self.post_transform:
            final_img = self.post_transform(canvas)
        else:
            final_img = canvas

        return final_img, edge_idx
    


import random
import matplotlib.font_manager
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from torchvision import transforms
class RandomStrokedTextOverlay:
    def __init__(self, texts, scale_range=(0.15, 0.30), stroke_width_ratio=0.08, 
                 font_path=None, post_transform=None):
        """
        Args:
            texts (list): List of strings to choose from.
            scale_range (tuple): Min and Max fraction of image height for font size.
                                 (0.15 = 15% of image height).
            stroke_width_ratio (float): Thickness of outline relative to font size.
            font_path (str, optional): Specific font path. If None, searches system.
            post_transform (callable, optional): PyTorch transforms to run after this.
        """
        self.texts = texts
        self.scale_range = scale_range
        self.stroke_width_ratio = stroke_width_ratio
        self.post_transform = post_transform
        
        # Font loading logic: Pre-fetch system fonts to avoid lag during augmentation
        try:
            self.system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        except:
            self.system_fonts = []
            
        self.specific_font = font_path

    def _get_random_color(self):
        """Returns a random RGB color."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _get_contrasting_stroke(self, color):
        """Returns black or white stroke depending on brightness of fill color."""
        # Calculate luminance
        luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        return (0, 0, 0) if luminance > 128 else (255, 255, 255)

    def _load_font(self, target_pixel_height):
        """Loads a font that roughly matches the target pixel height."""
        # 1. Choose a font path
        if self.specific_font:
            fp = self.specific_font
        elif self.system_fonts:
            fp = random.choice(self.system_fonts)
        else:
            fp = None 

        # 2. Load font object
        if fp:
            try:
                return ImageFont.truetype(fp, int(target_pixel_height))
            except Exception:
                pass 
        
        # Fallback for environments without system fonts
        try:
            return ImageFont.load_default(size=int(target_pixel_height))
        except TypeError:
            return ImageFont.load_default()

    def __call__(self, input_data):
        # 1. Handle Input (Unwrap tuples if necessary, common in datasets)
        if isinstance(input_data, (tuple, list)):
            img = input_data[0]
        else:
            img = input_data

        # Ensure we are working with a PIL Image
        is_tensor = False
        try:
            import torch
            from torchvision import transforms
            if isinstance(img, torch.Tensor):
                is_tensor = True
                img = transforms.ToPILImage()(img)
        except ImportError:
            pass
        
        # Work on a copy
        img = img.copy()


        # FIX: Ensure image is RGB. 
        # Grayscale images (mode 'L') will crash if we try to draw RGB text on them.
        if img.mode != 'RGB':
            img = img.convert('RGB')

        draw = ImageDraw.Draw(img)
        W, H = img.size

        # 2. Determine Size & Text
        text = random.choice(self.texts)
        
        # Initial height based on config
        target_h = int(H * random.uniform(self.scale_range[0], self.scale_range[1]))
        target_h = max(12, target_h)
        font = self._load_font(target_h)

        # 3. Measure and Downscale if needed
        # Helper logic to measure text
        def get_text_size(f, t):
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), t, font=f)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                return draw.textsize(t, font=f)

        text_w, text_h = get_text_size(font, text)

        # If text is wider than image, shrink it
        if text_w > W:
            # Scale down to fit with a small margin (e.g. 95% of width)
            scale_factor = (W * 0.95) / text_w
            target_h = int(target_h * scale_factor)
            target_h = max(10, target_h) # Minimum legible size
            
            # Reload font and re-measure
            font = self._load_font(target_h)
            text_w, text_h = get_text_size(font, text)

        # 4. Determine Location (Strictly Inside)
        max_x = max(0, W - text_w)
        max_y = max(0, H - text_h)
        
        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)

        # 5. Colors & Stroke
        text_color = self._get_random_color()
        stroke_color = self._get_contrasting_stroke(text_color)
        stroke_width = max(1, int(target_h * self.stroke_width_ratio))

        # 6. Draw
        draw.text(
            (start_x, start_y), 
            text, 
            font=font, 
            fill=text_color, 
            stroke_width=stroke_width, 
            stroke_fill=stroke_color
        )

        # 7. Post Transforms (e.g. back to Tensor)
        if self.post_transform:
            return self.post_transform(img)
            
        if is_tensor:
            return transforms.ToTensor()(img)
            
        return img