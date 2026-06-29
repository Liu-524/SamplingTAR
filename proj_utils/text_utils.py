import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

import json
import urllib.request

def get_imnet_wnid_to_name():
    """Returns a dictionary mapping ImageNet WNIDs to human-readable class names."""
    
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    urllib.request.urlretrieve(url, "imagenet_class_index.json")

    # Load the JSON
    with open("imagenet_class_index.json", "r") as f:
        imagenet_class_index = json.load(f)

    # The JSON format is { "0": ["n01440764", "tench"], ... }
    # We need a cleaner dictionary: { "n01440764": "tench" }
    wnid_to_name = {v[0]: v[1] for v in imagenet_class_index.values()}
    return wnid_to_name

def _get_random_color():
    """Returns a random RGB color."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def _get_contrasting_stroke(color):
    """Returns black or white stroke depending on brightness of fill color."""
    # Calculate luminance
    luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

def _load_font(target_pixel_height, font_path=None, system_fonts=None):
    """Loads a font that roughly matches the target pixel height."""
    fp = None
    if font_path:
        fp = font_path
    elif system_fonts and len(system_fonts) > 0:
        fp = random.choice(system_fonts)

    if fp:
        try:
            return ImageFont.truetype(fp, int(target_pixel_height))
        except Exception:
            pass 
    
    # Fallback
    try:
        return ImageFont.load_default(size=int(target_pixel_height))
    except TypeError:
        return ImageFont.load_default()
def write_on_image(img, text, scale_range=(0.15, 0.30), stroke_width_ratio=0.08, font_path=None, system_fonts=None):
    """
    Standalone function to write text onto a PIL Image.
    
    Args:
        img (PIL.Image): The input image.
        text (str): The text to write.
        scale_range (tuple): Min/Max height of text as fraction of image height.
        stroke_width_ratio (float): Stroke width relative to font size.
        font_path (str): Path to specific font.
        system_fonts (list): List of available system font paths (optimization).
    """
    # Work on a copy to avoid side effects
    img = img.copy()

    # Ensure image is RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # --- Center Crop to Square ---
    W, H = img.size
    min_dim = min(W, H)
    left = (W - min_dim) // 2
    top = (H - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))
    
    # Update dimensions after crop
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # Size Logic
    target_h = int(H * random.uniform(scale_range[0], scale_range[1]))
    target_h = max(12, target_h)
    
    # Load Font (pass system_fonts if provided to avoid re-scanning)
    if system_fonts is None:
        try:
            system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        except:
            system_fonts = []
            
    font = _load_font(target_h, font_path, system_fonts)

    # Helper logic to measure text
    def get_text_size(f, t):
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), t, font=f)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            return draw.textsize(t, font=f)

    text_w, text_h = get_text_size(font, text)

    # Shrink if too wide
    if text_w > W:
        scale_factor = (W * 0.95) / text_w
        target_h = int(target_h * scale_factor)
        target_h = max(10, target_h) 
        font = _load_font(target_h, font_path, system_fonts)
        text_w, text_h = get_text_size(font, text)

    # Location (Strictly Inside)
    max_x = max(0, W - text_w)
    max_y = max(0, H - text_h)
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)

    # Colors
    text_color = _get_random_color()
    stroke_color = _get_contrasting_stroke(text_color)
    stroke_width = max(1, int(target_h * stroke_width_ratio))

    # Draw
    draw.text(
        (start_x, start_y), 
        text, 
        font=font, 
        fill=text_color, 
        stroke_width=stroke_width, 
        stroke_fill=stroke_color
    )
    
    return img