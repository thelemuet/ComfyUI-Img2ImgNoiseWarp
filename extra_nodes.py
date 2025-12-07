import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

try:
    import folder_paths
except ImportError:
    class folder_paths:
        @staticmethod
        def get_annotated_filepath(x):
            return x
        @staticmethod
        def get_output_directory():
            return "output"


class LoadImageWithNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "path/to/your/folder"}),
                "filename_prefix": ("STRING", {"default": "frame"}),
            }
        }

    CATEGORY = "NoiseWarp"
    RETURN_TYPES = ("IMAGE", "NOISE_STATE", "STRING", "INT")
    RETURN_NAMES = ("image", "seed_noise", "status", "frame_count")
    FUNCTION = "load_image_and_noise"

    def load_image_and_noise(self, folder_path, filename_prefix):
        folder_path = Path(folder_paths.get_annotated_filepath(folder_path))
        
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        # Get all PNG files with matching prefix in alphabetical order
        png_files = sorted([
            f for f in os.listdir(folder_path) 
            if f.startswith(filename_prefix + "_") and f.lower().endswith('.png')
        ])
        if not png_files:
            raise ValueError(f"No PNG files found with prefix '{filename_prefix}_' in folder: {folder_path}")
        
        # Get the last (latest) image
        latest_image_name = png_files[-1]
        image_path = folder_path / latest_image_name
        frame_count = len(png_files)
        
        # Load the image
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        
        # Convert to RGB (ignore alpha if present)
        if i.mode != 'RGB':
            i = i.convert('RGB')
        
        # Handle different bit depths
        original_mode = i.mode
        
        if original_mode in ['I', 'I;16', 'RGB16']:
            max_value = 2**16 - 1
        else:
            max_value = 255.0
        
        image = np.array(i).astype(np.float32)
        image = image / max_value
        image = torch.from_numpy(image)[None,]
        
        # Try to load matching noise file
        noise_filename = latest_image_name.rsplit('.', 1)[0] + '.npz'
        noise_path = folder_path / noise_filename
        
        seed_noise = None
        status = ""
        
        if noise_path.exists():
            try:
                loaded = np.load(noise_path)
                seed_noise = torch.from_numpy(loaded['noise_state'])
                status = f"Loaded: {latest_image_name} + {noise_filename}"
            except Exception as e:
                status = f"Loaded: {latest_image_name} (noise load failed: {str(e)})"
        else:
            status = f"Loaded: {latest_image_name} (no noise, using seed)"
        
        return (image, seed_noise, status, frame_count)
        
    @classmethod
    def IS_CHANGED(s, folder_path, filename_prefix):
        folder_path = Path(folder_paths.get_annotated_filepath(folder_path))
        
        # Get all PNG files with matching prefix
        png_files = sorted([
            f for f in os.listdir(folder_path) 
            if f.startswith(filename_prefix + "_") and f.lower().endswith('.png')
        ])
        
        if not png_files:
            return "No PNG files found"
        
        # Get the latest image
        latest_image_name = png_files[-1]
        image_path = folder_path / latest_image_name
        
        # Return a hash of the file to detect changes
        import hashlib
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        
        # Also include the file count so it triggers when new files are added
        return f"{m.digest().hex()}_{len(png_files)}"

    @classmethod
    def VALIDATE_INPUTS(s, folder_path, filename_prefix):
        folder_path = Path(folder_paths.get_annotated_filepath(folder_path))
        if not folder_path.exists() or not folder_path.is_dir():
            return f"Invalid folder path: {folder_path}"
        png_files = [
            f for f in os.listdir(folder_path) 
            if f.startswith(filename_prefix + "_") and f.lower().endswith('.png')
        ]
        if not png_files:
            return f"No PNG files found with prefix '{filename_prefix}_' in folder: {folder_path}"
        return True


class SaveImageWithNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "seed_noise": ("NOISE_STATE",),
                "filename_prefix": ("STRING", {"default": "frame"}),
                "subfolder": ("STRING", {"default": "animations/01/"}),
            }
        }

    CATEGORY = "NoiseWarp"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_image_and_noise"

    def save_image_and_noise(self, images, seed_noise, filename_prefix, subfolder):
        # Get output directory
        output_dir = Path(folder_paths.get_output_directory()) / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the highest numbered file with this prefix
        existing_files = sorted([
            f for f in os.listdir(output_dir) 
            if f.startswith(filename_prefix + "_") and f.lower().endswith('.png')
        ])
        
        if existing_files:
            # Extract number from last file
            last_file = existing_files[-1]
            try:
                # Parse: "prefix_00042.png" -> 42
                number_part = last_file[len(filename_prefix) + 1:].split('.')[0]
                counter = int(number_part) + 1
            except (ValueError, IndexError):
                counter = 0
        else:
            counter = 0
        
        results = []
        
        # Save each image in the batch
        for image in images:
            # Convert tensor to numpy
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Generate filename
            filename = f"{filename_prefix}_{counter:05d}.png"
            filepath = output_dir / filename
            
            # Save image
            img.save(filepath, compress_level=4)
            
            # Save noise state as compressed npz
            noise_filename = f"{filename_prefix}_{counter:05d}.npz"
            noise_filepath = output_dir / noise_filename
            np.savez_compressed(noise_filepath, noise_state=seed_noise.cpu().numpy())
            
            results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "output"
            })
            
            counter += 1
        
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "LoadImageWithNoise": LoadImageWithNoise,
    "SaveImageWithNoise": SaveImageWithNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithNoise": "Load Image With Noise",
    "SaveImageWithNoise": "Save Image With Noise",
}
