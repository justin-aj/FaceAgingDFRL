# Minimal Face Aging Implementation with Stable Diffusion
# This version uses minimal dependencies to avoid compatibility issues

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
import shutil

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install it first.")

# Only import diffusers if torch is available
if TORCH_AVAILABLE:
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        DIFFUSERS_AVAILABLE = True
    except ImportError:
        DIFFUSERS_AVAILABLE = False
        print("Diffusers not available. Please install it with: pip install diffusers transformers")

# Define constants
DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = "./output"

# --- Utility Functions ---

def download_image(url, output_path=None):
    """Download an image from a URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return output_path
            else:
                return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to download image: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# --- Inference ---

def age_face(
    image_path,
    base_model_path=DEFAULT_BASE_MODEL,
    target_age="elderly",
    target_age_value=75,
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=7.5,
    output_path=None
):
    """Apply age transformation to a face using text-to-image prompting"""
    if not DIFFUSERS_AVAILABLE:
        print("Error: diffusers package not available. Please install it first.")
        return None
    
    # Load model
    print(f"Loading model {base_model_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load input image
    if isinstance(image_path, str):
        if image_path.startswith("http"):
            init_image = download_image(image_path)
        else:
            init_image = Image.open(image_path).convert("RGB")
    else:
        # Assume it's already a PIL image
        init_image = image_path
    
    # Resize image if needed
    width, height = init_image.size
    if width > 768 or height > 768:
        # Maintain aspect ratio
        if width > height:
            new_width = 768
            new_height = int(height * (768 / width))
        else:
            new_height = 768
            new_width = int(width * (768 / height))
        init_image = init_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create prompt for different ages
    age_descriptions = {
        "child": "a young child, about 5-8 years old",
        "teen": "a teenager, about 15-18 years old",
        "young_adult": "a young adult, about 25-30 years old",
        "middle_aged": "a middle-aged person, about 45-50 years old",
        "elderly": "an elderly person, about 70-80 years old",
    }
    
    # Get description or use target_age as is if not found
    age_desc = age_descriptions.get(target_age, f"a {target_age} person, {target_age_value} years old")
    
    # Craft prompt
    prompt = f"a highly detailed realistic photograph of {age_desc}, same person, same identity, detailed face, wrinkles, aging, high quality, detailed skin"
    
    # Add age-specific details
    if "elderly" in target_age or target_age_value >= 65:
        prompt += ", wrinkles, gray hair, aged skin"
    elif "middle" in target_age or 40 <= target_age_value < 65:
        prompt += ", slight wrinkles, mature face"
    elif "young" in target_age or 20 <= target_age_value < 40:
        prompt += ", youthful appearance"
    elif "teen" in target_age or 13 <= target_age_value < 20:
        prompt += ", teenage appearance, young face"
    elif "child" in target_age or target_age_value < 13:
        prompt += ", childlike features, young face, smooth skin"
    
    # Negative prompt to avoid distortion
    negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry"
    
    print(f"Generating {target_age} version with prompt: {prompt}")
    
    # Generate image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )
    
    aged_image = result.images[0]
    
    # Save output if specified
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        aged_image.save(output_path)
        print(f"Aged image saved to {output_path}")
    
    return aged_image

def generate_age_progression(
    image_path,
    base_model_path=DEFAULT_BASE_MODEL,
    output_dir="./age_progressions",
    age_steps=[("child", 7), ("young_adult", 25), ("middle_aged", 45), ("elderly", 75)],
    guidance_scale=7.5
):
    """Generate a series of age progressions for a face"""
    if not DIFFUSERS_AVAILABLE:
        print("Error: diffusers package not available. Please install it first.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    if isinstance(image_path, str):
        if image_path.startswith("http"):
            original_image = download_image(image_path)
        else:
            original_image = Image.open(image_path).convert("RGB")
    else:
        original_image = image_path
    
    # Save original image
    original_path = os.path.join(output_dir, "original.png")
    original_image.save(original_path)
    print(f"Original image saved to {original_path}")
    
    # Generate each age progression
    results = []
    for age_label, age_value in age_steps:
        output_path = os.path.join(output_dir, f"{age_label}_{age_value}.png")
        print(f"Generating {age_label} ({age_value} years) version...")
        
        aged_image = age_face(
            image_path=image_path,
            base_model_path=base_model_path,
            target_age=age_label,
            target_age_value=age_value,
            guidance_scale=guidance_scale,
            output_path=output_path
        )
        
        results.append((age_label, age_value, output_path))
    
    # Print summary
    print("\nAge progression complete! Results saved to:")
    print(f"Original: {original_path}")
    for age_label, age_value, path in results:
        print(f"{age_label} ({age_value} years): {path}")
    
    return results

# --- Setup ---

def setup_environment():
    """Set up the environment and install required packages"""
    try:
        import sys
        import subprocess
        
        # List of required packages
        requirements = [
            "torch",
            "numpy",
            "Pillow",
            "tqdm",
            "requests",
            "diffusers",
            "transformers",
            "accelerate"
        ]
        
        print("Installing required packages...")
        for package in requirements:
            try:
                __import__(package)
                print(f"✓ {package} already installed")
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("\nEnvironment setup complete!")
        return True
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return False

# --- Main Script ---

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Minimal Face Aging Implementation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up the environment")
    
    # Age face command
    age_parser = subparsers.add_parser("age", help="Age a face")
    age_parser.add_argument("--image", type=str, required=True, help="Input image path or URL")
    age_parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base model ID")
    age_parser.add_argument("--target_age", type=str, default="elderly", 
                           help="Target age label (child, teen, young_adult, middle_aged, elderly)")
    age_parser.add_argument("--target_age_value", type=int, default=75, help="Target age value")
    age_parser.add_argument("--output", type=str, default="aged_face.png", help="Output image path")
    
    # Age progression command
    progression_parser = subparsers.add_parser("progression", help="Generate age progression for a face")
    progression_parser.add_argument("--image", type=str, required=True, help="Input image path or URL")
    progression_parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base model ID")
    progression_parser.add_argument("--output_dir", type=str, default="./age_progressions", help="Output directory")
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_environment()
    
    elif args.command == "age":
        age_face(
            image_path=args.image,
            base_model_path=args.base_model,
            target_age=args.target_age,
            target_age_value=args.target_age_value,
            output_path=args.output
        )
    
    elif args.command == "progression":
        generate_age_progression(
            image_path=args.image,
            base_model_path=args.base_model,
            output_dir=args.output_dir
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()