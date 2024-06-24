import glob
import os
import argparse
import glob
from PIL import Image
from omegaconf import OmegaConf
from logging import ERROR

import torch
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionXLPipeline
from diffusers.utils.logging import set_verbosity
set_verbosity(ERROR)

from utils import instantiate_from_config
from models.stylecrafter import StyleCrafterInference


def infer(args):
    ## Step 1: Load models from config
    config = OmegaConf.load(args.config)
    model_config = config.model

    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        model_config.pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    )
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_config.image_encoder_path)
    style_crafter = instantiate_from_config(model_config.model)
    style_crafter.create_cross_attention_adapter(sdxl_pipe.unet)
    style_crafter.load_state_dict(torch.load(config.pretrained, map_location="cpu"))
    print("Successfully loaded StyleCrafter-SDXL from", config.pretrained)

    sc_pipe = StyleCrafterInference(sd_pipe=sdxl_pipe, image_encoder=image_encoder, style_crafter=style_crafter, device='cuda')

    ## Step 2: Load style images and prompts
    image_postfix = ["jpg", "jpeg", "png"]
    style_images_path = [glob.glob(os.path.join(args.style_dir, f"*.{postfix}")) for postfix in image_postfix]
    style_images_path = sorted([item for sublist in style_images_path for item in sublist])

    with open(args.prompts_file, "r") as f:
        prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]
    print(f"Loaded {len(style_images_path)} style images and {len(prompts)} prompts")
    
    os.makedirs(args.save_dir, exist_ok=True)

    ## Step 3: Infer
    for style_image_path in style_images_path:
        style_image = Image.open(style_image_path).convert("RGB")
        for prompt_idx, prompt in enumerate(prompts):
            print(f"===Inferring with style image {style_image_path} and prompt: {prompt}")
            
            images = sc_pipe.generate(
                pil_image=style_image,
                prompt=prompt,
                num_samples=args.num_samples,
                num_inference_steps=config.steps,
                seed=args.seed,
                scale=args.scale,
                guidance_scale=7.5,
                style_guidance_scale=5.0,
                width=config.width,
                height=config.height
            )

            save_name = f"style_{os.path.basename(style_image_path).split('.')[0]}_prompt_{prompt_idx + 1}"
            for img_idx, image in enumerate(images):
                image.save(os.path.join(args.save_dir, f"{save_name}_{img_idx}.png"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/infer/style_crafter_sdxl.yaml")
    parser.add_argument("--style_dir", type=str, default="testing_data/input_style")
    parser.add_argument("--prompts_file", type=str, default="testing_data/prompts.txt")
    parser.add_argument("--save_dir", type=str, default="testing_data/output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()

    infer(args)