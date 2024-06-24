import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from omegaconf import OmegaConf

from utils import instantiate_from_config
from dataset import make_style_image_dataloader

from diffusers.utils.logging import set_verbosity
from logging import ERROR

set_verbosity(ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--pretrained_state",
        type=str,
        default="",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    trainer_config = config.trainer
    model_config = config.model
    data_config = config.data

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=trainer_config.mixed_precision,
        log_with=trainer_config.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=trainer_config.gradient_accumulation_steps,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    print("Loading scheduler, tokenizer and models...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_config.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_config.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_config.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_config.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_config.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(model_config.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_config.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_config.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)

    print("Successfully loaded.")
    
    #ip-adapter
    style_crafter = instantiate_from_config(model_config.model)
    style_crafter.create_cross_attention_adapter(unet)
    if args.pretrained:
        style_crafter.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
        print("Loaded style crafter from", args.pretrained)
    style_crafter.unet = unet
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    optimizer = torch.optim.AdamW(style_crafter.get_trainable_parameters(), lr=trainer_config.learning_rate, weight_decay=trainer_config.weight_decay)
    print(f"num of parameters: {sum(p.numel() for p in style_crafter.parameters())}")
    
    # dataloader
    train_dataloader = make_style_image_dataloader(**data_config)
    
    print(accelerator.device)
    # Prepare everything with our `accelerator`.
    style_crafter, optimizer, train_dataloader = accelerator.prepare(style_crafter, optimizer, train_dataloader)
    print("Training model...")

    if args.pretrained_state != "":
        print("Loading pretrained state from", args.pretrained_state)
        accelerator.load_state(args.pretrained_state, strict=False)
        global_step = 20001
        print("Global step is", global_step)
    else:
        global_step = 0
    progress_bar = tqdm(
        range(global_step, trainer_config.max_train_steps),
        disable=not accelerator.is_main_process,
    )
    progress_bar.set_description("Steps")
    begin = time.perf_counter()
    while True:
        for step, batch in enumerate(train_dataloader):

            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(style_crafter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["jpg"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if trainer_config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += trainer_config.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    image_embeds = image_encoder(batch["style"].to(accelerator.device, dtype=weight_dtype), 
                                                 output_hidden_states=True).hidden_states[-2]
                # image_embeds_ = []
                # for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                #     if drop_image_embed == 1:
                #         image_embeds_.append(torch.zeros_like(image_embed))
                #     else:
                #         image_embeds_.append(image_embed)
                # image_embeds = torch.stack(image_embeds_)
            
                with torch.no_grad():
                    text_input_ids = tokenizer(batch['txt'], max_length=tokenizer.model_max_length, padding="max_length", 
                                               truncation=True, return_tensors='pt').input_ids

                    encoder_output = text_encoder(text_input_ids.to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]


                    text_input_ids_2 = tokenizer_2(batch['txt'], max_length=tokenizer_2.model_max_length, padding="max_length", 
                                                   truncation=True, return_tensors='pt').input_ids
                    encoder_output_2 = text_encoder_2(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                        
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]

                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = style_crafter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(data_config.batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    if global_step % trainer_config.log_steps == 0:
                        print("Step {}, data_time: {}, time: {}, step_loss: {}".format(
                            step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            
            try:
                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    if global_step % trainer_config.save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"adapter-{global_step}")
                        if accelerator.is_main_process:
                            os.makedirs(save_path, exist_ok=True)
                            # if multi-gpu, save the model in the format of single-gpu
                            try:
                                state_dict = style_crafter.module.state_dict()
                            except:
                                state_dict = style_crafter.state_dict()

                            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('unet')}

                            try:
                                torch.save(state_dict, save_path + "/adapter.ckpt")
                            except:
                                torch.save(state_dict, os.path.join(args.output_dir, "adapter-newest.ckpt"))

                        # save_dict = {
                        #     'image_context_model': style_crafter.module.image_context_model.state_dict(),
                        #     'scale_predictor': style_crafter.scale_predictor.state_dict(),
                        #     'kv_attn_layers': style_crafter.kv_attn_layers.state_dict(),
                        # }
                
                        

                if global_step % 10000 == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        try:
                            accelerator.save_state(save_path)
                        except:
                            accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-newest"))
            except Exception as e:
                print(e)
                print("Failed to save checkpoint!")
                continue


            if global_step >= trainer_config.max_train_steps:
                print("Training finished.")
                return

            
                
if __name__ == "__main__":
    main()    
