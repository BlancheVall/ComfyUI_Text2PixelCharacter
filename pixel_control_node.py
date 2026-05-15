import os
import torch
import numpy as np
from PIL import Image

import folder_paths
import comfy.sd
import comfy.utils

from nodes import (
    CLIPTextEncode,
    EmptyLatentImage,
    VAEDecode,
    common_ksampler,
)


NODE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(NODE_DIR, "checkpoints")
LORA_DIR = os.path.join(NODE_DIR, "lora")
PALETTE_PATH = os.path.join(NODE_DIR, "palette.png")


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)


folder_paths.folder_names_and_paths["text2pixel_checkpoints"] = (
    [CHECKPOINT_DIR],
    folder_paths.supported_pt_extensions
)

folder_paths.folder_names_and_paths["text2pixel_loras"] = (
    [LORA_DIR],
    folder_paths.supported_pt_extensions
)


def list_model_files(folder):
    if not os.path.exists(folder):
        return []

    valid_ext = (".safetensors", ".ckpt", ".pt", ".pth")
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ]

    return sorted(files)


class Text2PixelCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = list_model_files(CHECKPOINT_DIR)
        loras = list_model_files(LORA_DIR)

        if not checkpoints:
            checkpoints = ["NO_CHECKPOINT_FOUND"]

        if not loras:
            loras = ["NO_LORA_FOUND"]

        return {
            "required": {
                "ckpt_name": (
                    checkpoints,
                    {
                        "default": checkpoints[0]
                    }
                ),

                "lora_name": (
                    loras,
                    {
                        "default": loras[0]
                    }
                ),

                "character_prompt": ("STRING", {
                    "multiline": True,
                    "default": "red haired female knight, blue dress, brown leather boots, small cape"
                }),

                "seed": ("INT", {
                    "default": 217,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Addy/Pixel Tools"

    def generate(self, ckpt_name, lora_name, character_prompt, seed):
        if ckpt_name == "NO_CHECKPOINT_FOUND":
            raise FileNotFoundError(
                f"No checkpoint found in: {CHECKPOINT_DIR}"
            )

        steps = 40
        cfg = 5.0
        sampler_name = "euler"
        scheduler = "karras"
        denoise = 1.0

        width = 256
        height = 256
        batch_size = 1

        lora_strength_model = 1.20
        lora_strength_clip = 0.94

        style_prompt = (
            "Pxstyle, chibi, big head, small body, full body, full body RPG character sprite, simple white background, pixel art, clean outline,"
        )

        negative_prompt = (
            "smooth painting, blurry, anti aliasing, soft shading, realistic, semi realistic, illustration, painterly, brush strokes, detailed background, gradient background,"
        )

        positive_prompt = f"{style_prompt} {character_prompt}"

        # Load checkpoint from this node's checkpoints folder
        ckpt_path = folder_paths.get_full_path(
            "text2pixel_checkpoints",
            ckpt_name
        )

        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )

        # Load LoRA from this node's lora folder
        if lora_name != "NO_LORA_FOUND":
            lora_path = folder_paths.get_full_path(
                "text2pixel_loras",
                lora_name
            )

            lora = comfy.utils.load_torch_file(
                lora_path,
                safe_load=True
            )

            model, clip = comfy.sd.load_lora_for_models(
                model,
                clip,
                lora,
                strength_model=lora_strength_model,
                strength_clip=lora_strength_clip
            )

        # Encode prompts
        clip_encoder = CLIPTextEncode()

        positive = clip_encoder.encode(
            clip=clip,
            text=positive_prompt
        )[0]

        negative = clip_encoder.encode(
            clip=clip,
            text=negative_prompt
        )[0]

        # Create 256x256 latent
        latent_node = EmptyLatentImage()
        latent = latent_node.generate(
            width=width,
            height=height,
            batch_size=batch_size
        )[0]

        # KSampler
        sampled_latent = common_ksampler(
            model,
            int(seed),
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise
        )[0]

        # VAE Decode
        vae_decoder = VAEDecode()
        decoded_image = vae_decoder.decode(
            vae=vae,
            samples=sampled_latent
        )[0]

        # Pixel post-processing
        final_image = self.pixel_process(decoded_image)

        return (final_image,)

    def pixel_process(self, image):
        results = []

        palette_img = None
        if os.path.exists(PALETTE_PATH):
            palette_img = Image.open(PALETTE_PATH).convert("RGB")

        for img in image:
            np_img = img.cpu().numpy()
            np_img = (np_img * 255).clip(0, 255).astype(np.uint8)

            pil = Image.fromarray(np_img).convert("RGB")

            # 1. Resize to 64x64
            small = pil.resize(
                (64, 64),
                Image.Resampling.NEAREST
            )

            # 2. Color match using palette image as reference
            if palette_img is not None:
                small = self.color_match_rgb(small, palette_img, strength=0.15)

            # 3. Quantize to 32 colors
            small = small.convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
                colors=32,
                dither=Image.Dither.NONE
            ).convert("RGB")

            # 4. Resize back to 512x512
            final = small.resize(
                (512, 512),
                Image.Resampling.NEAREST
            )

            arr = np.array(final).astype(np.float32) / 255.0
            results.append(torch.from_numpy(arr))

        return torch.stack(results)

    def color_match_rgb(self, image, reference, strength=0.15):
        """
        Simple RGB color match.
        Similar purpose to ColorMatchV2: shift image color distribution
        toward reference palette image without hard remapping every pixel.
        """

        img = np.array(image).astype(np.float32)
        ref = np.array(reference.resize(image.size)).astype(np.float32)

        img_mean = img.reshape(-1, 3).mean(axis=0)
        img_std = img.reshape(-1, 3).std(axis=0) + 1e-6

        ref_mean = ref.reshape(-1, 3).mean(axis=0)
        ref_std = ref.reshape(-1, 3).std(axis=0) + 1e-6

        matched = (img - img_mean) / img_std * ref_std + ref_mean

        # strength 0.15 means close to your ColorMatchV2 strength 0.15
        output = img * (1.0 - strength) + matched * strength

        output = np.clip(output, 0, 255).astype(np.uint8)

        return Image.fromarray(output)