import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
from peft import PeftModel

# Paths
LORA_WEIGHTS_DIR = "lora_adapter/lora_weights"
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define prompts
prompts = [
    "sks person in a spacesuit, watercolor style, soft colors, detailed background",
    "sks person riding a horse, watercolor style, vibrant colors, dynamic pose",
    "sks person playing cricket, watercolor style, sunny day, action shot"
]

# Load the base Stable Diffusion pipeline
print("Loading Stable Diffusion pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Load the LoRA weights
print("Loading LoRA weights...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_DIR).to("cuda")

# Ensure the pipeline is in evaluation mode
pipe.unet.eval()

# Generate images for each prompt
generated_images = []
for i, prompt in enumerate(prompts):
    print(f"Generating image for prompt: {prompt}")
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]
    
    # Save the image
    output_path = os.path.join(OUTPUT_DIR, f"generated_image_{i+1}.png")
    image.save(output_path)
    print(f"Saved image to {output_path}")
    generated_images.append(image)

# Create and save a grid of all generated images
print("Creating image grid...")
grid = make_image_grid(generated_images, rows=1, cols=3)
grid.save(os.path.join(OUTPUT_DIR, "generated_images_grid.png"))
print(f"Saved image grid to {OUTPUT_DIR}/generated_images_grid.png")