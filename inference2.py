import os
import torch
from diffusers import DiffusionPipeline
from peft import PeftModel
import matplotlib.pyplot as plt

# Paths
LORA_WEIGHTS_DIR = "lora_adapter/lora_weights"
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get prompt from user input
prompt = input("Enter your prompt: ")

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

# Generate image
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
output_path = os.path.join(OUTPUT_DIR, "generated_image.png")
image.save(output_path)
print(f"Saved image to {output_path}")

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")  # Hide axes
plt.title("Generated Image")
plt.show()