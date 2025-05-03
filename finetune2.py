import os
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image
from datasets import Dataset, Image
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from torchvision import transforms
from torch.amp import autocast

# Paths
SOURCE_IMAGES_DIR = "source_images"
OUTPUT_DIR = "lora_adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Prepare the dataset
def create_dataset(image_dir):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
    dataset_dict = {"image": image_paths}
    dataset = Dataset.from_dict(dataset_dict).cast_column("image", Image())
    return dataset

# Load images from source_images folder
dataset = create_dataset(SOURCE_IMAGES_DIR)

# Step 2: Load Stable Diffusion v1-5 base model and VAE
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")  # Move VAE to CUDA

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")  # Move entire pipeline to CUDA

# Enable gradient checkpointing to save memory
pipe.unet.enable_gradient_checkpointing()

# Step 3: Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank of LoRA
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Target cross-attention layers
    lora_dropout=0.05,
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_config).to("cuda")  # Ensure LoRA model is on CUDA

# Step 4: Preprocess images for training
def preprocess_images(examples):
    # Define transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to 512x512
        transforms.ToTensor(),  # Convert to tensor (0 to 1)
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] for Stable Diffusion
    ])

    # Load and preprocess images
    images = [load_image(img) for img in examples["image"]]
    images = [transform(img).to("cuda").half() for img in images]  # Convert to tensor, move to CUDA, use float16

    # Convert images to latents using the VAE
    with autocast("cuda"):
        latents = [pipe.vae.encode(img.unsqueeze(0)).latent_dist.sample().squeeze(0) * 0.18215 for img in images]  # Remove batch dim and scale
    examples["latents"] = latents  # Store list of tensors
    examples["prompt"] = ["sks person"] * len(images)  # Add text prompt
    return examples

dataset = dataset.map(preprocess_images, batched=True, batch_size=1)  # Process one image at a time

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=100,  # Adjust based on dataset size (3-4 images)
    per_device_train_batch_size=1,  # Small batch size due to GPU memory constraints
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    save_steps=500,
    logging_steps=10,
    max_steps=500,  # Adjust based on convergence
    fp16=True,
    remove_unused_columns=False,
    label_names=[],  # Suppress label_names warning
)

# Step 6: Define a custom data collator for diffusion training
def collate_fn(examples):
    # Stack latents (each is a tensor of shape [channels, height, width])
    latents = torch.stack([example["latents"] for example in examples])  # Shape: [batch_size, channels, height, width]
    prompts = [example["prompt"] for example in examples]
    return {"latents": latents, "prompt": prompts}

# Step 7: Custom Trainer to handle diffusion loss
class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        latents = inputs["latents"].to("cuda").half()  # Ensure latents are on CUDA and float16
        prompts = inputs["prompt"]

        # Encode text prompts
        with autocast("cuda"):
            text_inputs = self.pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            text_embeddings = self.pipe.text_encoder(text_inputs.input_ids)[0].half()

            # Add noise to latents (diffusion process)
            noise = torch.randn_like(latents, device="cuda")  # Create noise on CUDA
            timesteps = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (latents.shape[0],), device="cuda")
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute loss (MSE between predicted and actual noise)
            loss = F.mse_loss(noise_pred, noise)

        return loss

# Step 8: Fine-tune the model
trainer = DiffusionTrainer(
    model=pipe.unet,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)
trainer.pipe = pipe  # Attach pipe to trainer for access in compute_loss

# Start fine-tuning
trainer.train()

# Step 9: Save the LoRA adapter
pipe.unet.save_pretrained(os.path.join(OUTPUT_DIR, "lora_weights"))
print(f"LoRA adapter saved to {OUTPUT_DIR}/lora_weights")