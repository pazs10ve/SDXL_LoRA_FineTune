import os
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from torchvision import transforms
from torch.amp import autocast
from torch.utils.data import Dataset as TorchDataset

# Paths
SOURCE_IMAGES_DIR = "source_images"
OUTPUT_DIR = "lora_adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Create a custom PyTorch dataset
class StableDiffusionDataset(TorchDataset):
    def __init__(self, image_dir, vae):
        # Get all image paths
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                            if img.endswith((".png", ".jpg", ".jpeg"))]
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Store the VAE for encoding
        self.vae = vae
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = self.image_paths[idx]
        #print(f"Processing image: {os.path.basename(img_path)}")
        img = load_image(img_path)
        img_tensor = self.transform(img)
        
        # Move to cuda temporarily for encoding
        with torch.no_grad():
            with autocast(device_type='cuda'):
                img_tensor_cuda = img_tensor.to("cuda")
                img_batch = img_tensor_cuda.unsqueeze(0).half()
                latent = self.vae.encode(img_batch).latent_dist.sample() * 0.18215
                # Move back to CPU for DataLoader compatibility
                latent_cpu = latent.squeeze(0).detach().cpu()
        
        # Return the latent and the corresponding prompt
        return {
            "latents": latent_cpu,  # Must be on CPU for DataLoader
            "prompt": "sks person"
        }

# Load VAE first
print("Loading VAE model...")
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Create dataset with the VAE
print(f"Creating dataset from images in {SOURCE_IMAGES_DIR}...")
train_dataset = StableDiffusionDataset(SOURCE_IMAGES_DIR, vae)

# Now load the rest of the pipeline
print("Loading Stable Diffusion pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    vae=vae,  # Reuse the same VAE
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Enable gradient checkpointing
#print("Enabling gradient checkpointing...")
pipe.unet.enable_gradient_checkpointing()

# Configure LoRA
print("Configuring LoRA adaptation...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA config to UNet
pipe.unet = get_peft_model(pipe.unet, lora_config).to("cuda")

# Define collator function
def collate_fn(examples):
    #print("\nCollating batch:")
    latents = []
    prompts = []
    
    for i, example in enumerate(examples):
        #print(f"Example {i} - latents shape: {example['latents'].shape}")
        # Keep tensors on CPU here - they'll be moved to GPU during training
        latents.append(example['latents'])
        prompts.append(example['prompt'])
    
    # Stack latents but keep on CPU for pin_memory to work
    latents_tensor = torch.stack(latents)
    #print(f"Final batch shape: {latents_tensor.shape}, Device: {latents_tensor.device}")
    return {"latents": latents_tensor, "prompt": prompts}

# Define custom trainer
class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        latents = inputs["latents"].to("cuda").half()
        prompts = inputs["prompt"]

        # Encode text prompts
        with autocast(device_type='cuda'):
            text_inputs = pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0].half()

            # Add noise to latents
            noise = torch.randn_like(latents, device="cuda")
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device="cuda")
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
        
        return (loss, None) if return_outputs else loss

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    save_steps=50,
    logging_steps=10,
    max_steps=1000,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=True,  
)

# Initialize trainer
trainer = DiffusionTrainer(
    model=pipe.unet,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

# Start training
print("\nStarting training...")
trainer.train()

# Save the trained model
print("Training complete! Saving LoRA weights...")
pipe.unet.save_pretrained(os.path.join(OUTPUT_DIR, "lora_weights"))
print(f"LoRA adapter saved to {OUTPUT_DIR}/lora_weights")

