# SDXL-LoRA-FineTune

A repository for fine-tuning Stable Diffusion models using LoRA (Low-Rank Adaptation) to create personalized image generators.

## Overview

This project allows you to fine-tune the Stable Diffusion v1-5 model using your own images to create a personalized image generator. The implementation uses LoRA, a parameter-efficient fine-tuning technique that significantly reduces the computational resources required for training while maintaining high-quality results.

## Features

- Fine-tune Stable Diffusion v1-5 with your own images
- Parameter-efficient training using LoRA adaptation
- Support for both batch training and interactive image generation
- Memory-efficient implementation with gradient checkpointing
- Easy-to-use inference scripts for generating new images

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU with at least 8GB VRAM
- Required Python packages (see below)

```
torch>=2.0.0
diffusers>=0.21.0
transformers>=4.30.0
peft>=0.4.0
datasets>=2.14.0
torchvision>=0.15.0
matplotlib>=3.7.0
```

## Project Structure

- `source_images/` - Directory for your training images
- `lora_adapter/` - Output directory for the trained LoRA weights
- `generated_images/` - Directory where generated images are saved
- `finetune.py` - Original fine-tuning script using PyTorch Dataset
- `finetune2.py` - Simplified fine-tuning script using Hugging Face Dataset
- `inference.py` - Batch inference script for generating multiple images
- `inference2.py` - Interactive inference script with user prompt input

## How to Use

### 1. Prepare Your Images

Place your training images in the `source_images/` directory. For best results:
- Use 3-10 high-quality images of the same subject/style
- Images should be clear and well-lit
- Consistent style/subject across images improves results

### 2. Fine-tune the Model

Run one of the fine-tuning scripts:

```bash
# Using the original implementation
python finetune.py

# OR using the simplified implementation
python finetune2.py
```

The training process will:
1. Load your images and preprocess them
2. Set up the Stable Diffusion model with LoRA configuration
3. Train for the specified number of steps/epochs
4. Save the LoRA weights to the `lora_adapter/lora_weights` directory

### 3. Generate Images

After training, you can generate images using one of the inference scripts:

```bash
# Generate multiple images with predefined prompts
python inference.py

# OR generate a single image with a custom prompt
python inference2.py
```

The generated images will be saved to the `generated_images/` directory.

## Training Parameters

You can adjust the following parameters in the fine-tuning scripts:

- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for the optimizer
- `max_steps`: Maximum number of training steps
- LoRA configuration:
  - `r`: Rank of the LoRA adaptation
  - `lora_alpha`: Scaling factor for LoRA
  - `target_modules`: Which modules to apply LoRA to

## Inference Parameters

You can adjust the following parameters in the inference scripts:

- `num_inference_steps`: Number of denoising steps (higher = better quality, slower)
- `guidance_scale`: How closely to follow the prompt (higher = more faithful but less creative)
- `height` and `width`: Output image dimensions

## Tips for Best Results

1. Use a consistent prompt format during inference: "sks person [description]"
2. Train for at least 500-1000 steps for good results
3. Experiment with different guidance scales during inference (5-9)
4. Use detailed prompts for better control over the generated images

## License

This project is provided for educational and research purposes only.

## Acknowledgments

This project utilizes the following open-source libraries:
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [PEFT](https://github.com/huggingface/peft) by Hugging Face
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face
