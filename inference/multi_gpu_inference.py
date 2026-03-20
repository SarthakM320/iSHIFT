"""
Single-GPU iSHIFT Inference Script
Processes a chunk of test_general.json based on GPU ID.

Usage:
    CUDA_VISIBLE_DEVICES=0 python multi_gpu_inference.py --gpu_id 0 --num_gpus 8
"""

import os
import sys
import json
import torch
import argparse
from PIL import Image
from typing import Dict
from tqdm import tqdm

# Model imports
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor
from transformers.models.qwen2_vl_ishift.modeling_qwen2_vl_ishift import (
    iSHIFT_Qwen2VLForConditionalGeneration,
    iSHIFT_Qwen2VLConfig,
    iSHIFT_Qwen2VLProcessor
)

# ================== Configuration ==================
MODEL_PATH = "models/iSHIFT"
TEST_FILE = "sample_dataset/train.json"
OUTPUT_DIR = "inference/results/train"
DTYPE = torch.bfloat16
DETECTION_IMAGE_SEQLEN = 256
IMAGE_RESOLUTION = 512  # Max image resolution (matching notebook)

# ================== Helper Functions ==================

def preprocess_image(image_path: str, max_resolution: int = 512) -> Image.Image:
    """
    Load and preprocess an image for iSHIFT model.
    
    Matches training preprocessing from LLaMA-Factory/src/llamafactory/data/mm_plugin.py:
    - BasePlugin._preprocess_image (lines 74-87): max size + RGB conversion
    - Qwen2vlPlugin._preprocess_image (lines 454-468): min size + aspect ratio
    """
    image = Image.open(image_path)
    
    # Step 1: Resize if max dimension exceeds image_resolution (BasePlugin)
    if max(image.width, image.height) > max_resolution:
        resize_factor = max_resolution / max(image.width, image.height)
        width = int(image.width * resize_factor)
        height = int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.NEAREST)
    
    # Step 2: Convert to RGB (BasePlugin)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Step 3: Ensure minimum size for Qwen2-VL (Qwen2vlPlugin)
    if min(image.width, image.height) < 28:
        width, height = max(image.width, 28), max(image.height, 28)
        image = image.resize((width, height), resample=Image.NEAREST)
    
    # Step 4: Handle extreme aspect ratios (Qwen2vlPlugin)
    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height), resample=Image.NEAREST)
    
    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height), resample=Image.NEAREST)
    
    return image


def process_images_for_dino(images, dino_processor):
    """Process images for DINO encoder."""
    processed = dino_processor(images=images, return_tensors="pt")
    return processed['pixel_values']

@torch.no_grad()
def run_dynamic_ishift_inference(
    text_prompt: str,
    image_path: str,
    model,
    tokenizer,
    processor,
    dino_processor,
    device,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
) -> Dict:
    """Dynamic multi-round iSHIFT inference (matching notebook pipeline)."""
    
    DETECTION_TOKEN = '<|detection_action_start|>'
    result = {'query': text_prompt, 'image_path': image_path}
    
    # Preprocess image (matching notebook - resize, aspect ratio, etc.)
    image = preprocess_image(image_path, IMAGE_RESOLUTION)
    
    image_inputs = processor.image_processor(images=[image], return_tensors='pt')
    pixel_values = image_inputs['pixel_values'].to(device).to(DTYPE)
    image_grid_thw = image_inputs['image_grid_thw'].to(device)
    
    # Calculate image tokens correctly using merge_size
    merge_size = processor.image_processor.merge_size
    merge_length = merge_size ** 2
    num_image_tokens = image_grid_thw[0].prod().item() // merge_length
    
    image_token = '<|image_pad|>'
    image_placeholder = f'<|vision_start|>{image_token * num_image_tokens}<|vision_end|>'
    
    # Round 1
    round1_conversation = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{image_placeholder}\n{text_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    input_ids = tokenizer.encode(round1_conversation, add_special_tokens=False, return_tensors='pt').to(device)
    
    # model.generate(
    #     input_ids=input_ids,
    #     attention_mask=torch.ones_like(input_ids),
    #     pixel_values=pixel_values,
    #     image_grid_thw=image_grid_thw,
    #     max_new_tokens=max_new_tokens,
    #     do_sample=False,
    #     pad_token_id=tokenizer.pad_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )
    
    round1_output = '<|start-latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|end-latent|>'
    
    # Round 2
    round2_conversation = (
        round1_conversation +
        f"{round1_output}<|im_end|>\n"
        f"<|im_start|>user\nRequire additional perception features if required and then answer the question based on your observations or answer the question directly if additional perception features are not required<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    input_ids = tokenizer.encode(round2_conversation, add_special_tokens=False, return_tensors='pt').to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    round2_new_tokens = outputs[0, input_ids.shape[1]:]
    round2_output = tokenizer.decode(round2_new_tokens, skip_special_tokens=False)
    
    needs_detection = DETECTION_TOKEN in round2_output
    
    if not needs_detection:
        result['final_answer'] = round2_output.replace('<|im_end|>', '').strip()
        result['used_detection'] = False
        return result
    
    # Round 3 (detection)
    result['used_detection'] = True
    detection_images = process_images_for_dino([image], dino_processor).to(device).to(DTYPE)
    
    detection_token = '<|detection_image_pad|>'
    detection_placeholder = f'<|vision_start|>{detection_token * DETECTION_IMAGE_SEQLEN}<|vision_end|>'
    round2_output_fixed = '<|detection_action_start|><|detection_action|><|detection_action_end|>'
    
    round3_conversation = (
        round2_conversation +
        f"{round2_output_fixed}<|im_end|>\n"
        f"<|im_start|>user\n{detection_placeholder}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    input_ids = tokenizer.encode(round3_conversation, add_special_tokens=False, return_tensors='pt').to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        detection_images=detection_images,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    round3_new_tokens = outputs[0, input_ids.shape[1]:]
    round3_output = tokenizer.decode(round3_new_tokens, skip_special_tokens=False)
    result['final_answer'] = round3_output.replace('<|im_end|>', '').strip()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Single-GPU iSHIFT Inference')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID (0-indexed)')
    parser.add_argument('--num_gpus', type=int, required=True, help='Total number of GPUs')
    parser.add_argument('--test_file', type=str, default=TEST_FILE)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    
    device = "cuda"
    output_file = os.path.join(args.output_dir, f"results_gpu{args.gpu_id}.jsonl")
    
    # Load test data
    print(f"[GPU {args.gpu_id}] Loading test data...")
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    
    # Calculate chunk for this GPU
    chunk_size = len(test_data) // args.num_gpus
    start_idx = args.gpu_id * chunk_size
    end_idx = start_idx + chunk_size if args.gpu_id < args.num_gpus - 1 else len(test_data)
    data_chunk = test_data[start_idx:end_idx]
    
    print(f"[GPU {args.gpu_id}] Processing samples {start_idx} to {end_idx-1} ({len(data_chunk)} samples)")
    
    # Load model
    print(f"[GPU {args.gpu_id}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = iSHIFT_Qwen2VLProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    SPECIAL_TOKENS = {
        'detection_image_token_id': tokenizer.convert_tokens_to_ids('<|detection_image_pad|>'),
        'detection_action_id': tokenizer.convert_tokens_to_ids('<|detection_action|>'),
        'detection_action_start_id': tokenizer.convert_tokens_to_ids('<|detection_action_start|>'),
    }
    
    config = iSHIFT_Qwen2VLConfig.from_pretrained(
        MODEL_PATH,
        detection_image_token_id=SPECIAL_TOKENS['detection_image_token_id'],
        detection_action_id=SPECIAL_TOKENS['detection_action_id'],
        detection_action_start_id=SPECIAL_TOKENS['detection_action_start_id'],
        para_start_id=tokenizer.convert_tokens_to_ids('<|im_start|>'),
        para_end_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),
        num_inner_forward_run=2,
        projector_scale=1,
        para_mask_id=0,
        para_mask_ratio=0.5,
        alignment=False,
        vision_encoder_ls=['dino'],
        trust_remote_code=True,
    )
    
    model = iSHIFT_Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    if model.detection_model is not None:
        model.detection_model = model.detection_model.to(device=device, dtype=DTYPE)
    if model.detection_projector is not None:
        model.detection_projector = model.detection_projector.to(device=device, dtype=DTYPE)
    
    print(f"[GPU {args.gpu_id}] Model loaded. Starting inference...")
    
    model.config.use_cache = True
    model.generation_config.use_cache = True

    # Create output dir and file
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process samples
    with open(output_file, 'w') as f:
        for i, sample in enumerate(tqdm(data_chunk, desc=f"GPU {args.gpu_id}")):
            global_idx = start_idx + i
            
            try:
                image_path = sample['images'][0]
                original_query = sample['messages'][0]['content']
                query = original_query.replace('<image>\n', '')
                query = query.replace('\nRequire additional perception features, and then answer the question.', '')
                query = query + '\nLet\'s think step by step.'
                
                result = run_dynamic_ishift_inference(
                    text_prompt=query,
                    image_path=image_path,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    dino_processor=dino_processor,
                    device=device,
                )
                
                output_record = {
                    'index': global_idx,
                    'output': result['final_answer'],
                    'gt': sample['messages'][-1]['content'],
                    'used_detection': result['used_detection'],
                }
                
            except Exception as e:
                print(f"[GPU {args.gpu_id}]   on sample {global_idx}: {e}")
                output_record = {
                    'index': global_idx,
                    'output': f"ERROR: {str(e)}",
                    'used_detection': False,
                }
            
            f.write(json.dumps(output_record) + '\n')
            f.flush()
            
            # Clear CUDA cache periodically to prevent memory buildup
            if (i + 1) % 1000 == 0:
                torch.cuda.empty_cache()
    
    print(f"[GPU {args.gpu_id}] Done! Results saved to {output_file}")


if __name__ == '__main__':
    main()
