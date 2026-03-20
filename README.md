# iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception (CVPR'26)

[arXiv Paper](https://arxiv.org/abs/2512.22009)
[Model](https://huggingface.co/SarthakM320/iSHIFT)

Official repository for iSHIFT, a lightweight 2.5B GUI agent that integrates latent thinking with adaptive visual perception for mobile device automation.

## Project Structure

```
iSHIFT/
├── configs/                   # Training config files (alignment + fine-tuning)
├── inference/                 # Inference and evaluation scripts
│   ├── multi_gpu.sh           # Multi-GPU inference launcher
│   ├── multi_gpu_inference.py # Single-GPU inference worker
│   └── llamafactory_evaluation.py # LLaMA-Factory based evaluation
├── modules/
│   ├── LLaMA-Factory/         # Modified LLaMA-Factory with iSHIFT template
│   └── transformers/          # Custom Qwen2-VL model with iSHIFT architecture
├── aitw/                      # Android In The Wild dataset utilities and metrics
├── sample_dataset/            # Sample training data with images
└── env.yml                    # Conda environment file
```

## Environment Setup

```bash
conda env create -f env.yml
conda activate ishift
conda install pip
pip install -e modules/LLaMA-Factory
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.45.2
cp -r ../modules/transformers/src/transformers/models/qwen2_vl_ishift src/transformers/models
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Download Checkpoint

```bash
huggingface-cli download SarthakM320/iSHIFT --local-dir models/iSHIFT
```

## Download Base Models for Training

```python
from transformers import Qwen2VLForConditionalGeneration, AutoModel, SamModel
qwen2b = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
dino = AutoModel.from_pretrained('facebook/dinov2-large')
sam = SamModel.from_pretrained("facebook/sam-vit-large")
qwen2b.save_pretrained("models/Qwen2-VL-2B-Instruct", safe_serialization = False)
dino.save_pretrained("models/DINOv2-Large", safe_serialization = False)
sam.save_pretrained("models/Sam-Large", safe_serialization = False)
```

## Training

iSHIFT uses a two-stage training pipeline via LLaMA-Factory:

### Stage 1: Alignment

Trains only the detection projector to align DINOv2 features with the language model.

```bash
llamafactory-cli train configs/Qwen2-VL-2B-alignment.yaml
```

Key settings: `lr=2e-3`, `epochs=5`, `batch_size=8`, `grad_accum=4`

### Stage 2: Fine-tuning

Freezes the detection model and visual encoders, trains the rest of the model with latent thinking and perception control tokens.

```bash
llamafactory-cli train configs/Qwen2-VL-2b-iSHIFT.yaml
```

Key settings: `lr=2e-5`, `epochs=1`, `batch_size=8`, `grad_accum=4`, `para_mask_ratio=0.5`

> **Note:** Update `model_name_or_path` in the fine-tuning config to point to your Stage 1 checkpoint, and `output_dir` in both configs to your desired save location.

## Inference

Run multi-GPU inference across 8 GPUs:

```bash
bash inference/multi_gpu.sh
```

This distributes the test data across GPUs, runs inference in parallel, and merges results into `inference/results/<test_name>/results_all.jsonl`.

To customize, edit `inference/multi_gpu.sh`:

- `NUM_GPUS` — number of GPUs
- `TEST_FILE` — path to your test JSON
- `OUTPUT_DIR` — results output directory

## Dataset Format

Training data follows a multi-turn ShareGPT conversation format:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nPrevious Actions: [...]\nGoal: [...]\nPredict the next action."},
    {"role": "assistant", "content": "<|start-latent|>...<|end-latent|>"},
    {"role": "user", "content": "Require additional perception features..."},
    {"role": "assistant", "content": "<|detection_action_start|>...<|detection_action_end|>"},
    {"role": "user", "content": "<detection_image>"},
    {"role": "assistant", "content": "Action Plan: [...]; Action Decision: {...}"}
  ],
  "images": ["path/to/screenshot.png"],
  "detection_images": ["path/to/screenshot.png"]
}
```

A sample dataset with 100  examples is provided in `sample_dataset/`.

## Citation

```bibtex
@misc{mehrotra2025ishiftlightweightslowfastgui,
      title={iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception}, 
      author={Sarthak Mehrotra and Sairam V C Rebbapragada and Mani Hemanth Reddy Bonthu and Vineeth N Balasubramanian},
      year={2025},
      eprint={2512.22009},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.22009}, 
}
```

