# GPU-Accelerated SetFit Training

This document explains how to use GPU acceleration for faster SetFit model training.

## Prerequisites

### 1. NVIDIA GPU with CUDA
- Ensure you have an NVIDIA GPU with at least 6GB VRAM
- Install the latest NVIDIA drivers from https://www.nvidia.com/Download/index.aspx

### 2. Install CUDA-enabled PyTorch
```bash
# For CUDA 11.8 (most common)
uv add torch --index https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1 (newer GPUs)
uv add torch --index https://download.pytorch.org/whl/cu121
```

### 3. Verify GPU Setup
```bash
uv run python gpu_check.py
```

Expected output:
```
✅ PyTorch: 2.x.x
✅ CUDA Available: 11.8/12.1
✅ GPU: NVIDIA GeForce RTX 3060 / GTX 1660 etc.
✅ VRAM: 6.0 GB
💡 Recommended settings for SetFit training:
   --batch-size 8 --accum-steps 4
```

## Usage

### Basic GPU Training
```bash
uv run python setfit_trainer_gpu.py --layer all
```

### Custom Batch Size
```bash
# For 6GB VRAM, these defaults work well:
uv run python setfit_trainer_gpu.py --layer all --batch-size 8 --accum-steps 4
```

You can tweak based on your specific GPU:
- **GTX 1660, RTX 3050, RTX 3060 (6GB)**: `--batch-size 8 --accum-steps 4` (effective batch=32)
- **RTX 2060, RTX 3060 Ti (8GB)**: `--batch-size 16 --accum-steps 4`
- **RTX 3070, RTX 4060 (8GB+)**: `--batch-size 16 --accum-steps 2`

### Disable Mixed Precision (if you get NaN errors)
```bash
uv run python setfit_trainer_gpu.py --layer all --no-mixed-precision
```

### Train Specific Layer Only
```bash
uv run python setfit_trainer_gpu.py --layer tier2
uv run python setfit_trainer_gpu.py --layer tier3
```

## Understanding Hyperparameters

- `batch_size`: Number of samples processed before weights update (per GPU)
- `accum_steps`: Accumulate gradients over N batches before updating
  - Larger `accum_steps` = more memory efficient but slower
  - Effective batch = `batch_size * accum_steps`
- `mixed precision`: Uses fp16 to halve memory usage
  - Recommended for all GPUs
  - Disable only if you encounter numerical instability

## Performance Expectations

| GPU | VRAM | Default Batch | Effective Batch | Training Time (all tiers) |
|-----|------|---------------|-----------------|--------------------------|
| GTX 1660 | 6GB | 8×4=32 | ~5-8 minutes |
| RTX 3060 | 12GB | 16×4=64 | ~3-5 minutes |
| RTX 2080 Ti | 11GB | 16×4=64 | ~3-5 minutes |
| CPU (no GPU) | - | 4×4=16 | ~20-30 minutes |

## After Training

1. Models are saved to `./setfit_models/`
2. Upload to Azure:
   ```bash
   uv run python model_azure.py
   ```
3. Restart FastAPI to hot-reload models or use `/api/models/reload` endpoint
4. Test on pipeline page: http://localhost:8000/models-pipeline

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or increase accumulation:
```bash
uv run python setfit_trainer_gpu.py --layer all --batch-size 4 --accum-steps 8
```

### NaN Losses
Disable mixed precision:
```bash
uv run python setfit_trainer_gpu.py --layer all --no-mixed-precision
```

### Multi-GPU Setup
The script automatically uses DataParallel if multiple GPUs are detected. Training will use all available GPUs.

## Notes

- Training is still relatively fast even on CPU (~20-30 min total) - only use GPU if you need to iterate quickly
- The 6GB VRAM limit is a hard constraint - you cannot increase batch size beyond what fits in memory
- Mixed precision is safe for SetFit - the model doesn't rely on extreme numerical precision

## Comparison: CPU vs GPU

| Aspect | CPU | GPU |
|--------|-----|-----|
| Training time | 20-30 min | 3-8 min |
| Memory usage | ~4GB RAM | ~4-5GB VRAM |
| Power consumption | Lower (CPU only) | Higher (GPU active) |
| Convenience | Always available | Requires CUDA setup |

**Recommendation**: Use GPU for development/experimentation, CPU for scheduled training.
