# UniVideo Fork: Multi-GPU + FlashAttention2

Two independent improvements, each in its own file.

## Files

| File | What it patches |
|---|---|
| `attn_processor_flash.py` | Drop-in FlashAttention2 processor replacing `HunyuanVideoAttnProcessor2_0` |
| `transformer_univideo_hunyuan_video_patched.py` | Diff-style notes showing every line to change in the original transformer file |
| `univideo_inference_multigpu.py` | Multi-GPU inference entry point replacing `univideo_inference.py` |
| `pipeline_multigpu_patch.md` | Notes on what to change inside `pipeline_univideo.py` |

## Quick-start

```bash
# 1. Install flash-attn (needs CUDA 11.8+ and matching torch)
pip install flash-attn --no-build-isolation

# 2. Run on 2 GPUs
torchrun --nproc_per_node=2 univideo_inference_multigpu.py \
    --demo_task t2v \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
```
