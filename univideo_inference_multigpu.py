"""
univideo_inference_multigpu.py
------------------------------
Multi-GPU inference for UniVideo using two complementary strategies:

  Strategy A — torch.distributed + device_map="auto" (tensor parallelism via
               HuggingFace Accelerate).  Best for a single node where the model
               doesn't fit in one GPU's VRAM.

  Strategy B — torchrun DDP sharding: each rank processes a different batch
               item in parallel (throughput scaling, not memory scaling).

Usage
-----
# Strategy A — model split across all visible GPUs (set via CUDA_VISIBLE_DEVICES)
python univideo_inference_multigpu.py \
    --demo_task t2v \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml \
    --strategy model_parallel

# Strategy B — DDP, each GPU handles one item of a batch
torchrun --nproc_per_node=2 univideo_inference_multigpu.py \
    --demo_task t2v \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml \
    --strategy ddp

Notes
-----
* Flash-attention is activated automatically when the `flash_attn` package
  is installed.  Pass --no_flash to force SDPA fallback.
* For DDP strategy the demo tasks are replicated; in production you'd feed
  different prompts to each rank via a dataloader.
"""

from __future__ import annotations

import os
import argparse
import yaml
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from transformer_univideo_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from mllm_encoder import MLLMInContext, MLLMInContextConfig
from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig
from utils import pad_image_pil_to_square, load_model

# FlashAttn processor (graceful fallback if flash_attn not installed)
from attn_processor_flash import HunyuanVideoFlashAttnProcessor


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_task", type=str, required=True,
                   choices=["understanding", "t2v", "t2i", "i2v",
                            "image_edit", "in_context_image_edit",
                            "in_context_video_gen",
                            "in_context_video_edit_addition",
                            "in_context_video_edit_swap",
                            "in_context_video_edit_style",
                            "video_edit", "stylization"])
    p.add_argument("--config", type=str,
                   default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml")
    p.add_argument("--strategy", type=str, default="model_parallel",
                   choices=["model_parallel", "ddp"],
                   help="model_parallel: split model across GPUs (memory). "
                        "ddp: each GPU runs the full model on a subset of batch.")
    p.add_argument("--no_flash", action="store_true",
                   help="Disable FlashAttention2 even if flash_attn is installed.")
    return p.parse_args()


# ──────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────

def init_ddp():
    """Initialise process group when launched with torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1  # single-process fallback
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def is_main(local_rank: int) -> bool:
    return local_rank == 0


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def build_pipeline(cfg: dict, strategy: str, local_rank: int, use_flash: bool):
    """
    Builds the full UniVideoPipeline.

    strategy == "model_parallel"
        Uses device_map="auto" (requires accelerate) to shard the transformer
        and VAE across all GPUs.  The MLLM encoder lands on GPU 0 by default
        (it's much smaller than the DiT).

    strategy == "ddp"
        Each rank loads the full model onto its own GPU.  The pipeline runs
        identically on each rank; you parallelise over the batch dimension
        outside.
    """
    mllm_config = MLLMInContextConfig(**cfg["mllm_config"])
    pipe_cfg    = UniVideoPipelineConfig(**cfg["pipeline_config"])
    transformer_ckpt_path  = cfg.get("transformer_ckpt_path")
    mllm_encoder_ckpt_path = cfg.get("mllm_encoder_ckpt", None)

    # ── MLLM encoder ──────────────────────────────────────────────────
    mllm_encoder = MLLMInContext(mllm_config)
    if mllm_encoder_ckpt_path is not None:
        if is_main(local_rank):
            print(f"[INIT] loading mllm_encoder ckpt from {mllm_encoder_ckpt_path}")
        mllm_encoder = load_model(mllm_encoder, mllm_encoder_ckpt_path)
    mllm_encoder.requires_grad_(False).eval()

    # ── VAE ───────────────────────────────────────────────────────────
    vae_kwargs = dict(subfolder="vae", low_cpu_mem_usage=False, device_map=None)
    if strategy == "model_parallel":
        vae_kwargs["device_map"] = "auto"   # shard if large enough
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        pipe_cfg.hunyuan_model_id, **vae_kwargs
    ).eval()

    # ── Transformer ───────────────────────────────────────────────────
    qwenvl_txt_dim = 3584
    transformer_kwargs = dict(
        subfolder="transformer",
        low_cpu_mem_usage=False,
        device_map=None,
        text_embed_dim=qwenvl_txt_dim,
    )
    if strategy == "model_parallel":
        # device_map="auto" shards layers across GPUs by parameter count.
        # This is the key change for fitting a 13B+ model on multiple GPUs.
        transformer_kwargs["device_map"] = "auto"

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pipe_cfg.hunyuan_model_id, **transformer_kwargs
    )

    # Reinitialise the Qwen projection head
    transformer.qwen_project_in = TwoLayerMLP(qwenvl_txt_dim, qwenvl_txt_dim * 4, 4096)
    with torch.no_grad():
        torch.nn.init.ones_(transformer.qwen_project_in.ln.weight)
        for layer in transformer.qwen_project_in.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    # ── Swap in FlashAttention processor ──────────────────────────────
    flash_proc = HunyuanVideoFlashAttnProcessor(use_flash=use_flash)
    _swap_attn_processors(transformer, flash_proc)
    if is_main(local_rank):
        fa_label = "FlashAttention2" if flash_proc.use_flash else "SDPA (fallback)"
        print(f"[INIT] Attention backend: {fa_label}")

    # ── Load fine-tuned checkpoint ────────────────────────────────────
    def rename_func(state_dict):
        return {
            k.replace("transformer.", "", 1) if k.startswith("transformer.") else k: v
            for k, v in state_dict.items()
        }

    if isinstance(transformer_ckpt_path, str):
        if is_main(local_rank):
            print(f"[INIT] loading transformer ckpt from {transformer_ckpt_path}")
        transformer = load_model(transformer, transformer_ckpt_path, rename_func=rename_func)

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pipe_cfg.hunyuan_model_id, subfolder="scheduler"
    )

    # ── Assemble pipeline ─────────────────────────────────────────────
    if strategy == "model_parallel":
        # device_map="auto" already placed modules; don't call .to(device=...)
        pipeline = UniVideoPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm_encoder=mllm_encoder,
            univideo_config=pipe_cfg,
        ).to(dtype=torch.bfloat16)
    else:
        device = f"cuda:{local_rank}"
        pipeline = UniVideoPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm_encoder=mllm_encoder,
            univideo_config=pipe_cfg,
        ).to(device=device, dtype=torch.bfloat16)

    return pipeline


def _swap_attn_processors(model: torch.nn.Module,
                           processor: HunyuanVideoFlashAttnProcessor):
    """
    Recursively replace every HunyuanVideoAttnProcessor2_0 with our
    FlashAttn processor.  Works for all Attention submodules.
    """
    from diffusers.models.attention_processor import Attention
    # Import original processor class to check isinstance
    try:
        from transformer_univideo_hunyuan_video import HunyuanVideoAttnProcessor2_0 as _Orig
    except ImportError:
        _Orig = None

    for module in model.modules():
        if isinstance(module, Attention):
            if _Orig is None or isinstance(module.processor, _Orig):
                module.set_processor(processor)


# ──────────────────────────────────────────────
# Demo task pipeline kwargs  (same as original)
# ──────────────────────────────────────────────

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, walking backwards, computer-generated environment, "
    "weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."
)


def get_pipeline_kwargs(demo_task: str):  # noqa: C901
    if demo_task == "understanding":
        return dict(
            prompts=["Describe this video in detail"],
            cond_video_path="demo/understanding/1.mp4",
            seed=42, task="understanding",
        ), None

    if demo_task == "t2i":
        return dict(
            prompts=["A cute hamster lies leisurely on a lifebuoy, wearing fashionable sunglasses, and drifts with the gentle waves on the shimmering sea surface. The hamster reclines comfortably, enjoying a peaceful and pleasant time. Cartoon style, the camera follows the subject moving, with a heartwarming and high picture quality."],
            negative_prompt=NEGATIVE_PROMPT,
            height=1024, width=1024, num_frames=1,
            num_inference_steps=50, guidance_scale=7.0,
            image_guidance_scale=1.0, seed=42, timestep_shift=7.0, task="t2i",
        ), "demo/t2i/output.jpg"

    if demo_task == "t2v":
        return dict(
            prompts=["a stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."],
            negative_prompt=NEGATIVE_PROMPT,
            height=480, width=854, num_frames=61,
            num_inference_steps=30, guidance_scale=6.0,
            image_guidance_scale=1.0, seed=42, timestep_shift=7.0, task="t2v",
        ), "demo/t2v/output.mp4"

    if demo_task == "i2v":
        return dict(
            prompts=["The video shows a small capybara wearing round glasses, holding a book titled 'UniVideo' on its cover."],
            negative_prompt=NEGATIVE_PROMPT,
            cond_image_path="demo/i2v/1.png",
            height=480, width=854, num_frames=129,
            num_inference_steps=30, guidance_scale=5.0,
            image_guidance_scale=1.0, seed=42, timestep_shift=7.0, task="i2v",
        ), "demo/i2v/output.mp4"

    if demo_task == "image_edit":
        return dict(
            prompts=["Change the background to dessert."],
            negative_prompt=NEGATIVE_PROMPT,
            cond_image_path="demo/image_edit/1.jpg",
            height=480, width=832, num_frames=1,
            num_inference_steps=50, guidance_scale=7.0,
            image_guidance_scale=2.0, seed=42, timestep_shift=7.0, task="i2i_edit",
        ), "demo/image_edit/output.jpg"

    if demo_task == "video_edit":
        return dict(
            prompts=["Change the man to look like he is sculpted from chocolate."],
            negative_prompt=NEGATIVE_PROMPT,
            cond_video_path="demo/video_edit/video.mp4",
            height=480, width=854, num_frames=129,
            num_inference_steps=50, guidance_scale=7.0,
            image_guidance_scale=2.0, seed=42, timestep_shift=7.0, task="v2v_edit",
        ), "demo/video_edit/output.mp4"

    if demo_task == "stylization":
        return dict(
            prompts=["Change the style of the video to minecraft."],
            negative_prompt=NEGATIVE_PROMPT,
            cond_video_path="demo/video_edit/video.mp4",
            height=480, width=832, num_frames=77,
            num_inference_steps=30, guidance_scale=7.0,
            image_guidance_scale=2.0, seed=42, timestep_shift=7.0, task="v2v_edit",
        ), "demo/video_edit/style/output.mp4"

    # in-context variants
    def _load_refs(paths):
        return [[pad_image_pil_to_square(Image.open(p).convert("RGB")) for p in paths]]

    if demo_task == "in_context_image_edit":
        return dict(
            prompts=["Let the woman wear the hat in the reference image."],
            negative_prompt=NEGATIVE_PROMPT,
            ref_images=_load_refs(["demo/in_context_image_edit/id.jpeg"]),
            cond_image_path="demo/in_context_image_edit/input.jpg",
            height=480, width=832, num_frames=1,
            num_inference_steps=30, guidance_scale=7.0,
            image_guidance_scale=2.0, seed=42, timestep_shift=7.0, task="i+i2i_edit",
        ), "demo/in_context_image_edit/output.jpg"

    if demo_task == "in_context_video_gen":
        return dict(
            prompts=["A man with short, light brown hair sits on a beach lounge chair with a Pikachu on his shoulder."],
            negative_prompt=NEGATIVE_PROMPT,
            ref_images=_load_refs(["demo/in_context_video_gen/1.png",
                                   "demo/in_context_video_gen/2.png",
                                   "demo/in_context_video_gen/3.jpg"]),
            height=480, width=832, num_frames=129,
            num_inference_steps=50, guidance_scale=5.0,
            image_guidance_scale=3.0, seed=42, timestep_shift=7.0, task="multiid",
        ), "demo/in_context_video_gen/output.mp4"

    if demo_task == "in_context_video_edit_addition":
        return dict(
            prompts=["Add the hat from the reference image to the video."],
            negative_prompt=NEGATIVE_PROMPT,
            ref_images=_load_refs(["demo/in_context_video_edit/id_addition/images.jpeg"]),
            cond_video_path="demo/in_context_video_edit/id_addition/reference.mp4",
            height=480, width=832, num_frames=129,
            num_inference_steps=30, guidance_scale=7.0,
            image_guidance_scale=2.0, seed=42, timestep_shift=7.0, task="i+v2v_edit",
        ), "demo/in_context_video_edit/id_addition/output.mp4"

    if demo_task == "in_context_video_edit_swap":
        return dict(
            prompts=["Use the man's face in the reference image to replace the man's face in the video."],
            negative_prompt=NEGATIVE_PROMPT,
            ref_images=_load_refs(["demo/in_context_video_edit/id_swap/ID.jpeg"]),
            cond_video_path="demo/in_context_video_edit/id_swap/origin.mp4",
            height=480, width=832, num_frames=129,
            num_inference_steps=50, guidance_scale=7.0,
            image_guidance_scale=2.0, seed=42, timestep_shift=7.0, task="i+v2v_edit",
        ), "demo/in_context_video_edit/id_swap/output.mp4"

    if demo_task == "in_context_video_edit_style":
        return dict(
            prompts=["Change the video to the style of the reference image."],
            negative_prompt=NEGATIVE_PROMPT,
            ref_images=_load_refs(["demo/in_context_video_edit/style/ref.jpg"]),
            cond_video_path="demo/in_context_video_edit/style/video.mp4",
            height=832, width=480, num_frames=61,
            num_inference_steps=30, guidance_scale=7.0,
            image_guidance_scale=2.0, seed=42, timestep_shift=7.0, task="i+v2v_edit",
        ), "demo/in_context_video_edit/style/output.mp4"

    raise ValueError(f"Unknown demo_task: {demo_task}")


# ──────────────────────────────────────────────
# Save output
# ──────────────────────────────────────────────

def save_output(output, output_path: str | None):
    if output_path is None:
        return
    if hasattr(output, "text") and output.text is not None:
        for i, t in enumerate(output.text):
            print(f"[Output {i}] {repr(t)}")
        return

    frames = output.frames[0]
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().float().numpy()

    F, H, W, C = frames.shape
    if F == 1:
        img = frames[0]
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img = (img * 255).clip(0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        Image.fromarray(img).save(output_path)
        print(f"[Saved image] {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        export_to_video(frames, output_path, fps=24)
        print(f"[Saved video] {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Init distributed (no-op if single GPU) ────────────────────────
    local_rank, world_size = init_ddp()
    if is_main(local_rank):
        print(f"[DIST] strategy={args.strategy}, world_size={world_size}")

    # ── Load config ───────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Build pipeline ────────────────────────────────────────────────
    pipeline = build_pipeline(
        cfg,
        strategy=args.strategy,
        local_rank=local_rank,
        use_flash=not args.no_flash,
    )

    # ── Get task kwargs ───────────────────────────────────────────────
    pipeline_kwargs, output_path = get_pipeline_kwargs(args.demo_task)

    # For DDP: all ranks run the same task.  In production, slice a
    # shared prompts list: pipeline_kwargs["prompts"] = my_shard[local_rank]
    # and gather results on rank 0 with dist.gather_object().

    # ── Run inference ─────────────────────────────────────────────────
    with torch.inference_mode():
        output = pipeline(**pipeline_kwargs)

    # ── Save on rank 0 only ───────────────────────────────────────────
    if is_main(local_rank) and output_path is not None:
        save_output(output, output_path)
    elif args.demo_task == "understanding":
        if hasattr(output, "text") and output.text:
            print(f"[Rank {local_rank}] {output.text[0]}")

    # Clean up process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
