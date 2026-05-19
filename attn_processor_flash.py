"""
attn_processor_flash.py
-----------------------
Drop-in replacement for HunyuanVideoAttnProcessor2_0 that uses
FlashAttention-2 (via the `flash_attn` package) when available,
with a clean fallback to PyTorch SDPA (flash kernel) otherwise.

Usage in transformer_univideo_hunyuan_video.py
----------------------------------------------
Replace every occurrence of:
    processor=HunyuanVideoAttnProcessor2_0()
with:
    processor=HunyuanVideoFlashAttnProcessor()

And add at the top of that file:
    from attn_processor_flash import HunyuanVideoFlashAttnProcessor
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Try to import flash_attn; fall back gracefully.
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    from flash_attn.bert_padding import unpad_input, pad_input
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False

from diffusers.models.attention_processor import Attention


class HunyuanVideoFlashAttnProcessor:
    """
    Attention processor for HunyuanVideo / UniVideo that uses
    FlashAttention-2 when the package is present and the inputs
    are on CUDA with a float16/bfloat16 dtype.

    Falls back to PyTorch 2.0 scaled_dot_product_attention (flash
    kernel via sdp_kernel) otherwise — same as the original.

    Key differences from HunyuanVideoAttnProcessor2_0
    --------------------------------------------------
    * Uses flash_attn_func for the common (no-mask, no-padding) path.
    * Uses flash_attn_varlen_func for the masked / variable-length path.
    * Both paths avoid materialising the full O(N²) attention matrix.
    """

    def __init__(self, use_flash: bool = True):
        self.use_flash = use_flash and _FLASH_AVAILABLE

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rotary(query, key, encoder_len, image_rotary_emb):
        from diffusers.models.embeddings import apply_rotary_emb
        if encoder_len > 0:
            query = torch.cat(
                [
                    apply_rotary_emb(query[:, :, :-encoder_len], image_rotary_emb),
                    query[:, :, -encoder_len:],
                ],
                dim=2,
            )
            key = torch.cat(
                [
                    apply_rotary_emb(key[:, :, :-encoder_len], image_rotary_emb),
                    key[:, :, -encoder_len:],
                ],
                dim=2,
            )
        else:
            query = apply_rotary_emb(query, image_rotary_emb)
            key   = apply_rotary_emb(key,   image_rotary_emb)
        return query, key

    # ------------------------------------------------------------------
    # flash path — no attention mask, no variable lengths
    # ------------------------------------------------------------------

    def _flash_attn(self, query, key, value):
        """
        query / key / value: (B, heads, N, head_dim)  — standard layout
        flash_attn_func expects: (B, N, heads, head_dim)
        Returns: (B, heads, N, head_dim)
        """
        assert query.device == key.device == value.device, \
            f"flash_attn requires Q/K/V on same device, got {query.device}/{key.device}/{value.device}"
        # Transpose to flash_attn layout
        q = query.transpose(1, 2)   # (B, N, H, D)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        return out.transpose(1, 2)  # back to (B, H, N, D)

    # ------------------------------------------------------------------
    # varlen flash path — handles attention mask / padding
    # ------------------------------------------------------------------

    def _flash_attn_varlen(self, query, key, value, attention_mask):
        """
        attention_mask: (B, 1, 1, N) or (B, N) boolean / float mask
                        True / 1.0  = keep,  False / 0.0 = ignore
        Returns: (B, heads, N, head_dim)
        """
        B, H, N, D = query.shape

        # Normalise mask to (B, N) bool, True = valid
        if attention_mask.dim() == 4:
            mask = attention_mask.squeeze(1).squeeze(1).bool()  # (B, N)
        else:
            mask = attention_mask.bool()

        # Flatten batch for varlen API
        q_flat = query.transpose(1, 2).reshape(B * N, H, D)   # (B*N, H, D)
        k_flat = key.transpose(1, 2).reshape(B * N, H, D)
        v_flat = value.transpose(1, 2).reshape(B * N, H, D)

        # Build cumulative sequence lengths (cu_seqlens)
        seqlens = mask.sum(dim=1).to(torch.int32)              # (B,)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=query.device)
        cu_seqlens[1:] = seqlens.cumsum(0)
        max_seqlen = int(seqlens.max().item())

        # Remove padding tokens
        valid_mask_flat = mask.reshape(-1)                      # (B*N,)
        q_unpad = q_flat[valid_mask_flat]
        k_unpad = k_flat[valid_mask_flat]
        v_unpad = v_flat[valid_mask_flat]

        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
        )

        # Re-pad to (B*N, H, D)
        out_flat = torch.zeros_like(q_flat)
        out_flat[valid_mask_flat] = out_unpad
        out = out_flat.reshape(B, N, H, D).transpose(1, 2)     # (B, H, N, D)
        return out

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # ---- merge for single-stream (no separate encoder projections) ----
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)  # (B, H, N, D)
        key   = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK norm
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. RoPE
        if image_rotary_emb is not None:
            enc_len = encoder_hidden_states.shape[1] if (
                attn.add_q_proj is None and encoder_hidden_states is not None
            ) else 0
            query, key = self._apply_rotary(query, key, enc_len, image_rotary_emb)

        # 4. Dual-stream encoder QKV
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            enc_k = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            enc_v = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            query = torch.cat([query, enc_q], dim=2)
            key   = torch.cat([key,   enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        # 5. Attention
        use_flash_here = (
            self.use_flash
            and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)
        )

        if attention_mask is not None:
            # Extend mask to cover encoder tokens (always valid)
            enc_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            if enc_len > 0:
                enc_mask = torch.ones(B, enc_len, dtype=torch.bool, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask_2d, enc_mask], dim=1)  # (B, N+enc_len)
        
        if use_flash_here:
            if attention_mask is None:
                hidden_states = self._flash_attn(query, key, value)
            else:
                hidden_states = self._flash_attn_varlen(query, key, value, attention_mask)
        else:
            # Original SDPA fallback
            from torch.nn.attention import sdpa_kernel, SDPBackend
            import contextlib
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection + split encoder
            # Correct guard — match the original's structure
            if encoder_hidden_states is not None:
                enc_seq = encoder_hidden_states.shape[1]
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, :-enc_seq],
                    hidden_states[:, -enc_seq:],
                )
                if getattr(attn, "to_out", None) is not None:
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)
                if getattr(attn, "to_add_out", None) is not None:
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
