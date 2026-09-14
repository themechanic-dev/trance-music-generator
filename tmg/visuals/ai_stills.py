"""AI stills with SD-Turbo (diffusers, CUDA): a handful of abstract images per production for the 'stills' generator.

First time the CUDA backend runs in this project family - the AutoDJ never could (Docker without a GPU).
The model (stabilityai/sd-turbo, ~2.5 GB fp16) is downloaded into models/hf on first use.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

MODEL_ID = "stabilityai/sd-turbo"
NEGATIVE = ("text, watermark, logo, signature, face, human, person, hands, blurry, low quality, "
            "jpeg artifacts, oversaturated, borders, frame")
STYLE = "abstract, non-representational, seamless, cinematic lighting, ultra detailed, dark background"
PROMPTS = [
    "vast deep space nebula, clouds of cyan and violet gas, distant starfield, volumetric depth",
    "endless neon geometric tunnel, glowing magenta and cyan edges, perfect symmetry, vanishing point",
    "flowing liquid chrome surface, mirror metal ripples, iridescent reflections, molten silver waves",
    "aurora borealis over a still black ocean, teal and violet curtains of light, mirrored reflection",
    "intricate fractal bloom unfolding, recursive petals, luminous edges, deep blue to magenta gradient",
    "iridescent smoke curling in darkness, oil-slick colours, slow motion, macro",
    "crystal cavern lit from within, refracted beams, amethyst and teal, sharp facets",
    "laser grid horizon at night, retro synthwave, purple haze, glowing lines to infinity",
    "bioluminescent jellyfish forms drifting in black water, soft glow, particles",
    "molten glass sculpture, orange and blue light inside, black void",
]


def generate(out_dir: Path, count: int, seed: int, *, width: int = 768, height: int = 432, steps: int = 2,
             progress=None) -> list[Path]:
    """Write `count` PNGs into out_dir. Returns their paths. Raises if CUDA/diffusers are unavailable."""
    import torch
    from diffusers import AutoPipelineForText2Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for the AI stills")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if progress:
        progress(0.0, "loading SD-Turbo")
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rng = random.Random(seed)
    paths_out: list[Path] = []
    for i in range(count):
        prompt = rng.choice(PROMPTS) + ", " + STYLE
        g = torch.Generator("cuda").manual_seed(seed + i)
        img = pipe(prompt=prompt, negative_prompt=NEGATIVE, num_inference_steps=steps, guidance_scale=0.0,
                   width=width, height=height, generator=g).images[0]
        p = out_dir / f"still-{seed}-{i:02d}.png"
        img.save(p)
        paths_out.append(p)
        if progress:
            progress((i + 1) / count, f"AI still {i + 1}/{count}")
    del pipe
    torch.cuda.empty_cache()
    if progress:
        progress(1.0, f"{count} stills in {time.perf_counter() - t0:.0f} s")
    return paths_out


__all__ = ["MODEL_ID", "PROMPTS", "generate"]
