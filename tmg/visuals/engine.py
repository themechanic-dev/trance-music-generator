"""The GPU engine: moderngl on a headless EGL context. Two slots (A, B) so shots can crossfade, feedback
textures for the generators that remember (flow, reaction), palette / spectrum / images as textures,
a post pass for the mix, the flash and the vignette, and a top-down RGB read-back for ffmpeg."""

from __future__ import annotations

import numpy as np

from tmg.visuals import shaders

FEEDBACK = {"flow", "reaction"}
REACTION_STEPS = 10
REACTION_WARMUP = 2500    # simulation steps run when a reaction shot starts, so it opens on a formed pattern
REACTION_SCALE = 0.5
REACTION_PRESETS = ((0.0367, 0.0649), (0.0545, 0.0620), (0.0295, 0.0561), (0.0250, 0.0600), (0.0392, 0.0649))


class _Slot:
    def __init__(self, ctx, width: int, height: int) -> None:
        self.ctx = ctx
        self.width, self.height = width, height
        self.color = ctx.texture((width, height), 3)
        self.color.repeat_x = self.color.repeat_y = False
        self.fbo = ctx.framebuffer(color_attachments=[self.color])
        self.generator: str | None = None
        self.seedv = (0.0, 0.0, 0.0, 0.0)
        self.palette_tex = ctx.texture((256, 1), 3)
        self.palette_tex.repeat_x = False
        self.started_frame = 0
        # feedback pair (full size) and reaction pair (half size, RG float)
        self.fb = [self._rgb(width, height) for _ in range(2)]
        self.fb_fbo = [ctx.framebuffer(color_attachments=[t]) for t in self.fb]
        rw, rh = max(64, int(width * REACTION_SCALE)), max(64, int(height * REACTION_SCALE))
        self.rx = [ctx.texture((rw, rh), 4, dtype="f4") for _ in range(2)]
        for t in self.rx:
            t.repeat_x = t.repeat_y = True
        self.rx_fbo = [ctx.framebuffer(color_attachments=[t]) for t in self.rx]
        self.rx_size = (rw, rh)
        self.cur = 0
        self.reaction_params = REACTION_PRESETS[0]
        self.img: tuple[int, int] = (0, 1)

    def _rgb(self, w: int, h: int):
        t = self.ctx.texture((w, h), 3)
        t.repeat_x = t.repeat_y = True
        return t

    def reset(self, rng: np.random.Generator) -> None:
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for t in self.fb:
            t.write(black.tobytes())
        rw, rh = self.rx_size
        state = np.zeros((rh, rw, 4), dtype=np.float32)
        state[..., 0] = 1.0
        for _ in range(int(rng.integers(4, 9))):
            cy, cx, r = int(rng.integers(6, rh - 6)), int(rng.integers(6, rw - 6)), int(rng.integers(3, 8))
            state[cy - r:cy + r, cx - r:cx + r, 1] = 1.0
        state[..., 1] += rng.random((rh, rw)).astype(np.float32) * 0.02
        for t in self.rx:
            t.write(state.tobytes())
        self.reaction_params = REACTION_PRESETS[int(rng.integers(0, len(REACTION_PRESETS)))]
        self.cur = 0


class Engine:
    def __init__(self, width: int, height: int, palettes: dict[str, np.ndarray], images: list[np.ndarray] | None = None,
                 backend: str = "egl") -> None:
        import moderngl

        self.gl = moderngl
        self.ctx = moderngl.create_context(standalone=True, backend=backend)
        self.width, self.height = width, height
        self.quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.programs: dict[str, object] = {}
        self.vaos: dict[str, object] = {}
        self.palettes = palettes
        self.slots = {"a": _Slot(self.ctx, width, height), "b": _Slot(self.ctx, width, height)}
        self.out_tex = self.ctx.texture((width, height), 3)
        self.out_fbo = self.ctx.framebuffer(color_attachments=[self.out_tex])
        self.spectrum_tex = self.ctx.texture((64, 1), 1, dtype="f4")
        self.spectrum_tex.repeat_x = False
        self.spectrum_tex.write(np.zeros(64, dtype=np.float32).tobytes())
        self.images = []
        for img in images or []:
            t = self.ctx.texture((img.shape[1], img.shape[0]), 3, img.tobytes())
            t.repeat_x = t.repeat_y = False
            t.build_mipmaps()
            self.images.append(t)
        self.post = self._program(shaders.POST, "post")
        self.reaction_sim = self._program(shaders.REACTION_SIM, "reaction_sim")
        self.renderer = self.ctx.info.get("GL_RENDERER", "?")
        self.has_warm = False

    # ---- programs ------------------------------------------------------------------
    def _program(self, fragment: str, key: str):
        if key not in self.programs:
            prog = self.ctx.program(vertex_shader=shaders.VERTEX, fragment_shader=fragment)
            self.programs[key] = prog
            self.vaos[key] = self.ctx.simple_vertex_array(prog, self.quad, "in_pos")
        return self.programs[key]

    def generator_program(self, name: str):
        if name not in self.programs:
            self._program(shaders.fragment_source(name), name)
        return self.programs[name]

    @staticmethod
    def _set(prog, name: str, value) -> None:
        if name in prog:
            prog[name].value = value

    # ---- shots ---------------------------------------------------------------------
    def begin_shot(self, slot_name: str, generator: str, palette_id: str, seed: int, frame: int) -> None:
        slot = self.slots[slot_name]
        if generator == "stills" and not self.images:
            generator = "domainwarp"
        slot.generator = generator
        rng = np.random.default_rng(seed)
        slot.seedv = tuple(float(x) for x in rng.random(4))
        lut = self.palettes.get(palette_id)
        if lut is None:
            lut = next(iter(self.palettes.values()))
        slot.palette_tex.write(np.ascontiguousarray(lut, dtype=np.uint8).tobytes())
        slot.started_frame = frame
        if self.images:
            n = len(self.images)
            i0 = int(rng.integers(0, n))
            i1 = int(rng.integers(0, n)) if n > 1 else i0
            if n > 1 and i1 == i0:
                i1 = (i0 + 1) % n
            slot.img = (i0, i1)
        slot.reset(rng)
        self.generator_program(generator)
        if generator == "reaction":
            self._reaction_steps(slot, REACTION_WARMUP, 0.0)

    def _render_slot(self, slot: _Slot, u: dict, frame: int, fps: float) -> None:
        gen = slot.generator or "plasma"
        prog = self.generator_program(gen)
        shot_t = (frame - slot.started_frame) / fps
        slot.palette_tex.use(0)
        self.spectrum_tex.use(1)
        if gen == "reaction":
            self._reaction_steps(slot, REACTION_STEPS + int(6 * u["energy"]), float(u["kick"]))
            slot.rx[slot.cur].use(2)
        elif gen in FEEDBACK:
            slot.fb[slot.cur].use(2)
        else:
            slot.fb[0].use(2)
        if self.images:
            self.images[slot.img[0]].use(3)
            self.images[slot.img[1]].use(4)
        for name, val in u.items():
            self._set(prog, name, val)
        self._set(prog, "res", (float(self.width), float(self.height)))
        self._set(prog, "shot_t", float(shot_t))
        self._set(prog, "seedv", slot.seedv)
        self._set(prog, "palette", 0)
        self._set(prog, "spectrum", 1)
        self._set(prog, "prev", 2)
        self._set(prog, "img0", 3)
        self._set(prog, "img1", 4)
        target = slot.fb_fbo[1 - slot.cur] if gen in FEEDBACK and gen != "reaction" else slot.fbo
        target.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.vaos[gen].render(self.gl.TRIANGLE_STRIP)
        if gen in FEEDBACK and gen != "reaction":
            slot.cur = 1 - slot.cur
            # the feedback result is the picture: copy into the slot colour buffer
            self.ctx.copy_framebuffer(slot.fbo, slot.fb_fbo[slot.cur])

    def _reaction_steps(self, slot: _Slot, steps: int, kick: float) -> None:
        feed, kill = slot.reaction_params
        sim = self.reaction_sim
        rw, rh = slot.rx_size
        self._set(sim, "texel", (1.0 / rw, 1.0 / rh))
        self._set(sim, "feed", feed)
        self._set(sim, "kill", kill)
        self._set(sim, "kick", kick)
        self._set(sim, "state", 2)
        self.ctx.viewport = (0, 0, rw, rh)
        for _ in range(steps):
            src, dst = slot.cur, 1 - slot.cur
            slot.rx[src].use(2)
            slot.rx_fbo[dst].use()
            self.vaos["reaction_sim"].render(self.gl.TRIANGLE_STRIP)
            slot.cur = dst

    # ---- frames ----------------------------------------------------------------------
    def set_spectrum(self, bands: np.ndarray) -> None:
        self.spectrum_tex.write(np.ascontiguousarray(bands, dtype=np.float32).tobytes())

    def render_frame(self, uniforms: dict, frame: int, fps: float, mix: float = 0.0) -> bytes:
        self._render_slot(self.slots["a"], uniforms, frame, fps)
        if mix > 0.0 and self.slots["b"].generator:
            self._render_slot(self.slots["b"], uniforms, frame, fps)
        post = self.post
        self.slots["a"].color.use(0)
        (self.slots["b"].color if mix > 0.0 else self.slots["a"].color).use(1)
        self._set(post, "texA", 0)
        self._set(post, "texB", 1)
        self._set(post, "mixv", float(mix))
        self._set(post, "impact", float(uniforms["impact"]))
        self._set(post, "kick", float(uniforms["kick"]))
        self._set(post, "phrase", float(uniforms["phrase"]))
        self._set(post, "res", (float(self.width), float(self.height)))
        self.out_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.vaos["post"].render(self.gl.TRIANGLE_STRIP)
        return self.out_fbo.read(components=3)

    def swap_slots(self) -> None:
        self.slots["a"], self.slots["b"] = self.slots["b"], self.slots["a"]

    def release(self) -> None:
        self.ctx.release()


__all__ = ["Engine", "FEEDBACK"]
