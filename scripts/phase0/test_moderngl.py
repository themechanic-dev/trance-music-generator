"""Φάση 0 — μέτρηση moderngl: plasma shader 1080p headless (EGL), fps + ανάγνωση καρέ."""
import sys
import time

import moderngl
import numpy as np

W, H = 1920, 1080
ctx = moderngl.create_context(standalone=True, backend="egl")
print("GL_RENDERER:", ctx.info["GL_RENDERER"])
print("GL_VERSION :", ctx.info["GL_VERSION"])

prog = ctx.program(
    vertex_shader="""
    #version 330
    in vec2 in_pos; out vec2 uv;
    void main(){ uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0.0,1.0); }
    """,
    fragment_shader="""
    #version 330
    uniform float t; uniform vec2 res; in vec2 uv; out vec4 f;
    void main(){
        vec2 p = (uv*2.0-1.0)*vec2(res.x/res.y,1.0);
        float v = 0.0;
        v += sin(p.x*3.0 + t);
        v += sin((p.y*3.0 + t)*0.7);
        v += sin((p.x*2.0 + p.y*2.0 + t)*0.5);
        vec2 c = p + vec2(sin(t*0.3), cos(t*0.2));
        v += sin(sqrt(dot(c,c))*6.0 - t*1.5);
        for(int i=0;i<24;i++){ v += 0.02*sin(p.x*float(i)*0.9 + p.y*float(24-i)*0.7 + t*0.1*float(i)); }
        float r = 0.5+0.5*sin(3.14159*v);
        float g = 0.5+0.5*sin(3.14159*v + 2.094);
        float b = 0.5+0.5*sin(3.14159*v + 4.188);
        f = vec4(r,g,b,1.0);
    }
    """,
)
quad = ctx.buffer(np.array([-1,-1, 1,-1, -1,1, 1,1], dtype="f4").tobytes())
vao = ctx.simple_vertex_array(prog, quad, "in_pos")
fbo = ctx.simple_framebuffer((W, H), components=4)
fbo.use()
prog["res"].value = (W, H)

def run(n, readback):
    ctx.finish(); t0 = time.perf_counter()
    for i in range(n):
        prog["t"].value = i / 30.0
        ctx.clear(); vao.render(moderngl.TRIANGLE_STRIP)
        if readback:
            data = fbo.read(components=3)
    ctx.finish(); return n / (time.perf_counter() - t0)

fps_gpu = run(300, False)
fps_rb  = run(120, True)
frame = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)
print(f"plasma 1080p render-only     : {fps_gpu:8.1f} fps")
print(f"plasma 1080p + readback to CPU: {fps_rb:8.1f} fps")
print("frame stats: mean", frame.mean().round(1), "std", frame.std().round(1), "(std>0 => πραγματική εικόνα)")
from PIL import Image

Image.fromarray(frame[::-1]).save(sys.argv[1] if len(sys.argv)>1 else "plasma_frame.png")
print("saved frame png")
