"""Βάση σύγκρισης: η μαθηματική δουλειά του plasma του AutoDJ (4 ημίτονα + swirl + LUT) σε numpy/CPU."""
import time

import numpy as np

W, H = 1920, 1080
aspect = W / H
ys = np.linspace(-1, 1, H, dtype=np.float32); xs = np.linspace(-aspect, aspect, W, dtype=np.float32)
X, Y = np.meshgrid(xs, ys); R = np.sqrt(X**2 + Y**2).astype(np.float32)
lut = np.stack([np.linspace(0, 255, 256)]*3, 1).astype(np.uint8)
p = dict(fx=2.5, fy=2.2, fd=1.9, fr=3.1, kx=1, ky=2, kd=1, kr=2, swirl=0.2)
def render(frame, total=300):
    t = np.float32(2*np.pi*frame/total); x, y, r = X, Y, R
    angle = np.float32(p["swirl"])*np.sin(t)*r; ca, sa = np.cos(angle), np.sin(angle)
    x, y = x*ca - y*sa, x*sa + y*ca
    v = np.sin(x*p["fx"] + t*p["kx"], dtype=np.float32)
    v += np.sin(y*p["fy"] + t*p["ky"], dtype=np.float32)
    v += np.sin((x+y)*p["fd"] + t*p["kd"], dtype=np.float32)
    v += np.sin(r*p["fr"] - t*p["kr"], dtype=np.float32)
    idx = (np.clip((v+4)/8, 0, 1)*255 + 0.5).astype(np.uint8)
    return lut[idx]
render(0); n = 30; t0 = time.perf_counter()
for i in range(n): f = render(i)
dt = time.perf_counter() - t0
print(f"numpy CPU plasma 1080p: {n/dt:.1f} fps  ({dt/n*1000:.0f} ms/καρέ)  shape={f.shape} dtype={f.dtype}")
