"""GLSL sources: one fragment shader per generator, sharing a prelude of uniforms and noise helpers.

Every generator is a function `vec3 scene()` over the full-screen quad. The uniforms are the same for
all of them, so the engine can drive any generator with the same per-frame signals: `kick` (1 on a
kick, decaying), `energy` (section level), `impact` (drop hit), `drop`, `phrase`, `beat`, `barphase`,
`bars`, `intensity`, the palette LUT, the live spectrum, the previous frame (for feedback), images.
"""

from __future__ import annotations

VERTEX = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

PRELUDE = """
#version 330
uniform vec2 res;
uniform float time;      // seconds since the track started
uniform float shot_t;    // seconds since this shot started
uniform float kick;      // 1.0 on a kick, exponential decay
uniform float energy;    // 0..1 smoothed section energy
uniform float impact;    // 1.0 on a drop hit / crash, ~1 s decay
uniform float drop;      // 1 in drops, climbing in builds, 0 elsewhere
uniform float phrase;    // 1 while a phrase is spoken
uniform float fillx;     // 1 during a fill
uniform float beat;      // 0..1 phase inside the beat
uniform float barphase;  // 0..1 phase inside the bar
uniform float bars;      // bars elapsed
uniform float rms;       // 0..1 loudness of this frame
uniform float intensity; // 0..1 from the shot's mood
uniform vec4 seedv;      // four random numbers in 0..1 per shot
uniform sampler2D palette;
uniform sampler2D spectrum;
uniform sampler2D prev;
uniform sampler2D img0;
uniform sampler2D img1;
uniform float imgmix;
in vec2 uv;
out vec4 frag;

vec3 pal(float v) { return texture(palette, vec2(clamp(v, 0.002, 0.998), 0.5)).rgb; }
float band(float i) { return texture(spectrum, vec2(clamp(i, 0.0, 1.0), 0.5)).r; }
float hash11(float n) { return fract(sin(n) * 43758.5453123); }
float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec2 hash22(vec2 p) { float n = hash21(p); return vec2(n, hash21(p + n + 0.37)); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1, 0)), c = hash21(i + vec2(0, 1)), d = hash21(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float n = i.x + i.y * 57.0 + i.z * 113.0;
    return mix(mix(mix(hash11(n), hash11(n + 1.0), f.x), mix(hash11(n + 57.0), hash11(n + 58.0), f.x), f.y),
               mix(mix(hash11(n + 113.0), hash11(n + 114.0), f.x), mix(hash11(n + 170.0), hash11(n + 171.0), f.x), f.y), f.z);
}
float fbm(vec2 p) { float v = 0.0, a = 0.5; mat2 m = mat2(1.6, 1.2, -1.2, 1.6); for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = m * p; a *= 0.5; } return v; }
float fbm3(vec3 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { v += a * vnoise3(p); p = p * 2.02 + vec3(1.7, 9.2, 3.1); a *= 0.5; } return v; }
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
vec2 centered() { return (uv * 2.0 - 1.0) * vec2(res.x / res.y, 1.0); }
"""

MAIN = """
void main() { frag = vec4(scene(), 1.0); }
"""

GENERATORS: dict[str, str] = {}

GENERATORS["plasma"] = """
vec3 scene() {
    vec2 p = centered();
    float t = shot_t * (0.5 + 0.9 * energy) + seedv.x * 40.0;
    float swirl = (0.15 + 0.35 * seedv.y) * sin(t * 0.3) * length(p) + kick * 0.25;
    p = rot(swirl) * p;
    float fx = 1.6 + 1.8 * seedv.z, fy = 1.6 + 1.8 * seedv.w;
    float v = sin(p.x * fx + t);
    v += sin(p.y * fy + t * 0.7);
    v += sin((p.x + p.y) * (1.2 + seedv.x) + t * 0.5);
    vec2 c = p + vec2(sin(t * 0.3), cos(t * 0.2));
    v += sin(length(c) * (2.0 + 2.5 * seedv.y + 2.0 * kick) - t * 1.5);
    float s = (v + 4.0) / 8.0;
    s = mix(s, fract(s * 1.5 + beat * 0.2), 0.35 * intensity * drop);
    return pal(s) * (0.85 + 0.35 * kick + 0.3 * impact);
}
"""

GENERATORS["waves"] = """
vec3 scene() {
    vec2 p = centered();
    float t = shot_t * (0.4 + 0.6 * energy);
    float total = 0.0, amp = 0.0;
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float ang = seedv.x * 3.14159 + fi * 0.9 + 0.2 * sin(t * 0.1 + fi);
        vec2 d = vec2(cos(ang), sin(ang));
        float freq = 4.0 + 9.0 * hash11(seedv.y * 10.0 + fi);
        float curve = 0.6 * hash11(seedv.z * 7.0 + fi) * (p.x * p.x - p.y * p.y);
        float a = 0.6 + 0.4 * hash11(seedv.w * 3.0 + fi);
        total += a * sin((dot(p, d) + curve) * freq + t * (1.0 + fi * 0.5));
        amp += a;
    }
    total += sin(length(p) * (9.0 + 4.0 * kick) - t * 2.0);
    amp += 1.0;
    float field = total / amp;
    float sharp = 1.0 + 1.6 * seedv.x + 1.5 * intensity * drop;
    float s = tanh(field * sharp) * 0.5 + 0.5;
    return pal(s) * (0.8 + 0.3 * kick + 0.3 * impact);
}
"""

GENERATORS["tunnel"] = """
vec3 scene() {
    vec2 p = centered();
    p += 0.12 * vec2(sin(shot_t * 0.7), cos(shot_t * 0.5)) * seedv.x;
    float r = length(p) + 0.001;
    float a = atan(p.y, p.x);
    float speed = 0.6 + 1.6 * energy + 0.8 * drop;
    float depth = 1.0 / r + shot_t * speed + seedv.y * 10.0;
    float twist = (seedv.z - 0.5) * 2.0;
    a += twist * shot_t * 0.3 + kick * 0.2;
    float ribs = floor(6.0 + seedv.w * 10.0);
    float rib = 0.5 + 0.5 * sin(a * ribs + depth * 2.0);
    float ring = 0.5 + 0.5 * sin(depth * 6.2831);
    float tex = fbm3(vec3(cos(a) * 2.0, sin(a) * 2.0, depth * 0.5));
    float mixn = 0.35 + 0.45 * seedv.x;
    float field = (1.0 - mixn) * (0.6 * rib + 0.4 * ring) + mixn * tex;
    float vig = pow(clamp(r * 1.15, 0.0, 1.0), 0.85);
    float s = clamp(field * vig, 0.0, 1.0);
    float pulse = 1.0 + 0.45 * kick * smoothstep(0.35, 0.0, r);
    return pal(s) * (0.85 + 0.3 * impact) * pulse;
}
"""

GENERATORS["domainwarp"] = """
vec3 scene() {
    vec2 p = centered() * (1.1 + 1.1 * seedv.x);
    float z = shot_t * (0.08 + 0.12 * energy) + seedv.y * 20.0;
    float warp = 1.8 + 2.0 * seedv.z + 0.6 * kick * intensity;
    float qx = fbm3(vec3(p, z));
    float qy = fbm3(vec3(p + vec2(5.2, 1.3), z + 7.0));
    float f = fbm3(vec3(p + warp * vec2(qx, qy), z + 13.0));
    float contrast = 0.9 + 0.6 * seedv.w;
    float s = clamp((f - 0.5) * contrast * 2.2 + 0.5, 0.0, 1.0);
    return pal(s) * (0.8 + 0.2 * energy + 0.25 * impact);
}
"""

GENERATORS["flow"] = """
vec3 scene() {
    vec2 p = centered();
    float z = shot_t * 0.15 + seedv.x * 30.0;
    float ang = fbm3(vec3(p * (1.2 + seedv.y), z)) * 12.566;
    vec2 vel = vec2(cos(ang), sin(ang)) * (0.0025 + 0.004 * energy + 0.006 * kick) * vec2(res.y / res.x, 1.0);
    vec3 carried = texture(prev, uv - vel).rgb * (0.955 + 0.03 * seedv.z);
    // new light: a few thousand drifting seeds, hashed per cell
    vec2 cellp = p * (18.0 + 22.0 * seedv.w);
    vec2 cell = floor(cellp);
    vec2 h = hash22(cell + seedv.xy);
    vec2 centre = cell + 0.5 + 0.35 * sin(shot_t * (0.5 + h) + h.yx * 6.2831);
    float d = length(cellp - centre);
    float spark = smoothstep(0.22, 0.0, d) * step(0.25 - 0.2 * energy, h.x);
    vec3 fresh = pal(fract(h.y + bars * 0.02)) * spark * (1.4 + 1.5 * kick);
    float glowfield = fbm3(vec3(p * 1.5, z)) * 0.18;                  // a faint moving bed so the frame is never black
    vec3 c = carried + fresh * 0.9 + pal(glowfield) * 0.12;
    return min(c, vec3(1.4));
}
"""

GENERATORS["kaleidoscope"] = """
vec3 scene() {
    vec2 p = centered();
    float n = floor(5.0 + seedv.x * 7.0);
    float a = atan(p.y, p.x) + shot_t * (0.15 + 0.35 * energy) * (seedv.y > 0.5 ? 1.0 : -1.0);
    float r = length(p);
    float seg = 6.2831 / n;
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r;
    q = rot(shot_t * 0.1) * q * (1.0 + 0.3 * kick);
    float f = fbm3(vec3(q * (2.0 + 2.0 * seedv.z), shot_t * 0.2 + seedv.w * 10.0));
    float rings = 0.5 + 0.5 * sin(r * (10.0 + 6.0 * seedv.x) - shot_t * 3.0 * (0.3 + drop));
    float s = clamp(f * 1.6 * (0.6 + 0.4 * rings) + 0.1 * impact, 0.0, 1.0);
    return pal(s) * (0.85 + 0.3 * kick + 0.4 * impact) * smoothstep(1.7, 0.6, r);
}
"""

GENERATORS["julia"] = """
vec3 scene() {
    vec2 p = centered() * (1.4 - 0.5 * seedv.x - 0.15 * kick * intensity);
    p = rot(shot_t * 0.07) * p;
    float ang = shot_t * (0.04 + 0.05 * energy) + seedv.y * 6.2831;
    vec2 c = vec2(-0.75 + 0.25 * seedv.z, 0.05 + 0.2 * seedv.w) + 0.12 * vec2(cos(ang), sin(ang * 1.3));
    vec2 z = p;
    float n = 0.0;
    const int MAXI = 80;
    for (int i = 0; i < MAXI; i++) {
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (dot(z, z) > 64.0) break;
        n += 1.0;
    }
    float smoothn = n - log2(max(1.0, log2(max(dot(z, z), 1.0))));
    float s = fract(smoothn * 0.04 + shot_t * 0.05 + 0.15 * impact);
    float inside = step(float(MAXI) - 0.5, n);
    return mix(pal(s), pal(0.05) * 0.4, inside) * (0.85 + 0.35 * kick + 0.3 * impact);
}
"""

GENERATORS["mandelbulb"] = """
float bulb(vec3 pos, float power) {
    vec3 z = pos; float dr = 1.0, r = 0.0;
    for (int i = 0; i < 8; i++) {
        r = length(z); if (r > 2.0) break;
        float theta = acos(z.z / r) * power, phi = atan(z.y, z.x) * power;
        float zr = pow(r, power);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) + pos;
    }
    return 0.5 * log(r) * r / dr;
}
vec3 scene() {
    vec2 p = centered();
    float power = 6.0 + 4.0 * seedv.x + 1.5 * sin(shot_t * 0.1) + 0.6 * kick * intensity;
    float t = shot_t * (0.15 + 0.2 * energy);
    vec3 ro = vec3(2.6 * cos(t), 1.1 * sin(t * 0.7), 2.6 * sin(t));
    vec3 ta = vec3(0.0);
    vec3 fw = normalize(ta - ro), rt = normalize(cross(vec3(0, 1, 0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(p.x * rt + p.y * up + 1.8 * fw);
    float dist = 0.0, d = 0.0; int steps = 0;
    for (int i = 0; i < 64; i++) {
        vec3 pos = ro + rd * dist;
        d = bulb(pos, power);
        if (d < 0.0015 * dist || dist > 8.0) break;
        dist += d * 0.8; steps = i;
    }
    if (dist > 8.0) {
        float bg = fbm(p * 2.0 + t) * 0.25;
        return pal(bg * 0.4) * 0.35;
    }
    float ao = 1.0 - float(steps) / 64.0;
    float glow = pow(ao, 1.5);
    float s = fract(dist * 0.25 + glow * 0.6 + seedv.y + 0.1 * impact);
    return pal(s) * (0.4 + 1.2 * glow) * (0.9 + 0.4 * kick + 0.3 * impact);
}
"""

GENERATORS["menger"] = """
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float menger(vec3 p) {
    float d = sdBox(p, vec3(1.0));
    float s = 1.0;
    for (int m = 0; m < 4; m++) {
        vec3 a = mod(p * s, 2.0) - 1.0;
        s *= 3.0;
        vec3 r = abs(1.0 - 3.0 * abs(a));
        float da = max(r.x, r.y), db = max(r.y, r.z), dc = max(r.z, r.x);
        float c = (min(da, min(db, dc)) - 1.0) / s;
        d = max(d, c);
    }
    return d;
}
vec3 scene() {
    vec2 p = centered();
    float t = shot_t * (0.2 + 0.25 * energy);
    vec3 ro = vec3(0.0, 0.0, -2.2 + 0.15 * sin(t)) ;
    ro.xy += 0.35 * vec2(sin(t * 0.6), cos(t * 0.4));
    vec3 rd = normalize(vec3(p, 1.6));
    rd.xy = rot(t * 0.25 + seedv.x) * rd.xy;
    rd.xz = rot(0.5 * sin(t * 0.3) + seedv.y) * rd.xz;
    float dist = 0.0, d = 0.0; int steps = 0;
    for (int i = 0; i < 80; i++) {
        vec3 pos = ro + rd * dist;
        pos = mod(pos + 1.0, 2.0) - 1.0;      // repeat the sponge: an endless corridor
        d = menger(pos);
        if (d < 0.001 || dist > 12.0) break;
        dist += d; steps = i;
    }
    float ao = 1.0 - float(steps) / 80.0;
    float fog = exp(-dist * 0.22);
    float s = fract(dist * 0.15 + seedv.z + 0.4 * ao + 0.1 * impact);
    return pal(s) * fog * (0.5 + 1.0 * ao) * (0.9 + 0.4 * kick + 0.3 * impact);
}
"""

GENERATORS["metaballs"] = """
vec3 scene() {
    vec2 p = centered();
    float t = shot_t * (0.5 + 0.6 * energy);
    float field = 0.0;
    for (int i = 0; i < 7; i++) {
        float fi = float(i);
        vec2 h = hash22(vec2(fi, seedv.x * 9.0));
        vec2 c = vec2(sin(t * (0.3 + 0.4 * h.x) + fi * 1.7), cos(t * (0.25 + 0.4 * h.y) + fi * 2.3)) * (0.6 + 0.5 * h.x);
        float rad = 0.12 + 0.12 * h.y + 0.12 * kick * intensity;
        field += rad * rad / (dot(p - c, p - c) + 0.002);
    }
    float s = clamp(field * 0.35, 0.0, 1.0);
    float edge = smoothstep(0.55, 0.75, s);
    return mix(pal(s * 0.6), pal(0.85 + 0.15 * sin(t)), edge) * (0.8 + 0.4 * kick + 0.3 * impact);
}
"""

GENERATORS["particles"] = """
vec3 scene() {
    vec2 p = centered();
    vec3 c = vec3(0.0);
    for (int layer = 0; layer < 3; layer++) {
        float fl = float(layer);
        float scale = 6.0 + fl * 5.0 + 6.0 * seedv.x;
        vec2 q = p * scale + vec2(shot_t * (0.15 + 0.35 * energy) * (fl + 1.0), shot_t * 0.1 * (0.5 - seedv.y));
        vec2 cell = floor(q), f = fract(q);
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            vec2 o = vec2(float(i), float(j));
            vec2 h = hash22(cell + o + fl * 17.0 + seedv.zw * 13.0);
            vec2 centre = o + 0.5 + 0.4 * sin(shot_t * (0.4 + h) + h.yx * 6.2831);
            float d = length(f - centre);
            float size = 0.05 + 0.08 * h.x + 0.1 * kick * intensity;
            float glow = size / (d * d * 25.0 + size);
            c += pal(fract(h.y + fl * 0.3 + bars * 0.01)) * glow * (0.5 + 0.7 * h.x);
        }
    }
    c *= 0.7 + 0.5 * energy + 0.4 * impact;
    c += pal(fbm(p * 1.2 + shot_t * 0.05) * 0.5) * 0.18;                 // nebula bed behind the particles
    return min(c, vec3(1.3));
}
"""

GENERATORS["spectrum"] = """
vec3 scene() {
    vec2 p = centered();
    float r = length(p), a = atan(p.y, p.x);
    float n = 64.0;
    float idx = fract((a / 6.2831) + 0.5 + shot_t * 0.02);
    float b = band(idx);
    float b2 = band(fract(idx + 0.5));
    float ring = 0.32 + 0.12 * rms + 0.08 * kick;
    float bar = smoothstep(ring + b * 0.55 + 0.01, ring + b * 0.55 - 0.01, r) * step(ring, r);
    float inner = smoothstep(ring, ring - 0.02, r);
    float core = smoothstep(0.2 + 0.1 * kick, 0.0, r) * (0.4 + 0.6 * rms);
    vec3 c = pal(0.25 + 0.7 * b) * bar * (0.7 + 0.5 * b);
    c += pal(0.1 + 0.3 * b2) * inner * 0.45 * (0.5 + kick);
    c += pal(0.9) * core;
    float mirror = smoothstep(0.02, 0.0, abs(r - (ring + b2 * 0.55 + 0.15)));
    c += pal(0.6) * mirror * 0.6 * intensity;
    float bg = fbm(p * 1.5 + shot_t * 0.1) * 0.15;
    c += pal(bg) * 0.25;
    return c * (0.9 + 0.4 * impact);
}
"""

GENERATORS["stills"] = """
vec2 kenburns(vec2 st, vec4 h, float t) {
    float zoom = mix(0.92, 0.66, t);
    vec2 centre = mix(h.xy, h.zw, t) * (1.0 - zoom) + zoom * 0.5;
    return centre + (st - 0.5) * zoom;
}
vec3 scene() {
    float tt = clamp(imgmix, 0.0, 1.0);
    float e = tt * tt * (3.0 - 2.0 * tt);
    vec2 st = uv;
    float strength = (2.0 + 6.0 * seedv.w + 4.0 * kick * intensity) / res.y;
    vec2 disp = vec2(fbm3(vec3(st * 4.0, shot_t * 0.3)) - 0.5, fbm3(vec3(st * 4.0 + 31.7, shot_t * 0.3 + 5.0)) - 0.5);
    st += disp * strength * 60.0;
    vec4 h0 = vec4(seedv.x, seedv.y, seedv.z, seedv.w);
    vec4 h1 = vec4(seedv.z, seedv.w, seedv.x, seedv.y);
    vec3 a = texture(img0, clamp(kenburns(st, h0, fract(shot_t * 0.02 + 0.3 * e)), 0.001, 0.999)).rgb;
    vec3 b = texture(img1, clamp(kenburns(st, h1, fract(shot_t * 0.02 + 0.5)), 0.001, 0.999)).rgb;
    vec3 c = mix(a, b, e);
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(c, pal(lum) , 0.25 * intensity);          // a wash of the track's palette
    return c * (0.85 + 0.15 * energy + 0.35 * impact + 0.15 * kick);
}
"""

# ---- reaction-diffusion: a simulation pass (ping-pong) and a display pass ---------------------

REACTION_SIM = """
#version 330
uniform sampler2D state;
uniform vec2 texel;
uniform float feed;
uniform float kill;
uniform float kick;
in vec2 uv;
out vec4 frag;
void main() {
    vec2 c = texture(state, uv).rg;
    vec2 lap = 0.2 * (texture(state, uv + vec2(texel.x, 0)).rg + texture(state, uv - vec2(texel.x, 0)).rg
                    + texture(state, uv + vec2(0, texel.y)).rg + texture(state, uv - vec2(0, texel.y)).rg)
             + 0.05 * (texture(state, uv + texel).rg + texture(state, uv - texel).rg
                    + texture(state, uv + vec2(texel.x, -texel.y)).rg + texture(state, uv + vec2(-texel.x, texel.y)).rg)
             - c;
    float a = c.r, b = c.g;
    float react = a * b * b;
    float na = a + (1.0 * lap.r - react + feed * (1.0 - a));
    float nb = b + (0.5 * lap.g + react - (kill + feed) * b);
    frag = vec4(clamp(na, 0.0, 1.0), clamp(nb, 0.0, 1.0), 0.0, 1.0);
}
"""

GENERATORS["reaction"] = """
vec3 scene() {
    vec2 st = uv;
    float b = texture(prev, st).g;                     // 'prev' carries the simulation state here
    float s = clamp(b * 3.2 + 0.05 * kick, 0.0, 1.0);
    vec2 p = centered();
    float edge = length(vec2(dFdx(b), dFdy(b))) * 8.0;
    float bed = fbm3(vec3(p * 1.3, shot_t * 0.05)) * 0.22;               // the dish is never pitch black
    return pal(max(s, bed)) * (0.9 + 0.25 * impact) + pal(0.95) * edge * 0.7 * intensity;
}
"""

POST = """
#version 330
uniform sampler2D texA;
uniform sampler2D texB;
uniform float mixv;
uniform float impact;
uniform float kick;
uniform float phrase;
uniform vec2 res;
in vec2 uv;
out vec4 frag;
void main() {
    vec2 st = vec2(uv.x, 1.0 - uv.y);          // flipped so the read-back is top-down for ffmpeg
    vec2 c = st - 0.5;
    float shift = (0.0025 * kick + 0.006 * impact) * length(c) * 2.0;
    vec3 a = vec3(texture(texA, st + vec2(shift, 0.0)).r, texture(texA, st).g, texture(texA, st - vec2(shift, 0.0)).b);
    vec3 b = vec3(texture(texB, st + vec2(shift, 0.0)).r, texture(texB, st).g, texture(texB, st - vec2(shift, 0.0)).b);
    vec3 col = mix(a, b, clamp(mixv, 0.0, 1.0));
    float vig = 1.0 - 0.55 * pow(length(c) * 1.35, 2.2);
    col *= clamp(vig, 0.0, 1.0);
    col += vec3(0.9, 0.95, 1.0) * impact * impact * 0.35;                 // the flash of the drop
    col = mix(col, col * vec3(1.0, 0.97, 0.92) + 0.03, phrase * 0.5);      // a warm lift while someone speaks
    col = col / (1.0 + col * 0.15);
    frag = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def fragment_source(name: str) -> str:
    return PRELUDE + GENERATORS[name] + MAIN


__all__ = ["GENERATORS", "MAIN", "POST", "PRELUDE", "REACTION_SIM", "VERTEX", "fragment_source"]
