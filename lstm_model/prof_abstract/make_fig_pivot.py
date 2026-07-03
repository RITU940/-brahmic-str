# Renders fig_pivot.png — the shared abugida pivot space diagram.
# Run with the conda env python (has PIL+libraqm for correct Indic shaping).
from PIL import Image, ImageDraw, ImageFont

W, H = 3000, 1210
INK, SEC, MUT = (11, 11, 11), (82, 81, 78), (137, 135, 129)
BLUE, BLUE_L, GRID = (42, 120, 214), (205, 226, 251), (225, 224, 217)
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

DEJA = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJA_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F = {
    "Devanagari": "/usr/share/fonts/truetype/Gargi/Gargi.ttf",
    "Bengali": "/usr/share/fonts/truetype/fonts-beng-extra/MuktiNarrow.ttf",
    "Tamil": "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
    "Telugu": "/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf",
    "Kannada": "/usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf",
    "Malayalam": "/usr/share/fonts/truetype/malayalam/Meera-Regular.ttf",
}
def font(path, size): return ImageFont.truetype(path, size)
def text(xy, s, fp, size, fill, anchor="la"):
    d.text(xy, s, font=font(fp, size), fill=fill, anchor=anchor)

def arrow(x0, y0, x1, y1, color=MUT, w=5, head=22):
    d.line([x0, y0, x1, y1], fill=color, width=w)
    import math
    a = math.atan2(y1 - y0, x1 - x0)
    for da in (2.6, -2.6):
        d.line([x1, y1, x1 - head * math.cos(a + da), y1 - head * math.sin(a + da)],
               fill=color, width=w)

# ── Band 1: the letter KA across six blocks -> one pivot codepoint ──────────
scripts = [
    ("Devanagari", "क", "U+0915"), ("Bengali", "ক", "U+0995"),
    ("Tamil", "க", "U+0B95"), ("Telugu", "క", "U+0C15"),
    ("Kannada", "ಕ", "U+0C95"), ("Malayalam", "ക", "U+0D15"),
]
bw, bh, gap, y0 = 350, 330, 60, 90
x0 = (W - 6 * bw - 5 * gap) // 2
text((x0, 18), "The letter KA sits at the SAME offset (+0x15) inside every Brahmic Unicode block:",
     DEJA_B, 44, INK)
centers = []
for i, (name, glyph, cp) in enumerate(scripts):
    x = x0 + i * (bw + gap)
    d.rounded_rectangle([x, y0, x + bw, y0 + bh], radius=18, outline=GRID, width=4)
    text((x + bw / 2, y0 + 36), name, DEJA, 38, SEC, anchor="ma")
    text((x + bw / 2, y0 + 175), glyph, F[name], 150, INK, anchor="mm")
    text((x + bw / 2, y0 + bh - 52), cp, DEJA, 34, MUT, anchor="ma")
    centers.append(x + bw / 2)

py0, pw, ph = y0 + bh + 160, 760, 200
px = (W - pw) // 2
for cx in centers:
    arrow(cx, y0 + bh + 12, px + pw / 2 + (cx - W / 2) * 0.22, py0 - 14)
d.rounded_rectangle([px, py0, px + pw, py0 + ph], radius=18, fill=BLUE_L,
                    outline=BLUE, width=6)
text((px + pw / 2, py0 + 58), "क", F["Devanagari"], 120, INK, anchor="mm")
text((px + pw / 2, py0 + 128), "one shared pivot code  (0x0900 + 0x15)", DEJA_B, 40,
     (24, 79, 149), anchor="ma")

# ── Band 2: whole words round-trip losslessly ───────────────────────────────
y2 = py0 + ph + 120
d.line([120, y2 - 55, W - 120, y2 - 55], fill=GRID, width=3)
text((120, y2 - 20), "Whole words map in and out losslessly (verified on the full benchmark):",
     DEJA_B, 44, INK)
yw = y2 + 90
items = [("తెలుగు", F["Telugu"]), ("తెలుగు→", None), ]
# Telugu word -> pivot
text((330, yw + 40), "తెలుగు", F["Telugu"], 100, INK, anchor="mm")
text((330, yw + 128), "Telugu", DEJA, 34, MUT, anchor="ma")
arrow(560, yw + 40, 760, yw + 40)
text((660, yw - 30), "to pivot", DEJA, 32, MUT, anchor="ma")
text((980, yw + 40), "तॆलुगु", F["Devanagari"], 100, (24, 79, 149), anchor="mm")
text((980, yw + 128), "shared pivot space", DEJA, 34, MUT, anchor="ma")
# Tamil word round trip
text((1650, yw + 40), "தமிழ்", F["Tamil"], 100, INK, anchor="mm")
text((1650, yw + 128), "Tamil", DEJA, 34, MUT, anchor="ma")
arrow(1880, yw + 40, 2060, yw + 40)
text((1970, yw - 30), "to pivot", DEJA, 32, MUT, anchor="ma")
text((2230, yw + 40), "तमिऴ्", F["Devanagari"], 100, (24, 79, 149), anchor="mm")
arrow(2410, yw + 40, 2590, yw + 40)
text((2500, yw - 30), "back", DEJA, 32, MUT, anchor="ma")
text((2760, yw + 40), "தமிழ்", F["Tamil"], 100, INK, anchor="mm")
text((2760, yw + 128), "identical  ✓", DEJA_B, 36, (0, 99, 0), anchor="ma")

img.save("/c/ujjwalb/ritu1/lstm_model/prof_abstract/fig_pivot.png")
print("fig_pivot.png", img.size)
