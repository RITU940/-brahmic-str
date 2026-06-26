"""
Method/architecture diagram (paper Fig. for Sec. 3), drawn with PIL so
Bengali conjuncts shape correctly (libraqm). Uses the REAL tokenizations
from method_fig_tokens.json and the real test crop for the example word.
Output: figures/fig0_method.png (300 dpi).
"""
import os, json, glob
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(BASE, 'method_fig_tokens.json')))
WORD, BPE, GRA = T['word'], T['bpe'], T['graphemes']

W, H = 2750, 1000
img = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(img)

beng = glob.glob('/usr/share/fonts/truetype/noto/NotoSerifBengali-Regular.ttf') or \
       glob.glob('/usr/share/fonts/truetype/noto/NotoSerifBengali*.ttf')
F_BN = ImageFont.truetype(beng[0], 40)
F_BN_S = ImageFont.truetype(beng[0], 33)
DEJA = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
DEJA_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F = ImageFont.truetype(DEJA, 30)
F_B = ImageFont.truetype(DEJA_B, 32)
F_S = ImageFont.truetype(DEJA, 25)

BLUE, ORANGE, RED, GREEN, GREY = '#4878CF', '#EE854A', '#D65F5F', '#2e8b57', '#666666'


def box(x, y, w, h, color, fill='white', width=4, r=18):
    d.rounded_rectangle([x, y, x + w, y + h], radius=r, outline=color,
                        fill=fill, width=width)


def ctext(cx, cy, s, font, fill='black'):
    b = d.textbbox((0, 0), s, font=font)
    d.text((cx - (b[2] - b[0]) / 2 - b[0], cy - (b[3] - b[1]) / 2 - b[1]), s,
           font=font, fill=fill)


def arrow(x1, y1, x2, y2, color=GREY, width=5):
    d.line([x1, y1, x2, y2], fill=color, width=width)
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    L, sp = 18, 0.45
    d.polygon([(x2, y2),
               (x2 - L * math.cos(a - sp), y2 - L * math.sin(a - sp)),
               (x2 - L * math.cos(a + sp), y2 - L * math.sin(a + sp))],
              fill=color)


def token_row(x, y, toks, color, font):
    """Draw tokens as adjacent boxes; return end x."""
    for t in toks:
        b = d.textbbox((0, 0), t, font=font)
        tw = max(b[2] - b[0] + 26, 46)
        box(x, y, tw, 64, color, width=3, r=8)
        ctext(x + tw / 2, y + 30, t, font, fill='black')
        x += tw + 8
    return x


# ---- input image crop -------------------------------------------------
import sys
sys.path.insert(0, BASE)
from metrics import normalize_bengali as N
sp = json.load(open(os.path.join(BASE, 'florence2_splits.json')))['test']
crop_path = next(r['image'] for r in sp if N(r['gt']) == N(WORD))
crop = Image.open(crop_path).convert('RGB')
crop.thumbnail((360, 170))
cx0, cy0 = 60, H // 2 - crop.height // 2 - 40
box(cx0 - 14, cy0 - 14, crop.width + 28, crop.height + 28, GREY, width=3, r=10)
img.paste(crop, (cx0, cy0))
ctext(cx0 + crop.width / 2, cy0 + crop.height + 45, 'scene-text crop', F_S, GREY)

# ---- encoder box -------------------------------------------------------
ex, ey, ew, eh = 520, H // 2 - 130 - 40, 330, 260
box(ex, ey, ew, eh, '#333333')
ctext(ex + ew / 2, ey + 60, 'Florence-2', F_B)
ctext(ex + ew / 2, ey + 110, 'vision encoder', F)
ctext(ex + ew / 2, ey + 160, '+ LoRA adapters', F)
ctext(ex + ew / 2, ey + 205, '(rank 16)', F_S, GREY)
arrow(cx0 + crop.width + 30, H // 2 - 40, ex - 14, H // 2 - 40)

# ---- two branches ------------------------------------------------------
bx = ex + ew + 90
by_top, by_bot = 95, 555
bw, bh = 900, 320

box(bx, by_top, bw, bh, BLUE)
ctext(bx + 250, by_top + 45, 'BPE decoder', F_B, BLUE)
ctext(bx + 640, by_top + 45, '(Banglish BPE, 71k vocab)', F_S, GREY)
end = token_row(bx + 35, by_top + 95, BPE, BLUE, F_BN_S)
ctext(bx + 240, by_top + 215, f'{len(BPE)} fragments — vowel signs', F_S, GREY)
ctext(bx + 240, by_top + 250, 'and conjuncts are torn apart', F_S, GREY)
d.text((bx + 580, by_top + 195), 'output:', font=F_S, fill='#c0392b')
d.text((bx + 685, by_top + 178), T['bpe_pred'], font=F_BN_S, fill='#c0392b')
d.text((bx + 845, by_top + 195), '✗', font=F_S, fill='#c0392b')
d.text((bx + 600, by_top + 225), f"conf. c₁ = {T['c1']:.2f}", font=F, fill=BLUE)

box(bx, by_bot, bw, bh, ORANGE)
ctext(bx + 290, by_bot + 45, 'Grapheme decoder', F_B, ORANGE)
ctext(bx + 680, by_bot + 45, '(+640 cluster tokens)', F_S, GREY)
token_row(bx + 35, by_bot + 95, GRA, ORANGE, F_BN)
ctext(bx + 245, by_bot + 220, f'{len(GRA)} grapheme clusters —', F_S, GREY)
ctext(bx + 245, by_bot + 255, 'matches what the image shows', F_S, GREY)
d.text((bx + 600, by_bot + 235), f"conf. c₂ = {T['c2']:.2f}", font=F, fill=ORANGE)

arrow(ex + ew + 14, H // 2 - 90, bx - 14, by_top + bh / 2)
arrow(ex + ew + 14, H // 2 + 10, bx - 14, by_bot + bh / 2)

# ---- fusion ------------------------------------------------------------
fx, fy, fw, fh = bx + bw + 90, H // 2 - 110 - 40, 360, 220
box(fx, fy, fw, fh, RED, width=5)
ctext(fx + fw / 2, fy + 55, 'max-confidence', F_B, RED)
ctext(fx + fw / 2, fy + 105, 'fusion', F_B, RED)
ctext(fx + fw / 2, fy + 165, 'argmax(c₁, c₂)', F, GREY)
arrow(bx + bw + 14, by_top + bh / 2, fx - 14, fy + fh / 2 - 35)
arrow(bx + bw + 14, by_bot + bh / 2, fx - 14, fy + fh / 2 + 35)

# ---- output ------------------------------------------------------------
ox = fx + fw + 80
arrow(fx + fw + 14, fy + fh / 2, ox - 10, fy + fh / 2)
b = d.textbbox((0, 0), WORD, font=F_BN)
box(ox, fy + fh / 2 - 55, b[2] - b[0] + 60, 110, GREEN, width=5)
ctext(ox + (b[2] - b[0] + 60) / 2, fy + fh / 2 - 8, WORD, F_BN, GREEN)
ctext(ox + (b[2] - b[0] + 60) / 2, fy + fh / 2 + 85, '✓ correct', F_S, GREEN)

# ---- training-time extras (dashed) --------------------------------------
ty = 925
d.text((520, ty - 22), 'training only:', font=F_S, fill=GREY)
for i, lab in enumerate(['grapheme-rarity-weighted synthetic curriculum',
                         'confidence-filtered self-training (unlabeled crops)']):
    x0 = 760 + i * 800
    d.rounded_rectangle([x0, ty - 35, x0 + 740, ty + 25], radius=12,
                        outline=GREY, width=2)
    ctext(x0 + 370, ty - 5, lab, F_S, GREY)

img.save(os.path.join(BASE, 'figures', 'fig0_method.png'), dpi=(300, 300))
print('saved figures/fig0_method.png')
