"""Compose paper_demo/figs/fig1_composite.png from Replay-mode screenshots.

Screenshots (paper_demo/figs/shots/*.png) are taken from the built frontend in Replay mode,
which makes no model or network calls; see README "Replay mode".
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(__file__).resolve().parent.parent / "paper_demo" / "figs"
S = D / "shots"
sup, wh, au = (Image.open(S / f"{n}.png").convert("RGB") for n in ("supported", "withheld", "audit"))
K, KA = 2800 / 1217, 2800 / 1990  # CSS px -> screenshot px
W, G, BAR = 1800, 36, 78
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 46)


def crop(im, box, f):
    return im.crop(tuple(round(v * f) for v in box))


def panel(im, title):
    im = im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
    out = Image.new("RGB", (W, im.height + BAR), "white")
    dr = ImageDraw.Draw(out)
    dr.rectangle((0, 0, W, BAR - 10), fill=(30, 41, 59))
    dr.text((18, 12), title, fill="white", font=FONT)
    out.paste(im, (0, BAR))
    return out


def column(ps):
    out = Image.new("RGB", (W, sum(p.height for p in ps) + G * (len(ps) - 1)), "white")
    y = 0
    for p in ps:
        out.paste(p, (0, y))
        y += p.height + G
    return out


left = column([
    panel(crop(wh, (405, 493, 1196, 1034), K), "(a) Tempting movement, no verifiable quote: withheld"),
    panel(crop(sup, (405, 1100, 1196, 1354), K), "(c) Trace timeline of the supported case"),
])
right = column([
    panel(crop(au, (186, 230, 1805, 1240), KA), "(e) Audit page: outcome counts, one expanded entry"),
    panel(crop(sup, (405, 770, 1196, 872), K), "(b) Supported movement: verifier quote"),
    panel(crop(sup, (405, 1372, 1196, 1660), K), "(d) Evidence pool: kept abstract, quote highlighted"),
])
out = Image.new("RGB", (2 * W + G, max(left.height, right.height)), "white")
out.paste(left, (0, 0))
out.paste(right, (W + G, 0))
out.save(D / "fig1_composite.png", optimize=True)
print(out.size)
