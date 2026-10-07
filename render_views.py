"""Render a labelled mesh from four sides with walls facing the camera, as one 2x2 image.

The default ChimeraX view looks down the tomogram z axis, where mostly vertical membranes
are seen edge-on. This tilts 90° about x (tomogram z becomes screen-up), turns 90° about the
vertical between shots, and tiles the four shots under a title and colour key, or with
--separate N saves the N views showing the most surface as separate images.

Usage: python render_views.py MESH_DIR OUT_PNG [--title TEXT] [--separate N]
    MESH_DIR: folder with view_chimerax.py and names.cxc (project_mesh.py / umap_hdbscan.py)
Run with: source /programs/sbgrid.shrc; DISPLAY=:10 (offscreen rendering is unavailable here)
"""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def crop(im, pad=30):
    """Trim the white background around the mesh."""
    a = np.asarray(im)
    ys, xs = np.nonzero((a < 245).any(axis=2))
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                    min(xs.max() + pad, im.width), min(ys.max() + pad, im.height)))


def header(im, title, colors, labels):
    """Put a title and a colour key above an image."""
    from PIL import ImageDraw, ImageFont
    try:
        font, small = ImageFont.truetype("DejaVuSans.ttf", 44), ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        font = small = ImageFont.load_default()
    probe = ImageDraw.Draw(im)
    key_w = 30 + sum(52 + int(probe.textlength(l, font=small)) + 50 for l in labels)
    title_w = 60 + int(probe.textlength(title, font=font))
    out = Image.new("RGB", (max(im.width, key_w, title_w), im.height + 140), "white")
    out.paste(im, ((out.width - im.width) // 2, 140))
    draw = ImageDraw.Draw(out)
    draw.text((30, 20), title, fill="black", font=font)
    x = 30
    for c, l in zip(colors, labels):
        draw.rectangle([x, 82, x + 40, 122], fill=c)
        draw.text((x + 52, 82), l, fill="#333333", font=small)
        x += 52 + int(draw.textlength(l, font=small)) + 50
    return out


def run_chimerax(mesh_dir, cmds):
    """Run commands after mesh_dir/view_chimerax.py; ChimeraX needs the xrdp display here."""
    with tempfile.TemporaryDirectory() as tmp:
        cxc = Path(tmp) / "render.cxc"
        cxc.write_text("\n".join(cmds + ["exit"]) + "\n")
        env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":10")}
        subprocess.run(["chimerax", str(mesh_dir / "view_chimerax.py"), str(cxc)],
                       cwd=mesh_dir, env=env, check=True, capture_output=True, timeout=600)


def overview_focus(args, colors, labels):
    """Whole mesh from above, and/or a close-up of chosen models with the rest hidden."""
    setup = [f"open {args.mesh_dir / 'names.cxc'}", "2dlabels delete", "windowsize 1300 950",
             "lighting soft", "graphics silhouettes true width 1.5"]
    outs, cmds = [], list(setup)
    if args.overview:
        out = args.out.with_name(f"{args.out.stem}_overview.png")
        cmds += ["view", f"save {out} supersample 3"]
        outs.append((out, args.title, colors, labels))
    if args.focus:
        lo, _, hi = args.focus.partition("-")
        keep = set(range(int(lo), int(hi or lo) + 1))
        hide = [] if args.keep_others else [i for i in range(1, len(labels) + 1) if i not in keep]
        out = args.out.with_name(f"{args.out.stem}_focus.png")
        target = f"#{','.join(map(str, sorted(keep)))}"
        cmds += ([f"hide #{','.join(map(str, hide))} models"] if hide else []) + [
            f"view {target}", "turn x -45", f"view {target}", f"save {out} supersample 3"]
        shown = sorted(keep) if hide else range(1, len(labels) + 1)
        note = "close-up, other clusters hidden" if hide else "close-up"
        outs.append((out, f"{args.title} ({note})",
                     [colors[i - 1] for i in shown], [labels[i - 1] for i in shown]))
    run_chimerax(args.mesh_dir, cmds)
    for out, title, c, l in outs:
        header(crop(Image.open(out).convert("RGB")), title, c, l).save(out)
        print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mesh_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--separate", type=int, default=0,
                        help="instead of a 2x2 sheet, save the N views showing the most surface as "
                             "separate images OUT_<angle>.png")
    parser.add_argument("--overview", action="store_true",
                        help="save OUT_overview.png: the whole mesh from above, as ChimeraX opens it")
    parser.add_argument("--focus", default="",
                        help="save OUT_focus.png: only these models (e.g. 2-4), framed close and "
                             "tilted 45 degrees; the others are hidden so they don't block the view")
    parser.add_argument("--keep-others", action="store_true",
                        help="with --focus: frame those models but leave the other clusters shown")
    args = parser.parse_args()
    args.mesh_dir = args.mesh_dir.resolve()
    args.out = args.out.resolve()

    names = (args.mesh_dir / "names.cxc").read_text()
    labels = re.findall(r'rename #\d+ "([^"]+)"', names)
    colors = re.search(r"colors = (\(.*?\))", (args.mesh_dir / "view_chimerax.py").read_text()).group(1)
    colors = eval(colors)  # tuple literal written by project_mesh.CHIMERAX_SCRIPT

    if args.overview or args.focus:
        overview_focus(args, colors, labels)
        return

    with tempfile.TemporaryDirectory() as tmp:
        shots = [Path(tmp) / f"v{i}.png" for i in range(4)]
        cmds = [f"open {args.mesh_dir / 'names.cxc'}", "2dlabels delete", "windowsize 1100 850",
                "turn x 90", "view", "lighting soft", "graphics silhouettes true width 1.5"]
        for i, shot in enumerate(shots):
            cmds += ["view", f"save {shot} supersample 3", "turn y 90"]
        cmds += ["exit"]
        cxc = Path(tmp) / "render.cxc"
        cxc.write_text("\n".join(cmds) + "\n")
        env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":10")}
        subprocess.run(["chimerax", str(args.mesh_dir / "view_chimerax.py"), str(cxc)],
                       cwd=args.mesh_dir, env=env, check=True, capture_output=True, timeout=600)
        raw = [Image.open(s).convert("RGB") for s in shots]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.separate:
        # Views with the most non-background pixels show the most surface.
        cover = [int((np.asarray(im) < 245).any(axis=2).sum()) for im in raw]
        for i in sorted(np.argsort(cover)[::-1][:args.separate]):
            out = args.out.with_name(f"{args.out.stem}_{90 * i}.png")
            header(crop(raw[i]), args.title, colors, labels).save(out)
            print(f"saved {out}")
        return

    ims = [crop(im) for im in raw]
    w, h = max(im.width for im in ims), max(im.height for im in ims)
    grid = Image.new("RGB", (2 * w, 2 * h), "white")
    for i, im in enumerate(ims):
        grid.paste(im, ((i % 2) * w, (i // 2) * h))
    sheet = header(grid, args.title, colors, labels)
    sheet.thumbnail((2400, 2400))
    sheet.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
