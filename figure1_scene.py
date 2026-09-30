"""
Manim clips animating Figure 1 of "Scaled Hippocampus-inspired Neural Networks
on Neuromorphic Memristive Hardware" (Kilgore, Kopsick, Ahmed, Ascoli, Adam).

Paper flow (panels a -> d): full-scale CA3 (8 types) -> scale network size ->
reduced-scale CA3 (still 8 types) -> scale neuron diversity -> 3-type CA3
(Pyramidal / Basket / MFA ORDEN) -> implementation -> FPGA/memristor hardware.
This adds one more beat after panel d: fade the 3-type network back out and
bring the full-scale network back in, pointing at the hardware, to close the
loop from "biological inspiration" to "what the hardware approximates".

    Clip1  full-scale CA3 network (89,226 neurons, 8 types)
    Clip2  "Scale network size": full-scale -> reduced-scale (still 8 types)
    Clip3  "Scale neuron diversity": reduced-scale (8 types) -> 3 types
    Clip4  3 types -> mapped onto a 25x25 memristor crossbar tile
    Clip5  close the loop: full-scale network pointing at the hardware

Real numbers used (from the paper text / this repo's data):
    full-scale:  89,226 neurons, 250,078,223 synapses, 8 neuron types
    small-scale: 179 neurons, 17,996 synapses, 3 neuron types
                 (data/CA3_0_0001_3_group_18: Pyramidal=160, MFA_ORDEN=12,
                  Basket=6 spiking neurons -> drawn as literal dot counts)
    hardware:    FPGA host + mixed-signal daughterboard, 18,316 memristors
                 used out of a 20,000-memristor array

Render:
    manim -ql figure1_scene.py Clip1 Clip2 Clip3 Clip4 Clip5
    manim -qh figure1_scene.py Clip1 Clip2 Clip3 Clip4 Clip5   # 1080p
"""

from pathlib import Path

import numpy as np
from manim import *

from raster_scene import NEURON_COLORS
from greedy_blocking import greedy_blocking
from greedy_scene import first_fit

config.background_color = WHITE
Text.set_default(font="Segoe UI", color=BLACK)

MUTED = GREY_D
EXC = "#1D3E8C"          # excitatory (pyramidal) connections - blue, like Fig 1
INH = "#D6A419"          # inhibitory connections - gold, like Fig 1
ARROW_COLOR = GREY_C
HILITE = "#9C6E0C"

TYPE_LABELS = {"MFA_ORDEN": "MFA ORDEN", "Basket_CCK+": "Basket CCK+"}

# clockwise from the top, matching the paper's circular layout
FULL_TYPES = ["Pyramidal", "MFA_ORDEN", "Basket", "Bistratified",
              "QuadD-LM", "Ivy", "Axo-axonic", "Basket_CCK+"]
SMALL_TYPES = ["Pyramidal", "Basket", "MFA_ORDEN"]

HW_PHOTO = Path(__file__).parent / "assets" / "fig1_hardware_photo.png"


def label(text, **kw):
    return Text(TYPE_LABELS.get(text, text), **kw)


# ---------------------------------------------------------------------------
# network diagram builder (shared by the 8-type and 3-type panels)
# ---------------------------------------------------------------------------
def _disk_points(n, radius, seed):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(-1, 1, 2)
        if x * x + y * y <= 1:
            pts.append((x * radius, y * radius))
    return pts


def build_network(types, center, diagram_radius, ring_radii, dot_counts,
                   dot_radii, font_size=20, seed0=0):
    """A circular arrangement of neuron-type clusters with all-to-all
    connections (blue from Pyramidal, gold between interneurons), matching
    Fig 1's style. Returns (VGroup, dict[type -> world position]).
    """
    n = len(types)
    positions = {}
    for i, t in enumerate(types):
        angle = PI / 2 - i * TAU / n
        positions[t] = center + diagram_radius * np.array([np.cos(angle), np.sin(angle), 0])

    lines = VGroup()
    for i, a in enumerate(types):
        for b in types[i + 1:]:
            color = EXC if "Pyramidal" in (a, b) else INH
            lines.add(Line(positions[a], positions[b], stroke_width=1.5,
                           stroke_color=color, stroke_opacity=0.55))

    clusters, labels = VGroup(), VGroup()
    for i, t in enumerate(types):
        c = positions[t]
        r = ring_radii[t]
        ring = Circle(radius=r, stroke_color=NEURON_COLORS[t], stroke_width=4,
                      fill_color=NEURON_COLORS[t], fill_opacity=0.05).move_to(c)
        pts = _disk_points(dot_counts[t], r * 0.82, seed0 + i)
        dots = VGroup(*[Dot([c[0] + x, c[1] + y, 0], radius=dot_radii[t],
                            color=GREY_E) for x, y in pts])
        clusters.add(VGroup(ring, dots))
        out_dir = normalize(c - center) if np.linalg.norm(c - center) > 1e-6 else UP
        lbl = label(t, font_size=font_size, color=BLACK)
        # push out by the ring radius plus a gap plus however far the label's
        # own bounding box extends in the outward direction, so it clears the
        # ring regardless of whether that direction is horizontal, vertical or
        # diagonal (a purely radial offset only works for vertical labels)
        half_extent = 0.5 * (lbl.width * abs(out_dir[0]) + lbl.height * abs(out_dir[1]))
        lbl.move_to(c + out_dir * (r + 0.18 + half_extent))
        labels.add(lbl)

    return VGroup(lines, clusters, labels), positions


def full_scale_network(center=ORIGIN, scale=1.0, seed0=0):
    diagram_radius = 2.35 * scale
    ring_radii = {t: (0.46 if t == "Pyramidal" else 0.34) * scale for t in FULL_TYPES}
    dot_counts = {t: (60 if t == "Pyramidal" else 34) for t in FULL_TYPES}
    dot_radii = {t: 0.024 * scale for t in FULL_TYPES}
    return build_network(FULL_TYPES, center, diagram_radius, ring_radii,
                         dot_counts, dot_radii, font_size=int(20 * scale), seed0=seed0)


def reduced_scale_network(center=ORIGIN, scale=1.0, seed0=100):
    diagram_radius = 1.55 * scale
    ring_radii = {t: (0.30 if t == "Pyramidal" else 0.22) * scale for t in FULL_TYPES}
    dot_counts = {t: (20 if t == "Pyramidal" else 11) for t in FULL_TYPES}
    dot_radii = {t: 0.02 * scale for t in FULL_TYPES}
    return build_network(FULL_TYPES, center, diagram_radius, ring_radii,
                         dot_counts, dot_radii, font_size=int(18 * scale), seed0=seed0)


def small_scale_network(center=ORIGIN, scale=1.0, seed0=200):
    diagram_radius = 1.25 * scale
    ring_radii = {"Pyramidal": 0.55 * scale, "Basket": 0.22 * scale, "MFA_ORDEN": 0.3 * scale}
    dot_counts = {"Pyramidal": 160, "Basket": 6, "MFA_ORDEN": 12}   # real spiking counts
    dot_radii = {"Pyramidal": 0.011 * scale, "Basket": 0.03 * scale, "MFA_ORDEN": 0.022 * scale}
    return build_network(SMALL_TYPES, center, diagram_radius, ring_radii,
                         dot_counts, dot_radii, font_size=int(20 * scale), seed0=seed0)


def fit_width(mob, width):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def stage_arrow(text, start, end):
    arrow = Arrow(start, end, buff=0.1, color=ARROW_COLOR, stroke_width=6,
                 max_tip_length_to_length_ratio=0.25)
    txt = Text(text, font_size=22, color=MUTED, weight=BOLD)
    txt.next_to(arrow, UP, buff=0.12)
    return VGroup(arrow, txt)


def title_text(s):
    t = Text(s, font_size=34, weight=BOLD, color=BLACK)
    t.to_edge(UP, buff=0.35)
    return t


def caption_text(s, font_size=22, color=MUTED):
    return Text(s, font_size=font_size, color=color)


# ---------------------------------------------------------------------------
# Clip 1 - full-scale CA3 network
# ---------------------------------------------------------------------------
class Clip1(Scene):
    def construct(self):
        title = title_text("Full-scale CA3 network")
        net, _ = full_scale_network(center=ORIGIN + DOWN * 0.15, scale=0.88)
        cap = fit_width(caption_text("89,226 neurons  ·  8 neuron types  ·  250,078,223 synapses"), 12.5)
        cap.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(title))
        lines, clusters, labels = net
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.01), run_time=2)
        self.play(LaggedStart(*[FadeIn(c) for c in clusters], lag_ratio=0.08),
                  LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.08), run_time=1.5)
        self.play(FadeIn(cap, shift=UP * 0.15))
        self.wait(2)


# ---------------------------------------------------------------------------
# Clip 2 - scale network size: full-scale -> reduced-scale (still 8 types)
# ---------------------------------------------------------------------------
class Clip2(Scene):
    def construct(self):
        title = title_text("Full-scale CA3 network")
        net, _ = full_scale_network(center=ORIGIN + DOWN * 0.15, scale=0.88)
        cap = fit_width(caption_text("89,226 neurons  ·  8 neuron types  ·  250,078,223 synapses"), 12.5)
        cap.to_edge(DOWN, buff=0.35)
        self.add(title, net, cap)
        self.wait(0.3)

        left_center = LEFT * 3.6 + DOWN * 0.2
        self.play(net.animate.scale(0.62 / 0.88).move_to(left_center),
                  FadeOut(cap), run_time=1.2)
        left_cap = caption_text("Full scale", font_size=20)
        left_cap.next_to(net, DOWN, buff=0.35)
        self.play(FadeIn(left_cap))

        arrow = stage_arrow("Scale network size", LEFT * 1.05, RIGHT * 1.05)
        self.play(FadeIn(arrow))
        self.play(FadeOut(title), FadeIn(title_text("Scaling down the network size")))

        right_center = RIGHT * 3.6 + DOWN * 0.2
        reduced, _ = reduced_scale_network(center=right_center)
        r_lines, r_clusters, r_labels = reduced
        self.play(LaggedStart(*[Create(l) for l in r_lines], lag_ratio=0.01), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(c) for c in r_clusters], lag_ratio=0.08),
                  LaggedStart(*[FadeIn(l) for l in r_labels], lag_ratio=0.08), run_time=1.2)
        right_cap = fit_width(caption_text(
            "Reduced scale  ·  same 8 types  ·  similar periodic activity", font_size=20), 6.2)
        right_cap.next_to(reduced, DOWN, buff=0.35)
        self.play(FadeIn(right_cap))
        self.wait(2)


# ---------------------------------------------------------------------------
# Clip 3 - scale neuron diversity: reduced-scale (8 types) -> 3 types
# ---------------------------------------------------------------------------
class Clip3(Scene):
    def construct(self):
        # rebuild Clip2's end state instantly
        title = title_text("Scaling down the network size")
        left_center = LEFT * 3.6 + DOWN * 0.2
        full_net, _ = full_scale_network(center=left_center, scale=0.62)
        left_cap = caption_text("Full scale", font_size=20).next_to(full_net, DOWN, buff=0.35)
        arrow1 = stage_arrow("Scale network size", LEFT * 1.05, RIGHT * 1.05)
        right_center = RIGHT * 3.6 + DOWN * 0.2
        reduced, _ = reduced_scale_network(center=right_center)
        right_cap = fit_width(caption_text(
            "Reduced scale  ·  same 8 types  ·  similar periodic activity", font_size=20), 6.2)
        right_cap.next_to(reduced, DOWN, buff=0.35)
        self.add(title, full_net, left_cap, arrow1, reduced, right_cap)
        self.wait(0.3)

        # remove the full-scale network entirely
        self.play(FadeOut(full_net), FadeOut(left_cap), FadeOut(arrow1),
                  FadeOut(title), run_time=0.8)

        # move the reduced (8-type) network to the left
        self.play(reduced.animate.move_to(left_center),
                  FadeOut(right_cap), run_time=1.0)
        new_left_cap = caption_text("Reduced scale  ·  8 types", font_size=20)
        new_left_cap.next_to(reduced, DOWN, buff=0.35)
        self.play(FadeIn(new_left_cap))

        arrow2 = stage_arrow("Scale neuron diversity", LEFT * 1.05, RIGHT * 1.05)
        self.play(FadeIn(arrow2), FadeIn(title_text("Scaling down neuron diversity")))

        small, _ = small_scale_network(center=right_center)
        s_lines, s_clusters, s_labels = small
        self.play(LaggedStart(*[Create(l) for l in s_lines], lag_ratio=0.15), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(c) for c in s_clusters], lag_ratio=0.25),
                  LaggedStart(*[FadeIn(l) for l in s_labels], lag_ratio=0.25), run_time=1.2)
        small_cap = fit_width(caption_text(
            "3 types  ·  179 neurons  ·  17,996 synapses  ·  still resting-state", font_size=20), 6.2)
        small_cap.next_to(small, DOWN, buff=0.35)
        self.play(FadeIn(small_cap))
        self.wait(2)


# ---------------------------------------------------------------------------
# Clip 4 - 3 types -> crossbar array (25x25 memristor tile)
# ---------------------------------------------------------------------------
def build_crossbar_tile(center, tile_size=25, cell=0.135, seed=41):
    """A small illustrative run of the real greedy packer, just to get a
    realistic mix of block sizes to drop into the tile (see greedy_scene.py
    for the full, narrated version of this algorithm)."""
    res = greedy_blocking(tile_size, tile_size, 6, 1, 0.45, min_accept_frac=0.95,
                          decay=0.9, max_tries=40, fallback_K=3, seed=seed)
    shapes = [(len(pre), len(post)) for pre, post in res.mappings]
    slots, used = first_fit(shapes, size=tile_size)

    t0 = center + LEFT * (tile_size - 1) * cell / 2 + UP * (tile_size - 1) * cell / 2
    cell_pos = lambda i, j: t0 + RIGHT * j * cell + DOWN * i * cell
    grid = VGroup(*[Square(cell, stroke_color="#C7CDD6", stroke_width=1,
                           fill_color="#EDEFF2", fill_opacity=1).move_to(cell_pos(i, j))
                    for i in range(tile_size) for j in range(tile_size)])
    blocks = VGroup()
    for n, ((w, h), (i, j)) in enumerate(zip(shapes, slots)):
        color = EXC if n % 2 == 0 else "#4C74C9"   # alternate shades so blocks read distinctly
        blk = Rectangle(width=w * cell, height=h * cell, stroke_color=BLACK,
                        stroke_width=1, fill_color=color, fill_opacity=0.85)
        blk.move_to(cell_pos(i + (h - 1) / 2, j + (w - 1) / 2))
        blocks.add(blk)
    return grid, blocks, int(used.sum()), tile_size * tile_size


class Clip4(Scene):
    def construct(self):
        # rebuild Clip3's end state instantly (8-type network already gone)
        title = title_text("Scaling down neuron diversity")
        left_center = LEFT * 3.6 + DOWN * 0.2
        right_center = RIGHT * 3.6 + DOWN * 0.2
        small, _ = small_scale_network(center=left_center)
        left_cap = fit_width(caption_text(
            "3 types  ·  179 neurons  ·  17,996 synapses", font_size=20), 6.2)
        left_cap.next_to(small, DOWN, buff=0.35)
        self.add(title, small, left_cap)
        self.wait(0.3)

        arrow = stage_arrow("Implementation", LEFT * 1.05, RIGHT * 1.05)
        self.play(FadeOut(title), FadeIn(title_text("Mapping connections onto hardware")))
        self.play(FadeIn(arrow))

        grid, blocks, used, total = build_crossbar_tile(right_center)
        tile_title = fit_width(caption_text("25 × 25 memristor crossbar tile", font_size=20), 6.2)
        tile_title.next_to(grid, UP, buff=0.3)
        self.play(FadeIn(tile_title),
                  LaggedStart(*[FadeIn(sq) for sq in grid], lag_ratio=0.001, run_time=1.2))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in blocks], lag_ratio=0.06), run_time=2.5)

        stats = fit_width(caption_text(f"Devices used: {used} / {total}  ({100 * used / total:.0f}%)",
                                       font_size=20, color="#2E7D32"), 6.2)
        stats.next_to(grid, DOWN, buff=0.35)
        self.play(FadeIn(stats))
        self.wait(2)


# ---------------------------------------------------------------------------
# Clip 5 - close the loop: full-scale network pointing at the hardware
# ---------------------------------------------------------------------------
class Clip5(Scene):
    def construct(self):
        # rebuild Clip4's end state instantly
        title = title_text("Mapping connections onto hardware")
        left_center = LEFT * 3.6 + DOWN * 0.2
        right_center = RIGHT * 3.6 + DOWN * 0.2
        small, _ = small_scale_network(center=left_center)
        left_cap = fit_width(caption_text(
            "3 types  ·  179 neurons  ·  17,996 synapses", font_size=20), 6.2)
        left_cap.next_to(small, DOWN, buff=0.35)
        arrow = stage_arrow("Implementation", LEFT * 1.05, RIGHT * 1.05)
        grid, blocks, used, total = build_crossbar_tile(right_center)
        tile_title = fit_width(caption_text("25 × 25 memristor crossbar tile", font_size=20), 6.2)
        tile_title.next_to(grid, UP, buff=0.3)
        stats = fit_width(caption_text(f"Devices used: {used} / {total}  ({100 * used / total:.0f}%)",
                                       font_size=20, color="#2E7D32"), 6.2)
        stats.next_to(grid, DOWN, buff=0.35)
        self.add(title, small, left_cap, arrow, grid, blocks, tile_title, stats)
        self.wait(0.3)

        # remove the 3-type network, its caption and the "Implementation" arrow
        self.play(FadeOut(small), FadeOut(left_cap), FadeOut(arrow),
                  FadeOut(title), run_time=0.8)

        # bring back the full-scale network on the left, pointing at hardware
        full_net, _ = full_scale_network(center=left_center, scale=0.62)
        full_cap = fit_width(caption_text(
            "Full-scale CA3 network  ·  89,226 neurons  ·  8 types", font_size=18), 6.2)
        full_cap.next_to(full_net, DOWN, buff=0.3)
        self.play(FadeIn(full_net, scale=0.9), FadeIn(full_cap),
                  FadeIn(title_text("What the hardware approximates")))

        # shrink the crossbar diagram up and bring in the real hardware photo
        crossbar_group = VGroup(grid, blocks, tile_title, stats)
        self.play(crossbar_group.animate.scale(0.55).move_to(right_center + UP * 1.15),
                  run_time=1.0)

        hw_photo = ImageMobject(str(HW_PHOTO))
        hw_photo.width = 3.2
        hw_photo.move_to(right_center + DOWN * 1.35)
        hw_cap = fit_width(caption_text(
            "FPGA + mixed-signal memristor daughterboard", font_size=18), 5.6)
        hw_cap.next_to(hw_photo, DOWN, buff=0.2)
        self.play(FadeIn(hw_photo), FadeIn(hw_cap))

        final_arrow = stage_arrow("", LEFT * 1.05, RIGHT * 1.05)
        self.play(FadeIn(final_arrow))

        closing = fit_width(caption_text(
            "A biologically-realistic CA3 network, downscaled and mapped onto"
            " real neuromorphic memristor hardware",
            font_size=22, color=BLACK), 13.0)
        closing.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(closing, shift=UP * 0.15))
        self.wait(3)
