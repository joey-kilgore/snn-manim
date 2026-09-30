"""
connection_score_scene.py

Criterion 4 of the CA3-scaling scoring rubric (arXiv:2609.16429): the change
in connection-probability score, C(N). Optuna tunes each scaled network's
connection probabilities; C(N) measures how close those tuned probabilities
land to the "base scaled" targets that fall out of the paper's downscaling
law, so that as few changes as possible are needed away from the projected
biology.

Network scaling (Eq. 1 in the paper):

    y = a^x * y0

Population size scales with x = 2/3, total synapse count scales with x = 1,
for a shared scale factor a (a = 0.0001 for CA3_0_0001_3_group). Dividing the
scaled synapse count by the scaled pre/post population sizes gives the base
scaled connection probability for every (pre, post) pair - the target that
network tuning is scored against.

Per connection, the score is a normalized Gaussian centered on that target,
with std = 1/10th of the target, evaluated at the probability the tuned
network actually has:

    score_c = exp( -0.5 * ((p_N,c - p_target,c) / (0.1 * p_target,c))^2 )
    C(N)    = (1 / #Connections) * sum_c score_c

For a 3-neuron-type network there are 3x3 = 9 directed connection types.

This scene walks that pipeline for CA3_0_0001_3_group_18 (Pyramidal, Basket,
MFA_ORDEN): full-scale populations/probabilities (left) -> the scaling law
recalculating a target probability (center) -> the probability actually
loaded from network.dat via sim-loader (right) -> 9 Gaussians, one per
connection, scored and averaged into C(N).

Render:
    manim -pql connection_score_scene.py ConnectionScore     # quick preview
    manim -pqh connection_score_scene.py ConnectionScore     # 1080p
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from manim import *

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "sim-loader"))
sys.path.insert(0, str(ROOT / "sim-loader" / "scripts"))

from networkParams import getCA3FullScale          # noqa: E402
from simulation import Simulation                  # noqa: E402
from raster_scene import GROUPS, COLORS, MUTED, fit  # noqa: E402

config.background_color = WHITE
Text.set_default(font="Segoe UI", color=BLACK)

NETWORK_DAT = ROOT / "data" / "CA3_0_0001_3_group_18" / "network.dat"

SCALE_A = 0.0001      # the paper's scale factor a for this network
POP_EXP = 2 / 3        # population scales as a^(2/3)
SYN_EXP = 1.0          # synapse count scales as a^1

# CA3_0_0001_3_group_18's groups, mapped to their full-scale (8 neuron type)
# indices in networkParams.getCA3FullScale()
FULL_IDX = {"Pyramidal": 0, "Basket": 2, "MFA_ORDEN": 6}
ABBR = {"Pyramidal": "Pyr", "Basket": "Bas", "MFA_ORDEN": "MFA"}

HILITE = "#9C6E0C"
TARGET_COLOR = BLUE_E
LOADED_COLOR = "#9C6E0C"
CELL_MAX = BLUE_E

PAIRS = [(pre, post) for pre in GROUPS for post in GROUPS]


# ---------------------------------------------------------------------------
# Data: full-scale -> scaled target -> loaded (from network.dat)
# ---------------------------------------------------------------------------
def gaussian(x, mu, sigma):
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def load_data():
    groupInfo, connInfo, _firingRates = getCA3FullScale()

    full_pop = {g: groupInfo[FULL_IDX[g]]["pop"] for g in GROUPS}
    full_prob = {(pre, post): connInfo[(FULL_IDX[pre], FULL_IDX[post])][1]
                 for pre, post in PAIRS}

    scaled_pop = {g: SCALE_A ** POP_EXP * full_pop[g] for g in GROUPS}
    full_syn = {k: full_prob[k] * full_pop[k[0]] * full_pop[k[1]] for k in PAIRS}
    scaled_syn = {k: SCALE_A ** SYN_EXP * full_syn[k] for k in PAIRS}
    scaled_prob = {k: scaled_syn[k] / (scaled_pop[k[0]] * scaled_pop[k[1]]) for k in PAIRS}

    sim = Simulation(str(NETWORK_DAT))
    name_to_idx = {g.groupNameClean: i for i, g in enumerate(sim.groups)}
    loaded_pop = {g: sim.groups[name_to_idx[g]].endN - sim.groups[name_to_idx[g]].startN + 1
                  for g in GROUPS}
    counts = Counter()
    for syn in sim.synapses[0]:
        counts[(syn.gGrpIdPre, syn.gGrpIdPost)] += 1
    loaded_syn = {(pre, post): counts[(name_to_idx[pre], name_to_idx[post])] for pre, post in PAIRS}
    loaded_prob = {k: loaded_syn[k] / (loaded_pop[k[0]] * loaded_pop[k[1]]) for k in PAIRS}

    sigma = {k: 0.1 * scaled_prob[k] for k in PAIRS}
    scores = {k: gaussian(loaded_prob[k], scaled_prob[k], sigma[k]) for k in PAIRS}
    change_score = sum(scores.values()) / len(PAIRS)

    return dict(full_pop=full_pop, full_prob=full_prob, full_syn=full_syn,
                scaled_pop=scaled_pop, scaled_syn=scaled_syn, scaled_prob=scaled_prob,
                loaded_pop=loaded_pop, loaded_syn=loaded_syn, loaded_prob=loaded_prob,
                sigma=sigma, scores=scores, change_score=change_score)


def fmt_prob(p):
    return f"{p:.2f}" if p >= 1 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
HEADER_Y = 3.75
CAPTION_Y = 3.28

COL_CX = {"full": -4.6, "scaled": 0.0, "loaded": 4.6}
COL_W = 3.6
TITLE_Y = 2.82
POP_Y = 2.32
COL_LABEL_Y = 1.85
MAT_TOP, MAT_BOTTOM = 1.55, -0.85
ROW_H = (MAT_TOP - MAT_BOTTOM) / 3
ROW_LABEL_DX = -COL_W / 2 - 0.25

DERIV_LINE_Y = -1.3
WORK_Y0 = -1.75
WORK_DY = 0.4


def col_x(cx, c):
    return cx - COL_W / 2 + (c + 0.5) * COL_W / 3


def row_y(r):
    return MAT_TOP - (r + 0.5) * ROW_H


class ConnectionScore(Scene):
    def construct(self):
        self.data = load_data()
        self.intro()
        self.full_scale_column()
        self.scaling_derivation()
        self.loaded_column()
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.section_a_mobjects()])
        self.gaussian_grid()
        self.finale()

    # ------------------------------------------------------------- helpers
    def say(self, text, y=CAPTION_Y, rt=0.5, font_size=32):
        new = fit(Text(text, font_size=font_size, color=MUTED), 12.6).move_to([0, y, 0])
        if getattr(self, "caption", None) is not None:
            self.play(FadeOut(self.caption, shift=UP * 0.1), FadeIn(new, shift=UP * 0.1), run_time=rt)
        else:
            self.play(FadeIn(new, shift=UP * 0.1), run_time=rt)
        self.caption = new

    def section_a_mobjects(self):
        return [m for name in ("full_group", "scaled_group", "loaded_group", "deriv_group",
                                "caption")
                if (m := getattr(self, name, None)) is not None]

    # ------------------------------------------------------------- intro
    def intro(self):
        title = Text("Connection score", font_size=54, weight=BOLD)
        sub = fit(Text("Criterion 4 of the CA3-scaling rubric: how close a scaled network's "
                       "connection probabilities land on their targets", font_size=26, color=MUTED),
                  12.6).next_to(title, DOWN, buff=0.3)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.15))
        self.wait(1.3)
        self.play(FadeOut(sub))
        self.play(title.animate.scale(0.5 * 52 / 54).to_edge(UP, buff=0.12))
        self.header = title
        self.caption = None

    # ------------------------------------------------------------- matrix
    def build_matrix(self, cx, prob_dict, decimals_fn=fmt_prob, opacity_cap=1.0):
        """A 3x3 (pre x post) probability grid centered at cx. Returns
        (VGroup of cells+labels, {pair: (rect, text)})."""
        cells = {}
        mobs = VGroup()
        row_labels = VGroup(*[
            Text(ABBR[g], font_size=22, color=COLORS[g])
            .move_to([cx + ROW_LABEL_DX, row_y(r), 0])
            for r, g in enumerate(GROUPS)
        ])
        col_labels = VGroup(*[
            Text(ABBR[g], font_size=22, color=COLORS[g])
            .move_to([col_x(cx, c), COL_LABEL_Y, 0])
            for c, g in enumerate(GROUPS)
        ])
        mobs.add(row_labels, col_labels)
        for r, pre in enumerate(GROUPS):
            for c, post in enumerate(GROUPS):
                p = prob_dict[(pre, post)]
                opac = min(p / opacity_cap, 1.0)
                rect = Rectangle(width=COL_W / 3 * 0.92, height=ROW_H * 0.88,
                                 stroke_color=MUTED, stroke_width=1,
                                 fill_color=CELL_MAX, fill_opacity=opac)
                rect.move_to([col_x(cx, c), row_y(r), 0])
                txt = Text(decimals_fn(p), font_size=18,
                          color=(WHITE if opac > 0.55 else BLACK))
                txt.move_to(rect.get_center())
                cells[(pre, post)] = (rect, txt)
                mobs.add(rect, txt)
        return mobs, cells

    def pop_row(self, cx, pop_dict, fmt):
        row = VGroup(*[
            VGroup(Dot(radius=0.05, color=COLORS[g]),
                  Text(f"{ABBR[g]} {fmt(pop_dict[g])}", font_size=20, color=MUTED))
            .arrange(RIGHT, buff=0.08)
            for g in GROUPS
        ]).arrange(RIGHT, buff=0.28)
        fit(row, COL_W + 0.3)
        row.move_to([cx, POP_Y, 0])
        return row

    # ------------------------------------------------------------- column A
    def full_scale_column(self):
        d = self.data
        title = Text("Full-scale CA3", font_size=28, weight=BOLD).move_to([COL_CX["full"], TITLE_Y, 0])
        pop = self.pop_row(COL_CX["full"], d["full_pop"], lambda n: f"{n:,}")
        matrix, self.full_cells = self.build_matrix(COL_CX["full"], d["full_prob"])
        self.full_group = VGroup(title, pop, matrix)

        self.say("The full-scale network: 8 neuron types, biologically-measured "
                  "connection probabilities (Kopsick et al.)")
        self.play(FadeIn(title), FadeIn(pop))
        self.play(LaggedStart(*[FadeIn(m) for m in matrix], lag_ratio=0.03), run_time=1.3)
        self.wait(0.6)

    # ------------------------------------------------------------- column B
    def scaling_derivation(self):
        d = self.data
        title = Text("Scaled ×10⁻⁴ (target)", font_size=28, weight=BOLD,
                     color=TARGET_COLOR).move_to([COL_CX["scaled"], TITLE_Y, 0])
        self.play(FadeIn(title))

        # empty skeleton: dashes until the derivation fills them in
        pop_placeholder = self.pop_row(COL_CX["scaled"], {g: "?" for g in GROUPS},
                                       lambda s: s)
        empty_prob = {k: 0.0 for k in PAIRS}
        matrix, self.scaled_cells = self.build_matrix(COL_CX["scaled"], empty_prob)
        self.play(FadeIn(pop_placeholder), FadeIn(matrix))

        law = MathTex(r"y = a^{x}\, y_0", font_size=40, color=BLACK).move_to([0, DERIV_LINE_Y, 0])
        self.say(f"Eq. 1: scale factor a = {SCALE_A}. Population scales by a^(2/3), "
                 f"total synapses by a^1", y=CAPTION_Y - 0.02)
        self.play(FadeIn(law))
        self.wait(0.8)
        self.play(FadeOut(law))

        # --- step 1: scale populations
        step1 = fit(Text("Population:  N' = a^(2/3) · N", font_size=26, color=BLACK), 8.5)
        step1.move_to([0, WORK_Y0, 0])
        self.play(FadeIn(step1))
        new_pop = self.pop_row(COL_CX["scaled"], d["scaled_pop"], lambda n: f"{n:.0f}")
        self.play(ReplacementTransform(pop_placeholder, new_pop))
        self.wait(0.4)

        # --- step 2/3: worked example for Pyramidal -> Pyramidal
        pre, post = "Pyramidal", "Pyramidal"
        step2 = fit(Text(f"Synapses ({ABBR[pre]}→{ABBR[post]}):  "
                         f"S = p·N_pre·N_post = {d['full_prob'][(pre, post)]} × "
                         f"{d['full_pop'][pre]:,} × {d['full_pop'][post]:,} "
                         f"≈ {d['full_syn'][(pre, post)]:,.0f}", font_size=24, color=BLACK), 12.6)
        step2.move_to([0, WORK_Y0 - WORK_DY, 0])
        self.play(ReplacementTransform(step1, step2))
        self.wait(0.5)

        step3 = fit(Text(f"S' = a · S ≈ {d['scaled_syn'][(pre, post)]:,.0f}     "
                         f"p' = S' / (N'_pre · N'_post) ≈ {d['scaled_prob'][(pre, post)]:.3f}",
                         font_size=24, color=TARGET_COLOR), 12.6)
        step3.move_to([0, WORK_Y0 - 2 * WORK_DY, 0])
        self.play(FadeIn(step3))
        self.wait(0.5)

        rect, txt = self.scaled_cells[(pre, post)]
        new_rect = Rectangle(width=rect.width, height=rect.height, stroke_color=MUTED,
                             stroke_width=1, fill_color=CELL_MAX,
                             fill_opacity=min(d["scaled_prob"][(pre, post)], 1.0)).move_to(rect)
        new_txt = Text(fmt_prob(d["scaled_prob"][(pre, post)]), font_size=18,
                       color=WHITE if min(d["scaled_prob"][(pre, post)], 1.0) > 0.55 else BLACK)
        new_txt.move_to(rect.get_center())
        self.play(Transform(rect, new_rect), Transform(txt, new_txt),
                  Indicate(rect, color=TARGET_COLOR, scale_factor=1.08))
        self.scaled_cells[(pre, post)] = (rect, txt)
        self.wait(0.4)

        note = fit(Text("Note: the calculated target can exceed 1 (e.g. Bas→Pyr ≈ 3.2) - "
                        "the network shrank faster than its synapse count, so the actual "
                        "probability is capped at 1", font_size=20, color=MUTED), 12.6)
        note.move_to([0, WORK_Y0 - 3 * WORK_DY, 0])

        step4 = fit(Text("Repeat for the remaining 8 connections...", font_size=24, color=BLACK),
                    9).move_to([0, WORK_Y0 - WORK_DY, 0])
        self.play(FadeOut(step2), FadeOut(step3), FadeIn(step4), FadeIn(note))

        remaining = [(rect, txt, k) for k, (rect, txt) in self.scaled_cells.items() if k != (pre, post)]
        anims = []
        for rect, txt, k in remaining:
            p = min(d["scaled_prob"][k], 1.0)
            new_rect = Rectangle(width=rect.width, height=rect.height, stroke_color=MUTED,
                                 stroke_width=1, fill_color=CELL_MAX, fill_opacity=p).move_to(rect)
            new_txt = Text(fmt_prob(d["scaled_prob"][k]), font_size=18,
                           color=WHITE if p > 0.55 else BLACK).move_to(rect.get_center())
            anims.append(AnimationGroup(Transform(rect, new_rect), Transform(txt, new_txt)))
        self.play(LaggedStart(*anims, lag_ratio=0.12), run_time=1.6)
        self.wait(0.5)
        self.play(FadeOut(step4), FadeOut(note))

        self.scaled_group = VGroup(title, new_pop, matrix)
        self.deriv_group = VGroup()

    # ------------------------------------------------------------- column C
    def loaded_column(self):
        d = self.data
        title = Text("Loaded (network.dat)", font_size=28, weight=BOLD,
                     color=LOADED_COLOR).move_to([COL_CX["loaded"], TITLE_Y, 0])
        self.say("Read CA3_0_0001_3_group_18/network.dat with sim-loader: what CARLsim6 "
                 "actually tuned", y=CAPTION_Y - 0.02)
        self.play(FadeIn(title))

        pop = self.pop_row(COL_CX["loaded"], d["loaded_pop"], lambda n: f"{n}")
        self.play(FadeIn(pop))

        matrix, self.loaded_cells = self.build_matrix(COL_CX["loaded"], d["loaded_prob"])
        self.play(LaggedStart(*[FadeIn(m) for m in matrix], lag_ratio=0.03), run_time=1.3)
        self.loaded_group = VGroup(title, pop, matrix)
        self.wait(0.8)

        self.say("Full-scale, scaled target, and loaded probability - now score how close "
                 "loaded landed to target")
        self.wait(1.0)

    # ------------------------------------------------------------- gaussians
    def gaussian_grid(self):
        d = self.data
        self.caption = None

        self.say("For each connection: a normalized Gaussian centered on its target, "
                 "std = 10% of target - scored at the loaded probability")
        formula = MathTex(
            r"\text{score}_c = \exp\!\left(-\tfrac{1}{2}\left(\dfrac{p_{N,c}-p_{target,c}}"
            r"{0.1\,p_{target,c}}\right)^{2}\right)",
            font_size=34, color=BLACK).next_to(self.caption, DOWN, buff=0.18)
        self.play(FadeIn(formula))
        self.wait(1.2)
        self.play(FadeOut(formula))

        GRID_LEFT, GRID_RIGHT = -6.3, 6.3
        GRID_TOP, GRID_BOTTOM = 2.15, -2.55
        cell_w = (GRID_RIGHT - GRID_LEFT) / 3
        cell_h = (GRID_TOP - GRID_BOTTOM) / 3

        self.score_labels = {}
        plots = {}
        DETAILED = 2
        order = list(PAIRS)

        for i, (pre, post) in enumerate(order):
            r, c = GROUPS.index(pre), GROUPS.index(post)
            cell_top = GRID_TOP - r * cell_h
            rx0 = GRID_LEFT + c * cell_w + 0.4
            rx1 = GRID_LEFT + (c + 1) * cell_w - 0.15
            ry0 = cell_top - cell_h + 0.38
            ry1 = cell_top - 0.4

            mu, sigma = d["scaled_prob"][(pre, post)], d["sigma"][(pre, post)]
            loaded = d["loaded_prob"][(pre, post)]
            score = d["scores"][(pre, post)]

            lo = max(0.0, min(mu - 3.2 * sigma, loaded))
            hi = max(mu + 3.2 * sigma, loaded)
            span = hi - lo if hi > lo else 1.0
            lo, hi = lo - 0.08 * span, hi + 0.08 * span

            def px(x, rx0=rx0, rx1=rx1, lo=lo, hi=hi):
                return rx0 + (x - lo) / (hi - lo) * (rx1 - rx0)

            def py(y, ry0=ry0, ry1=ry1):
                return ry0 + y * (ry1 - ry0)

            xs = np.linspace(lo, hi, 50)
            pts = [[px(x), py(gaussian(x, mu, sigma)), 0] for x in xs]
            curve = VMobject(stroke_color=TARGET_COLOR, stroke_width=2.5)
            curve.set_points_smoothly(pts)
            axis = Line([rx0, ry0, 0], [rx1, ry0, 0], stroke_color=MUTED, stroke_width=1.2)
            y_axis = Line([rx0, ry0, 0], [rx0, ry1, 0], stroke_color=MUTED, stroke_width=1.2)
            y_ticks = VGroup(*[
                VGroup(Line([rx0 - 0.05, py(v), 0], [rx0, py(v), 0], stroke_color=MUTED,
                            stroke_width=1.2),
                       Text(s, font_size=14, color=MUTED).move_to([rx0 - 0.14, py(v), 0]))
                for v, s in ((0.0, "0"), (1.0, "1"))
            ])
            axes = VGroup(axis, y_axis, y_ticks)
            if c == 0:
                y_title = Text("score", font_size=14, color=MUTED).rotate(PI / 2)
                y_title.move_to([rx0 - 0.3, (ry0 + ry1) / 2, 0])
                axes.add(y_title)
            x_tick = VGroup(
                Line([px(mu), ry0, 0], [px(mu), ry0 - 0.05, 0], stroke_color=MUTED, stroke_width=1.2),
                Text(fmt_prob(mu), font_size=14, color=TARGET_COLOR).move_to([px(mu), ry0 - 0.16, 0]),
            )
            target_line = DashedLine([px(mu), ry0, 0], [px(mu), py(1.0), 0],
                                     stroke_color=MUTED, stroke_width=1.2, dash_length=0.06)
            load_h = gaussian(loaded, mu, sigma)
            load_line = Line([px(loaded), ry0, 0], [px(loaded), py(load_h), 0],
                             stroke_color=COLORS[pre], stroke_width=2.5)
            load_dot = Dot([px(loaded), py(load_h), 0], radius=0.055, color=HILITE)

            header_y = cell_top - 0.14
            label = Text(f"{ABBR[pre]} → {ABBR[post]}", font_size=20, color=COLORS[pre])
            label.move_to([rx0, header_y, 0], aligned_edge=LEFT)
            score_lbl = Text(f"score = {score:.3f}", font_size=18, color=HILITE, weight=BOLD)
            score_lbl.move_to([rx1, header_y, 0], aligned_edge=RIGHT)

            group = VGroup(axes, curve, target_line, x_tick, load_line, load_dot, label)
            plots[(pre, post)] = group
            self.score_labels[(pre, post)] = score_lbl

            if i < DETAILED:
                self.play(FadeIn(label), Create(axes))
                self.play(Create(curve), Create(target_line), FadeIn(x_tick))
                self.play(Create(load_line), FadeIn(load_dot))
                self.play(FadeIn(score_lbl))
                self.wait(0.25)
            else:
                self.play(FadeIn(group), FadeIn(score_lbl), run_time=0.4)

        self.plots = plots
        self.wait(0.5)

    # ------------------------------------------------------------- finale
    def finale(self):
        d = self.data
        self.say(f"C(N) = average of all 9 scores ≈ {d['change_score']:.3f}", y=CAPTION_Y - 0.02)

        avg_formula = MathTex(
            rf"C(N) = \tfrac{{1}}{{9}}\sum_c \text{{score}}_c \approx {d['change_score']:.3f}",
            font_size=38, color=HILITE).move_to([0, -3.25, 0])
        box = SurroundingRectangle(avg_formula, color=HILITE, buff=0.2)
        self.play(*[Indicate(lbl, color=HILITE, scale_factor=1.1)
                    for lbl in self.score_labels.values()],
                  run_time=1.2)
        self.play(FadeIn(avg_formula), Create(box))
        self.wait(3)
