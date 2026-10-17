"""
firing_rate_score_scene.py

Criterion 3 of the CA3-scaling scoring rubric (arXiv:2609.16429): the
firing rate (frequency) score F(N). For each neuron subtype g, the mean
firing rate f_N,g measured over the final 1 second of the simulation is
scored against the full-scale target f_Full,g (sim-loader/networkParams.py,
getCA3FullScale()'s firingRates) using a normalized Gaussian:

    PDF_norm(x; mu, sigma) = exp( -(x - mu)^2 / (2*sigma^2) )

peaking at 1.0 when x = mu, with mu = f_Full,g and sigma = 0.1 * f_Full,g
(one tenth of the target rate). The per-subtype scores are averaged:

    F(N) = (1 / #Groups) * sum_g PDF_norm(f_Full,g, 0.1*f_Full,g, f_N,g)

Uses the CA3_0_0001_3_group_18 recording (same data/window as
activity_score_scene.py): raster -> per-subtype mean firing rate vs. target
-> the Gaussian built for one subtype worked in full -> all three subtypes'
curves and scores -> combined into F(N).

Each clip is its own Scene so it can be dropped into a presentation. A clip
rebuilds the exact state the previous clip ended on (instantly, no
animation) and then animates only its own slice:

    Clip1  raster -> per-subtype firing rate vs. target, as a bar chart
    Clip2  the Gaussian score worked in full for one subtype (Pyramidal)
    Clip3  every subtype's curve, combined into the firing rate score F(N)

Render:
    manim -pql firing_rate_score_scene.py FiringRateScore     # quick preview, full video
    manim -pqh firing_rate_score_scene.py FiringRateScore     # 1080p
    manim -pql firing_rate_score_scene.py Clip1 Clip2 Clip3   # the three clips separately
"""

import os
import sys

import numpy as np
from manim import *

from raster_scene import GROUPS, COLORS, MUTED, load_spikes, SPIKES_CSV, fit, label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sim-loader"))
from networkParams import getCA3FullScale

config.background_color = WHITE

WINDOW_MS = 1000  # final 1 second of the recording, matches activity_score_scene

ACCENT = "#9C6E0C"
TRACE_COLOR = "#000000"

Text.set_default(font="Segoe UI", color=BLACK)

# ---------------------------------------------------------------------------
# Layout (manim default frame: x in [-7.11, 7.11], y in [-4, 4])
# boxes are (left, bottom, right, top) of the plotting area itself
# ---------------------------------------------------------------------------
TITLE_Y = 3.72
CAPTION_Y = 3.28

FULL_BOX = (-5.6, -2.2, 5.9, 2.7)

# once a graph is done it shrinks into the thumbnail slot on the left, and an
# arrow leads from it to the next graph, built in MAIN_BOX
MAIN_BOX = (-1.7, -2.2, 6.7, 2.7)
THUMB_CENTER = np.array([-5.3, 0.25, 0])
THUMB_W, THUMB_H = 3.4, 3.0
ARROW_X = (-3.45, -2.75)

GAUSS_TOP, GAUSS_BOTTOM = 1.3, -1.15
GAUSS_BOXES = [
    (-6.2, GAUSS_BOTTOM, -2.13, GAUSS_TOP),
    (-1.83, GAUSS_BOTTOM, 2.23, GAUSS_TOP),
    (2.53, GAUSS_BOTTOM, 6.6, GAUSS_TOP),
]
READOUT_ROW_Y = GAUSS_BOTTOM - 0.66
SCORE_ROW_Y = GAUSS_BOTTOM - 0.94
FINALE_Y = -2.85


def crisp_text(s, font_size=24, **kwargs):
    return Text(s, font_size=font_size * 2, **kwargs).scale(0.5)


def norm_gauss(x, mu, sigma):
    return np.exp(-((np.asarray(x, dtype=float) - mu) ** 2) / (2 * sigma ** 2))


def nice_ticks(vmax, n=4):
    step = vmax / n
    magnitude = 10 ** np.floor(np.log10(step))
    step = next((m * magnitude for m in (1, 2, 2.5, 5, 10) if step <= m * magnitude), 10 * magnitude)
    ticks, v = [], 0.0
    while v <= vmax + 1e-9:
        ticks.append((v, f"{v:g}"))
        v += step
    return ticks


# ---------------------------------------------------------------------------
# Panel: a framed linear-axis plot; data mapped into [0,1] of the box so
# nothing can be drawn outside the frame.
# ---------------------------------------------------------------------------
class Panel:
    def __init__(self, box, xspec, yspec, xticks, yticks, xlabel, ylabel):
        self.box = box
        self.left, self.bottom, self.right, self.top = box
        self.w = self.right - self.left
        self.h = self.top - self.bottom
        self.xspec, self.yspec = xspec, yspec
        self.xtick_defs, self.ytick_defs = xticks, yticks
        self.xlabel_text, self.ylabel_text = xlabel, ylabel

    def norm_x(self, x):
        lo, hi = self.xspec
        return np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0, 1)

    def norm_y(self, y):
        lo, hi = self.yspec
        return np.clip((np.asarray(y, dtype=float) - lo) / (hi - lo), 0, 1)

    def c2p(self, x, y):
        u, v = self.norm_x(x), self.norm_y(y)
        return np.stack([self.left + u * self.w, self.bottom + v * self.h, np.zeros_like(u + v)], axis=-1)

    def x_of(self, x):
        return float(self.left + self.norm_x(x) * self.w)

    def y_of(self, y):
        return float(self.bottom + self.norm_y(y) * self.h)

    def curve(self, xs, ys, color, width=2.5):
        return VMobject(stroke_color=color, stroke_width=width).set_points_smoothly(self.c2p(xs, ys))

    def x_tick(self, v, s):
        x = self.x_of(v)
        mark = Line([x, self.bottom, 0], [x, self.bottom - 0.07, 0], stroke_color=MUTED, stroke_width=1.1)
        lbl = crisp_text(s, font_size=13, color=MUTED).move_to([x, self.bottom - 0.22, 0])
        return VGroup(mark, lbl)

    def y_tick(self, v, s):
        y = self.y_of(v)
        mark = Line([self.left, y, 0], [self.left - 0.07, y, 0], stroke_color=MUTED, stroke_width=1.1)
        if s is None:
            return VGroup(mark)
        lbl = crisp_text(s, font_size=12, color=MUTED).move_to([self.left - 0.22, y, 0], aligned_edge=RIGHT)
        return VGroup(mark, lbl)

    def build(self):
        self.rect = Rectangle(width=self.w, height=self.h, stroke_color=MUTED, stroke_width=1.3)
        self.rect.move_to([(self.left + self.right) / 2, (self.bottom + self.top) / 2, 0])
        self.xticks = VGroup(*[self.x_tick(v, s) for v, s in self.xtick_defs])
        self.yticks = VGroup(*[self.y_tick(v, s) for v, s in self.ytick_defs])
        self.xlabel = crisp_text(self.xlabel_text, font_size=14, color=MUTED)
        self.xlabel.move_to([(self.left + self.right) / 2, self.bottom - 0.42, 0])
        if self.ylabel_text:
            self.ylabel = crisp_text(self.ylabel_text, font_size=13, color=MUTED).rotate(PI / 2)
            labelled = [t[1] for t in self.yticks if len(t) > 1]
            label_left = min(l.get_left()[0] for l in labelled) if labelled else self.left
            self.ylabel.move_to([label_left - 0.16 - self.ylabel.width / 2, (self.bottom + self.top) / 2, 0])
        else:
            self.ylabel = VGroup()
        self.frame = VGroup(self.rect, self.xticks, self.yticks, self.xlabel, self.ylabel)
        return self.frame


# ---------------------------------------------------------------------------
class FiringRateScore(Scene):
    def construct(self):
        self.prepare_data()
        self.intro()
        self.section_raster()
        self.section_avg_freq()
        self.section_gaussian_worked()
        self.section_all_subtypes()
        self.finale()

    # ------------------------------------------------------------- data prep
    def prepare_data(self):
        self.spikes = load_spikes(SPIKES_CSV)
        t_end = max(ms for g in GROUPS for _, ms in self.spikes[g]) + 1
        self.t_start, self.t_end = t_end - WINDOW_MS, t_end
        self.n_rows = {g: max(nid for nid, _ in self.spikes[g]) + 1 for g in GROUPS}
        offsets, cum = {}, 0
        for g in GROUPS:
            offsets[g] = cum
            cum += self.n_rows[g]
        self.offsets, self.total_neurons = offsets, cum

        self.rate = {}
        for g in GROUPS:
            count = sum(1 for _, ms in self.spikes[g] if self.t_start <= ms < self.t_end)
            self.rate[g] = count / self.n_rows[g] / (WINDOW_MS / 1000)

        groupInfo, _, firingRates = getCA3FullScale()
        self.target = {info["nType"]: firingRates[i] for i, info in groupInfo.items()
                       if info["nType"] in GROUPS}
        self.sigma = {g: 0.1 * self.target[g] for g in GROUPS}

        self.score = {g: float(norm_gauss(self.rate[g], self.target[g], self.sigma[g])) for g in GROUPS}
        self.final_score = sum(self.score.values()) / len(GROUPS)

        self.max_rate = max(max(self.rate.values()), max(self.target.values())) * 1.1

    # -------------------------------------------------------------- chrome
    def intro(self, animate=True):
        self.title = crisp_text("Firing rate score", font_size=30, weight=BOLD).move_to(UP * TITLE_Y)
        if not animate:
            self.add(self.title)
            return
        sub = crisp_text(f"CA3_0_0001_3_group_18  ·  mean firing rate over the final {WINDOW_MS} ms",
                         font_size=16, color=MUTED).next_to(self.title, DOWN, buff=0.12)
        self.play(FadeIn(self.title, shift=DOWN * 0.15), FadeIn(sub))
        self.wait(0.6)
        self.play(FadeOut(sub))

    def set_caption(self, text_str, animate=True):
        new_caption = fit(crisp_text(text_str, font_size=20, color=ACCENT), 11.6).move_to(UP * CAPTION_Y)
        if not animate:
            if hasattr(self, "caption"):
                self.remove(self.caption)
            self.add(new_caption)
        elif hasattr(self, "caption"):
            self.play(ReplacementTransform(self.caption, new_caption), run_time=0.4)
        else:
            self.play(FadeIn(new_caption), run_time=0.4)
        self.caption = new_caption

    # --------------------------------------------------------------- raster
    def section_raster(self, animate=True):
        self.set_caption(f"The last {WINDOW_MS} ms of the recording, by subtype", animate=animate)
        panel = Panel(FULL_BOX, (self.t_start, self.t_end), (0, self.total_neurons),
                     [(t, str(t)) for t in (self.t_start, (self.t_start + self.t_end) // 2, self.t_end)],
                     [], "Time (ms)", "Global Neuron ID")
        panel.build()

        dividers = VGroup(*[
            Line([panel.left, panel.y_of(self.offsets[g]), 0], [panel.right, panel.y_of(self.offsets[g]), 0],
                stroke_color=MUTED, stroke_width=1, stroke_opacity=0.5)
            for g in GROUPS[1:]
        ])

        dot_r = min(0.03, panel.h / self.total_neurons * 0.7)
        dots = VGroup()
        for g in GROUPS:
            for nid, ms in self.spikes[g]:
                if self.t_start <= ms < self.t_end:
                    dots.add(Dot(point=panel.c2p(ms, self.offsets[g] + nid + 0.5), radius=dot_r, color=COLORS[g]))

        legend = VGroup(*[
            VGroup(Dot(radius=0.06, color=COLORS[g]),
                  crisp_text(f"{label(g)} ({self.n_rows[g]})", font_size=16, color=MUTED)).arrange(RIGHT, buff=0.1)
            for g in GROUPS
        ]).arrange(RIGHT, buff=0.5).next_to(panel.rect, UP, buff=0.12)

        if animate:
            self.play(Create(panel.frame), Create(dividers), FadeIn(legend))
            self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.0006, run_time=1.8))
            self.wait(0.6)
        else:
            self.add(panel.frame, dividers, dots, legend)

        self.raster_group = VGroup(panel.frame, dividers, dots, legend)

    # --------------------------------------------------- side-thumbnail flow
    def shrink_to_side(self, mob):
        s = min(THUMB_W / mob.width, THUMB_H / mob.height)
        return mob.animate.scale(s).move_to(THUMB_CENTER)

    def side_arrow(self):
        return Arrow([ARROW_X[0], THUMB_CENTER[1], 0], [ARROW_X[1], THUMB_CENTER[1], 0], buff=0,
                     color=ACCENT, stroke_width=5, max_tip_length_to_length_ratio=0.35)

    def advance(self, finished, animate=True):
        """Shrink the finished graph into the side slot (cross-fading out the
        previous thumbnail) and point an arrow at where the next one goes."""
        old = [m for m in (getattr(self, "thumb", None), getattr(self, "arrow", None)) if m]
        if not animate:
            for m in old:
                self.remove(m)
            s = min(THUMB_W / finished.width, THUMB_H / finished.height)
            finished.scale(s).move_to(THUMB_CENTER)
            self.thumb, self.arrow = finished, self.side_arrow()
            self.add(self.thumb, self.arrow)
            return
        self.play(self.shrink_to_side(finished), *[FadeOut(m) for m in old], run_time=1.1)
        self.thumb, self.arrow = finished, self.side_arrow()
        self.play(GrowArrow(self.arrow), run_time=0.5)

    # ---------------------------------------------------------- avg freq bars
    def section_avg_freq(self, animate=True):
        self.set_caption("Average firing rate per subtype vs. the full-scale target", animate=animate)
        self.advance(self.raster_group, animate=animate)

        left, bottom, right, top = MAIN_BOX
        label_w = 2.0
        axis_left, axis_right = left + label_w, right - 0.3
        axis_w = axis_right - axis_left

        def x_of_rate(v):
            return axis_left + v / self.max_rate * axis_w

        row_h = (top - bottom) / len(GROUPS)
        bar_h = row_h * 0.4

        axis_ticks = VGroup(*[
            VGroup(Line([x_of_rate(v), bottom, 0], [x_of_rate(v), bottom - 0.07, 0],
                       stroke_color=MUTED, stroke_width=1.1),
                  crisp_text(s, font_size=13, color=MUTED).move_to([x_of_rate(v), bottom - 0.22, 0]))
            for v, s in nice_ticks(self.max_rate)
        ])
        axis_line = Line([axis_left, bottom, 0], [axis_right, bottom, 0], stroke_color=MUTED, stroke_width=1.3)
        axis_lbl = crisp_text("Firing rate (Hz)", font_size=15, color=MUTED).move_to(
            [(axis_left + axis_right) / 2, bottom - 0.55, 0])
        if animate:
            self.play(Create(axis_line), FadeIn(axis_ticks), FadeIn(axis_lbl))
        else:
            self.add(axis_line, axis_ticks, axis_lbl)

        self.bar_rows = {}
        for i, g in enumerate(GROUPS):
            y = top - (i + 0.5) * row_h
            name_lbl = VGroup(Dot(radius=0.07, color=COLORS[g]), crisp_text(label(g), font_size=18)) \
                .arrange(RIGHT, buff=0.1).move_to([left + label_w / 2, y, 0])

            bar1 = Rectangle(width=max(x_of_rate(self.rate[g]) - axis_left, 0.001), height=bar_h,
                             stroke_width=0, fill_color=COLORS[g], fill_opacity=0.85)
            bar1.move_to([axis_left, y, 0], aligned_edge=LEFT)

            tick_x = x_of_rate(self.target[g])
            value_lbl = crisp_text(f"{self.rate[g]:.2f} Hz", font_size=16, color=COLORS[g])
            value_lbl.move_to([max(x_of_rate(self.rate[g]), tick_x) + 0.15, y, 0], aligned_edge=LEFT)

            target_tick = DashedLine([tick_x, y - bar_h / 2 - 0.1, 0], [tick_x, y + bar_h / 2 + 0.1, 0],
                                     stroke_color=ACCENT, stroke_width=2.2, dash_length=0.06)
            target_lbl = crisp_text(f"target {self.target[g]:.2f} Hz", font_size=14, color=ACCENT)
            target_lbl.next_to(target_tick, UP, buff=0.06)
            if target_lbl.get_right()[0] > axis_right:
                target_lbl.next_to(target_tick, UP, buff=0.06).align_to(target_tick, RIGHT)

            if animate:
                bar0 = Rectangle(width=0.001, height=bar_h, stroke_width=0, fill_color=COLORS[g], fill_opacity=0.85)
                bar0.move_to([axis_left, y, 0], aligned_edge=LEFT)
                self.play(FadeIn(name_lbl), run_time=0.3)
                self.add(bar0)
                self.play(Transform(bar0, bar1), FadeIn(value_lbl), run_time=0.6)
                self.play(Create(target_tick), FadeIn(target_lbl), run_time=0.4)
                self.bar_rows[g] = VGroup(name_lbl, bar0, value_lbl, target_tick, target_lbl)
            else:
                self.add(name_lbl, bar1, value_lbl, target_tick, target_lbl)
                self.bar_rows[g] = VGroup(name_lbl, bar1, value_lbl, target_tick, target_lbl)

        if animate:
            self.wait(0.6)
        self.bars_group = VGroup(axis_line, axis_ticks, axis_lbl, *self.bar_rows.values())

    # ------------------------------------------------------- gaussian, worked
    def xmax_for(self, g):
        mu, sigma = self.target[g], self.sigma[g]
        return max(mu + 4 * sigma, self.rate[g] * 1.15, mu * 1.3)

    def section_gaussian_worked(self, animate=True):
        if animate:
            self.set_caption("Score = a normalized Gaussian centered on the target, worked for Pyramidal")
        self.advance(self.bars_group, animate=animate)
        if animate:
            self.play(Indicate(self.bar_rows["Pyramidal"], color=ACCENT, scale_factor=1.08))

        g = "Pyramidal"
        mu, sigma = self.target[g], self.sigma[g]
        xmax = self.xmax_for(g)
        panel = Panel(MAIN_BOX, (0, xmax), (0, 1.3),
                     nice_ticks(xmax), [(0, "0"), (0.5, "0.5"), (1.0, "1.0")],
                     "Firing rate (Hz)", "Normalized score")
        panel.build()

        formula = crisp_text("Normalized score(x) = exp( -(x - target)² / (2 · (0.1·target)²) )",
                             font_size=18, color=ACCENT).next_to(panel.rect, UP, buff=0.08)

        xs = np.linspace(0, xmax, 300)
        curve = panel.curve(xs, norm_gauss(xs, mu, sigma), COLORS[g], width=3)

        band = Rectangle(width=panel.x_of(mu + sigma) - panel.x_of(mu - sigma), height=panel.h,
                         stroke_width=0, fill_color=COLORS[g], fill_opacity=0.08)
        band.move_to([panel.x_of(mu), (panel.bottom + panel.top) / 2, 0])
        mu_line = DashedLine([panel.x_of(mu), panel.bottom, 0], [panel.x_of(mu), panel.y_of(1.0), 0],
                             stroke_color=COLORS[g], stroke_width=2, dash_length=0.07)
        mu_lbl = crisp_text(f"target = {mu:.2f} Hz", font_size=15, color=COLORS[g]).next_to(mu_line, UP, buff=0.08)
        sigma_lbl = crisp_text(f"σ = 0.1 × target = {sigma:.3f} Hz", font_size=14, color=MUTED)
        sigma_lbl.move_to([panel.x_of(mu), panel.bottom - 0.75, 0])

        actual_x = panel.x_of(self.rate[g])
        actual_line = DashedLine([actual_x, panel.bottom, 0], [actual_x, panel.y_of(1.0), 0],
                                 stroke_color=MUTED, stroke_width=2, dash_length=0.07)
        actual_lbl = crisp_text(f"measured = {self.rate[g]:.2f} Hz", font_size=15, color=MUTED)
        actual_lbl.next_to(actual_line, UP, buff=0.08)
        if actual_lbl.get_right()[0] > panel.right:
            actual_lbl.align_to(panel.rect, RIGHT)

        score_dot = Dot(panel.c2p(self.rate[g], self.score[g]), radius=0.09, color=ACCENT)
        score_lbl = crisp_text(f"F({label(g)}) = {self.score[g]:.3f}", font_size=20, weight=BOLD, color=ACCENT)
        score_lbl.next_to(panel.rect, DOWN, buff=0.5)

        if animate:
            self.play(Create(panel.frame))
            self.play(FadeIn(formula))
            self.play(Create(curve), run_time=1.4)
            self.play(FadeIn(band), Create(mu_line), FadeIn(mu_lbl), FadeIn(sigma_lbl))
            self.wait(0.5)
            self.set_caption(f"Read the curve at the measured rate ({self.rate[g]:.2f} Hz)")
            self.play(FadeOut(sigma_lbl), Create(actual_line), FadeIn(actual_lbl))
            self.play(FadeIn(score_dot, scale=0.5), FadeIn(score_lbl))
            self.wait(0.8)
        else:
            self.set_caption(f"Read the curve at the measured rate ({self.rate[g]:.2f} Hz)", animate=False)
            self.add(panel.frame, formula, curve, band, mu_line, mu_lbl, actual_line, actual_lbl,
                     score_dot, score_lbl)

        self.worked = dict(frame=panel.frame, curve=curve, mu_line=mu_line, actual_line=actual_line,
                           dot=score_dot, score_lbl=score_lbl)
        self.worked_extras = VGroup(formula, band, mu_lbl, actual_lbl)

    # --------------------------------------------------------- all subtypes
    def mini_pieces(self, i, g):
        """The small Gaussian panel for subtype g in slot i. The panels share a
        y-scale, so only the leftmost one carries y-axis labels."""
        box = GAUSS_BOXES[i]
        mu, sigma = self.target[g], self.sigma[g]
        xmax = self.xmax_for(g)
        labelled = i == 0
        panel = Panel(box, (0, xmax), (0, 1.15), nice_ticks(xmax, 3),
                     [(0, "0" if labelled else None), (1.0, "1.0" if labelled else None)],
                     "Hz", "score" if labelled else None)
        panel.build()

        xs = np.linspace(0, xmax, 220)
        cx = (box[0] + box[2]) / 2
        readout = crisp_text(f"target {mu:.2f} · measured {self.rate[g]:.2f}", font_size=12, color=MUTED)
        return dict(
            frame=panel.frame,
            title=crisp_text(label(g), font_size=17, weight=BOLD, color=COLORS[g]).next_to(panel.rect, UP, buff=0.08),
            curve=panel.curve(xs, norm_gauss(xs, mu, sigma), COLORS[g], width=2.5),
            mu_line=DashedLine([panel.x_of(mu), panel.bottom, 0], [panel.x_of(mu), panel.y_of(1.0), 0],
                               stroke_color=COLORS[g], stroke_width=1.6, dash_length=0.05),
            actual_line=DashedLine([panel.x_of(self.rate[g]), panel.bottom, 0],
                                   [panel.x_of(self.rate[g]), panel.y_of(1.0), 0],
                                   stroke_color=MUTED, stroke_width=1.6, dash_length=0.05),
            dot=Dot(panel.c2p(self.rate[g], self.score[g]), radius=0.07, color=ACCENT),
            readout=readout.move_to([cx, READOUT_ROW_Y, 0]),
            score_lbl=crisp_text(f"F({label(g)}) = {self.score[g]:.3f}", font_size=18, weight=BOLD,
                                 color=ACCENT).move_to([cx, SCORE_ROW_Y, 0]),
        )

    def section_all_subtypes(self):
        self.set_caption("Every subtype gets its own curve, scaled to its own target")

        self.mini_dots = {}
        first = self.mini_pieces(0, GROUPS[0])
        self.play(FadeOut(self.thumb), FadeOut(self.arrow), FadeOut(self.worked_extras),
                  *[ReplacementTransform(self.worked[k], first[k]) for k in self.worked],
                  run_time=1.3)
        self.play(FadeIn(first["title"]), FadeIn(first["readout"]), run_time=0.5)
        self.mini_dots[GROUPS[0]] = first["dot"]

        for i, g in enumerate(GROUPS[1:], start=1):
            p = self.mini_pieces(i, g)
            self.mini_dots[g] = p["dot"]
            self.play(Create(p["frame"]), FadeIn(p["title"]), run_time=0.5)
            self.play(Create(p["curve"]), run_time=0.9)
            self.play(Create(p["mu_line"]), Create(p["actual_line"]), FadeIn(p["readout"]), run_time=0.5)
            self.play(FadeIn(p["dot"], scale=0.5), FadeIn(p["score_lbl"]), run_time=0.4)

        self.wait(0.6)

    # ------------------------------------------------------------- finale
    def finale(self):
        self.set_caption("Combine the three subtype scores into the firing rate score F(N)")

        terms = " + ".join(f"{self.score[g]:.3f}" for g in GROUPS)
        summary = VGroup(
            crisp_text(f"F(N) = (1/{len(GROUPS)}) × ({terms}) = {self.final_score:.3f}",
                      font_size=22, weight=BOLD, color=ACCENT),
            crisp_text("F(N) is weighted 0.1 in the overall objective O(N) = 0.4·A(N) + 0.4·P(N) + 0.1·F(N) + 0.1·C(N)",
                      font_size=13, color=MUTED),
        ).arrange(DOWN, buff=0.15)
        fit(summary, FULL_BOX[2] - FULL_BOX[0])
        summary.move_to([(FULL_BOX[0] + FULL_BOX[2]) / 2, FINALE_Y, 0])

        self.play(FadeIn(summary), *[Flash(self.mini_dots[g], color=ACCENT, flash_radius=0.22) for g in GROUPS])
        self.wait(3)


# ============================================================== clips 1-3
class Clip1(FiringRateScore):
    """Raster -> per-subtype firing rate vs. target, as a bar chart."""

    def construct(self):
        self.prepare_data()
        self.intro()
        self.section_raster()
        self.section_avg_freq()
        self.wait(1)


class Clip2(FiringRateScore):
    """The Gaussian score worked in full for one subtype (Pyramidal)."""

    def construct(self):
        self.prepare_data()
        self.intro(animate=False)
        self.section_raster(animate=False)
        self.section_avg_freq(animate=False)
        self.section_gaussian_worked()
        self.wait(1)


class Clip3(FiringRateScore):
    """Every subtype's curve, combined into the firing rate score F(N)."""

    def construct(self):
        self.prepare_data()
        self.intro(animate=False)
        self.section_raster(animate=False)
        self.section_avg_freq(animate=False)
        self.section_gaussian_worked(animate=False)
        self.section_all_subtypes()
        self.finale()
