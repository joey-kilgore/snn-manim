"""
activity_score_scene.py

Criterion 1 of the CA3-scaling scoring rubric (arXiv:2609.16429): the
activity score. The final 1 second of the simulation is split into 66ms
bins (15 bins, chosen so each bin covers at least one cycle of the
full-scale network's ~16Hz rhythm), separately per neuron subtype. A(N) is
the squared fraction of those (subtype, bin) cells that contain at least
one spike:

    A(N) = ( 1/(#Groups * #Bins) * sum_g sum_b S_g,b )^2,   S_g,b in {0,1}

S_g,b is 1 if subtype g has >=1 spike in bin b, else 0. Squaring rewards
filling every bin over every subtype equally (a network that is fully
active in half its bins scores 0.25, not 0.5).

Left panel: the CA3_0_0001_3_group_18 raster over its last 1000 ms (reusing
the data loader, groups and colors from raster_scene.py). Right panel: the
15-bin grid per subtype, filled in live as each cell is checked, plus the
running fraction/score and where that score sits on the A(N) curve.

Render:
    manim -pql activity_score_scene.py ActivityScore     # quick preview
    manim -pqh activity_score_scene.py ActivityScore     # 1080p

Split into two clips (clip 1 ends as the first cell is highlighted as
active; clip 2 opens on that same frame and runs to the end):
    manim -pqh activity_score_scene.py ActivityScoreClip1 ActivityScoreClip2
"""

import numpy as np
from manim import *

from raster_scene import GROUPS, COLORS, MUTED, load_spikes, SPIKES_CSV, fit, label

config.background_color = WHITE

N_BINS = 15
BIN_MS = 1000 / N_BINS    # ~66.67 ms, "66ms bins" in the paper
TOTAL_CELLS = len(GROUPS) * N_BINS

DETAILED = 5              # first N (subtype, bin) checks are narrated slowly

ACTIVE_INACTIVE = "#D8DCE3"
HILITE = "#9C6E0C"
CURVE_COLOR = BLUE_E

# ---------------------------------------------------------------------------
# Layout (frame: x in [-7.11, 7.11], y in [-4, 4])
# ---------------------------------------------------------------------------
RASTER_LEFT, RASTER_RIGHT = -7.0, -0.5
RASTER_TOP, RASTER_BOTTOM = 2.4, -3.0

RIGHT_LEFT, RIGHT_RIGHT = -0.05, 6.95

GRID_LABEL_W = 1.0
GRID_TOP, GRID_BOTTOM = 2.4, 1.3
GRID_LEFT = RIGHT_LEFT + GRID_LABEL_W
GRID_RIGHT = RIGHT_RIGHT

PANEL_CX = (RIGHT_LEFT + RIGHT_RIGHT) / 2
STAT_CX = (GRID_LEFT + GRID_RIGHT) / 2
STAT_Y = [0.5, 0.1, -0.3]
MSG_Y = -0.75

PLOT_SIDE = 1.85
PLOT_X0, PLOT_Y0 = PANEL_CX - PLOT_SIDE / 2, -3.0

# grid rows top -> bottom mirror the raster's bottom -> top group stacking
GRID_ROWS = list(reversed(GROUPS))

Text.set_default(font="Segoe UI", color=BLACK)


def load_window():
    spikes = load_spikes(SPIKES_CSV)
    t_end = max(ms for g in GROUPS for _, ms in spikes[g]) + 1
    t_start = t_end - 1000
    n_rows = {g: max(nid for nid, _ in spikes[g]) + 1 for g in GROUPS}
    offsets, cum = {}, 0
    for g in GROUPS:
        offsets[g] = cum
        cum += n_rows[g]
    return spikes, t_start, t_end, n_rows, offsets, cum


def compute_bins(spikes, t_start):
    """S[g] = [bool]*N_BINS: whether subtype g has >=1 spike in bin b."""
    S = {}
    for g in GROUPS:
        bins = [False] * N_BINS
        for _, ms in spikes[g]:
            if t_start <= ms < t_start + N_BINS * BIN_MS:
                bins[min(int((ms - t_start) / BIN_MS), N_BINS - 1)] = True
        S[g] = bins
    return S


class ActivityScore(Scene):
    CLIP = None   # None = whole scene; 1 / 2 = before / after the first highlighted cell

    def construct(self):
        if self.CLIP == 2:
            self.next_section("setup", skip_animations=True)
        self.spikes, self.t_start, self.t_end, self.n_rows, self.offsets, self.total_neurons = load_window()
        self.S = compute_bins(self.spikes, self.t_start)

        self.intro()
        self.build_raster()
        self.build_bins_overlay()
        self.build_grid()
        self.count_bins()
        self.finale()

    # ------------------------------------------------------------- geometry
    def x_of(self, t):
        return RASTER_LEFT + (t - self.t_start) / 1000 * (RASTER_RIGHT - RASTER_LEFT)

    def row_y(self, g, local_id):
        frac = (self.offsets[g] + local_id + 0.5) / self.total_neurons
        return RASTER_BOTTOM + frac * (RASTER_TOP - RASTER_BOTTOM)

    def band(self, g):
        raster_h = RASTER_TOP - RASTER_BOTTOM
        y0 = RASTER_BOTTOM + self.offsets[g] / self.total_neurons * raster_h
        y1 = RASTER_BOTTOM + (self.offsets[g] + self.n_rows[g]) / self.total_neurons * raster_h
        return y0, y1

    def cell_rect(self, g, b):
        y0, y1 = self.band(g)
        x0, x1 = self.x_of(self.t_start + b * BIN_MS), self.x_of(self.t_start + (b + 1) * BIN_MS)
        return Rectangle(width=x1 - x0, height=y1 - y0, stroke_width=0,
                         fill_color=HILITE, fill_opacity=0.22).move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])

    def plot_of(self, x, y):
        return [PLOT_X0 + x * PLOT_SIDE, PLOT_Y0 + y * PLOT_SIDE, 0]

    # ------------------------------------------------------------- sections
    def intro(self):
        self.header = Text("Activity score", font_size=52, weight=BOLD).to_edge(UP, buff=0.12).scale(0.5)
        self.play(FadeIn(self.header))

    def build_raster(self):
        raster_w, raster_h = RASTER_RIGHT - RASTER_LEFT, RASTER_TOP - RASTER_BOTTOM
        self.raster_box = Rectangle(width=raster_w, height=raster_h, stroke_color=MUTED,
                                    stroke_width=1.5)
        self.raster_box.move_to([(RASTER_LEFT + RASTER_RIGHT) / 2, (RASTER_TOP + RASTER_BOTTOM) / 2, 0])

        dividers = VGroup(*[
            Line([RASTER_LEFT, RASTER_BOTTOM + self.offsets[g] / self.total_neurons * raster_h, 0],
                 [RASTER_RIGHT, RASTER_BOTTOM + self.offsets[g] / self.total_neurons * raster_h, 0],
                 stroke_color=MUTED, stroke_width=1, stroke_opacity=0.5)
            for g in GROUPS[1:]
        ])

        dot_r = min(0.028, raster_h / self.total_neurons * 0.7)
        dots = VGroup()
        for g in GROUPS:
            for nid, ms in self.spikes[g]:
                if self.t_start <= ms < self.t_end:
                    dots.add(Dot(point=[self.x_of(ms), self.row_y(g, nid), 0], radius=dot_r,
                                color=COLORS[g]))

        tick_ts = (self.t_start, (self.t_start + self.t_end) // 2, self.t_end)
        tick_edges = (LEFT, None, RIGHT)
        time_ticks = VGroup(*[
            Text(f"{t}", font_size=30, color=MUTED).scale(0.5).move_to(
                [self.x_of(t), RASTER_BOTTOM - 0.4, 0], aligned_edge=(edge if edge is not None else ORIGIN)
            )
            for t, edge in zip(tick_ts, tick_edges)
        ])
        time_lbl = Text("Time (ms) · final 1000 ms of the run", font_size=32, color=MUTED).scale(0.5)
        time_lbl.next_to(time_ticks, DOWN, buff=0.08)

        legend = VGroup(*[
            VGroup(Dot(radius=0.05, color=COLORS[g]), Text(label(g), font_size=28, color=MUTED).scale(0.5))
            .arrange(RIGHT, buff=0.06)
            for g in GROUPS
        ]).arrange(RIGHT, buff=0.3).next_to(self.raster_box, UP, buff=0.1)

        self.play(Create(self.raster_box), Create(dividers), FadeIn(legend))
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.0008, run_time=1.8))
        self.play(FadeIn(time_ticks), FadeIn(time_lbl))
        self.wait(0.5)

    def build_bins_overlay(self):
        cap = fit(Text(f"Bin the window into {N_BINS} × 66 ms increments, per subtype",
                       font_size=44, color=HILITE).scale(0.5), 6.4)
        cap.next_to(self.header, DOWN, buff=0.15)
        self.play(FadeIn(cap))

        bin_lines = VGroup(*[
            DashedLine([self.x_of(self.t_start + i * BIN_MS), RASTER_TOP, 0],
                       [self.x_of(self.t_start + i * BIN_MS), RASTER_BOTTOM, 0],
                       stroke_color=HILITE, stroke_width=3.0, stroke_opacity=1.0, dash_length=0.2)
            for i in range(1, N_BINS)
        ])
        bin_labels = VGroup(*[
            Text(str(b + 1), font_size=24, color=MUTED)
            .move_to([self.x_of(self.t_start + (b + 0.5) * BIN_MS), RASTER_BOTTOM - 0.18, 0])
            .scale(0.5)
            for b in range(N_BINS)
        ])
        self.play(LaggedStart(*[Create(l) for l in bin_lines], lag_ratio=0.06), run_time=1.2)
        self.play(FadeIn(bin_labels))
        self.play(FadeOut(cap))
        self.wait(0.3)

    def build_grid(self):
        row_h = (GRID_TOP - GRID_BOTTOM) / len(GROUPS)
        col_w = (GRID_RIGHT - GRID_LEFT) / N_BINS

        def center(r, b):
            return [GRID_LEFT + (b + 0.5) * col_w, GRID_TOP - (r + 0.5) * row_h, 0]

        self.cells = [[
            Square(min(col_w, row_h) * 0.8, stroke_color=MUTED, stroke_width=1,
                  fill_color=ACTIVE_INACTIVE, fill_opacity=1).move_to(center(r, b))
            for b in range(N_BINS)
        ] for r in range(len(GRID_ROWS))]

        row_labels = VGroup(*[
            Text(label(g), font_size=28, color=COLORS[g]).move_to([RIGHT_LEFT + GRID_LABEL_W / 2 - 0.05,
                                                             GRID_TOP - (r + 0.5) * row_h, 0]).scale(0.5)
            for r, g in enumerate(GRID_ROWS)
        ])
        for lbl in row_labels:
            fit(lbl, GRID_LABEL_W - 0.1)

        grid_group = VGroup(*[sq for row in self.cells for sq in row])
        axis_lbl = Text(f"{N_BINS} bins × {len(GROUPS)} subtypes = {TOTAL_CELLS} cells",
                        font_size=36, color=MUTED)
        axis_lbl.next_to(grid_group, DOWN, buff=0.2)
        axis_lbl.scale(0.5)
        self.play(FadeIn(row_labels), LaggedStart(*[FadeIn(sq) for sq in grid_group], lag_ratio=0.006,
                                                   run_time=1.2))
        self.play(FadeIn(axis_lbl))
        self.axis_lbl = axis_lbl

        self.count_lbl = self.stat_text(f"active cells: 0 / {TOTAL_CELLS}", STAT_Y[0])
        self.frac_lbl = self.stat_text(f"fraction = 0 / {TOTAL_CELLS} = 0.000", STAT_Y[1])
        self.score_lbl = self.stat_text("A(N) = 0.000² = 0.000", STAT_Y[2], color=HILITE, bold=True)
        self.play(FadeIn(self.count_lbl), FadeIn(self.frac_lbl), FadeIn(self.score_lbl))

        self.build_curve()
        self.wait(0.3)

    def stat_text(self, s, y, color=BLACK, bold=False):
        mob = Text(s, font_size=42, color=color, weight=BOLD if bold else NORMAL)
        return mob.move_to([STAT_CX, y, 0]).scale(0.5)

    def build_curve(self):
        pts = [self.plot_of(x, x ** 2) for x in np.linspace(0, 1, 60)]
        curve = VMobject(stroke_color=CURVE_COLOR, stroke_width=2.5)
        curve.set_points_smoothly(pts)
        x_axis = Line(self.plot_of(0, 0), self.plot_of(1, 0), stroke_color=MUTED, stroke_width=1.5)
        y_axis = Line(self.plot_of(0, 0), self.plot_of(0, 1), stroke_color=MUTED, stroke_width=1.5)
        x_lbl = Text("fraction of active cells", font_size=28, color=MUTED).scale(0.5)
        x_lbl.next_to(x_axis, DOWN, buff=0.3)
        y_lbl = Text("A(N)", font_size=28, color=MUTED).scale(0.5).rotate(PI / 2).next_to(y_axis, LEFT, buff=0.1)
        end_lbls = VGroup(*[
            Text(s, font_size=26, color=MUTED).scale(0.5).next_to(self.plot_of(x, 0), DOWN, buff=0.08)
            for s, x in (("0", 0), ("1", 1))
        ])
        y_end = Text("1", font_size=26, color=MUTED).scale(0.5).next_to(self.plot_of(0, 1), LEFT, buff=0.08)

        self.frac_tracker = ValueTracker(0)
        self.score_dot = always_redraw(lambda: Dot(
            point=self.plot_of(self.frac_tracker.get_value(), self.frac_tracker.get_value() ** 2),
            radius=0.07, color=HILITE))

        self.play(Create(x_axis), Create(y_axis), Create(curve), FadeIn(x_lbl), FadeIn(y_lbl),
                  FadeIn(end_lbls), FadeIn(y_end))
        self.add(self.score_dot)

    # --------------------------------------------------------------- counting
    def grid_row(self, g):
        return GRID_ROWS.index(g)

    def update_stats(self, count):
        frac = count / TOTAL_CELLS
        new_count = self.stat_text(f"active cells: {count} / {TOTAL_CELLS}", STAT_Y[0])
        new_frac = self.stat_text(f"fraction = {count} / {TOTAL_CELLS} = {frac:.3f}", STAT_Y[1])
        new_score = self.stat_text(f"A(N) = {frac:.3f}² = {frac ** 2:.3f}", STAT_Y[2],
                                   color=HILITE, bold=True)
        anims = [ReplacementTransform(self.count_lbl, new_count),
                 ReplacementTransform(self.frac_lbl, new_frac),
                 ReplacementTransform(self.score_lbl, new_score),
                 self.frac_tracker.animate.set_value(frac)]
        self.count_lbl, self.frac_lbl, self.score_lbl = new_count, new_frac, new_score
        return anims

    def count_bins(self):
        count = 0
        slow_done = 0
        first = True
        for g in GROUPS:
            bins = list(range(N_BINS))
            slow_bins = []
            while slow_done < DETAILED and bins:
                slow_bins.append(bins.pop(0))
                slow_done += 1
            fast_bins = bins

            for b in slow_bins:
                active = self.S[g][b]
                cell = self.cells[self.grid_row(g)][b]
                hi = self.cell_rect(g, b)
                verdict = Text("spike found → active" if active else "no spike → inactive",
                              font_size=40, color=HILITE if active else MUTED)
                verdict.move_to([PANEL_CX, MSG_Y, 0])
                verdict.scale(0.5)
                self.play(FadeIn(hi), run_time=0.35)
                self.play(FadeIn(verdict), run_time=0.3)
                if first:
                    self.clip_break()
                    first = False
                count += int(active)
                self.play(cell.animate.set_fill(COLORS[g] if active else ACTIVE_INACTIVE, opacity=1),
                          FadeOut(hi), FadeOut(verdict), *self.update_stats(count), run_time=0.45)

            if fast_bins:
                highlights = VGroup(*[self.cell_rect(g, b) for b in fast_bins])
                self.play(LaggedStart(*[FadeIn(h) for h in highlights], lag_ratio=0.05), run_time=0.6)
                fills = []
                for b in fast_bins:
                    active = self.S[g][b]
                    cell = self.cells[self.grid_row(g)][b]
                    fills.append(cell.animate.set_fill(COLORS[g] if active else ACTIVE_INACTIVE,
                                                        opacity=1))
                    count += int(active)
                self.play(LaggedStart(*fills, lag_ratio=0.05), FadeOut(highlights), run_time=0.8)
                self.play(*self.update_stats(count), run_time=0.5)

        self.final_count = count

    def clip_break(self):
        """Cut between clip 1 and clip 2: the first cell is highlighted with its verdict."""
        if self.CLIP == 1:
            self.next_section("rest", skip_animations=True)
        elif self.CLIP == 2:
            self.next_section("clip2")

    def finale(self):
        frac = self.final_count / TOTAL_CELLS
        score = frac ** 2
        if self.final_count == TOTAL_CELLS:
            msg = "Every cell is active: a perfect activity score"
        else:
            msg = f"{self.final_count} / {TOTAL_CELLS} cells active"
        summary = VGroup(
            Text(msg, font_size=44, weight=BOLD, color=HILITE).scale(0.5),
        ).arrange(DOWN, buff=0.15)
        fit(summary, RIGHT_RIGHT - RIGHT_LEFT - 0.4)
        summary.move_to([PANEL_CX, PLOT_Y0 - 0.75, 0])
        self.play(FadeIn(summary), Indicate(self.score_lbl, color=HILITE, scale_factor=1.15),
                  Flash(self.score_dot, color=HILITE, flash_radius=0.25))
        self.wait(3)


class ActivityScoreClip1(ActivityScore):
    CLIP = 1


class ActivityScoreClip2(ActivityScore):
    CLIP = 2
