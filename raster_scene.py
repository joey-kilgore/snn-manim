"""
raster_scene.py

Animates the last several hundred ms of the 3-group CA3 recording in
data/CA3_0_0001_3_group_18/: a raster plot that builds up spike by spike as
time sweeps forward, with each group's representative-neuron voltage trace
drawn in sync underneath.

Raster style matches the project's existing matplotlib rasters: one row per
neuron (equal spacing per neuron, not per group), spikes drawn as dots, rows
stacked by group along a single "Global Neuron ID" axis - so a small group
naturally shows up as a thin band, a big group as a tall one.

Data:
    detailed_spikes.csv  (group, neuron_id, spike_ms) for every spike
    voltages.csv         each group's mean voltage, sampled every 1 ms

Render:
    manim -pql raster_scene.py RasterReveal     # quick preview
    manim -pqh raster_scene.py RasterReveal     # 1080p
"""

import csv
from pathlib import Path

import numpy as np
from manim import *

config.background_color = WHITE

DATA_DIR = Path(__file__).parent / "data" / "CA3_0_0001_3_group_18"
SPIKES_CSV = DATA_DIR / "detailed_spikes.csv"
VOLTAGES_CSV = DATA_DIR / "voltages.csv"  # per-group mean voltage, not a single representative neuron

WINDOW_MS = 1000         # length of the window the sweep reveals
SWEEP_TIME = 8.0         # seconds the sweep itself takes on screen
FAST_SWEEP = False       # True halves SWEEP_TIME, for quick previews

# raster rows, bottom to top: the big group first so it forms the tall block
# at the bottom, same as the reference matplotlib plots.
GROUPS = ["Pyramidal", "Basket", "MFA_ORDEN"]

# same palette used for the matplotlib rasters elsewhere in this project.
NEURON_COLORS = {
    "Pyramidal":     "#000000",  # black, same as the source palette
    "Basket":        "#1D6FA4",  # darker blue
    "MFA_ORDEN":     "#C5003E",  # deep magenta
    "Bistratified":  "#007A55",  # darker teal
    "QuadD-LM":      "#E8A000",  # deeper gold
    "Ivy":           "#89CFF0",  # light blue
    "Axo-axonic":    "#FF6B00",  # bright orange
    "Basket_CCK+":   "#D4AAFF",  # light purple
}
COLORS = {g: NEURON_COLORS[g] for g in GROUPS}

MUTED = GREY_D
PLAYHEAD = "#9C6E0C"

# ---------------------------------------------------------------------------
# Layout (manim default frame: x in [-7.11, 7.11], y in [-4, 4])
# ---------------------------------------------------------------------------
TITLE_Y = 3.7
RASTER_TOP, RASTER_BOTTOM = 2.7, -1.6
VOLT_TOP, VOLT_BOTTOM = -1.8, -2.8
AXIS_LABEL_Y = -3.0
TIME_LABEL_Y = -3.3
LEGEND_Y = -3.65

PANEL_LEFT, PANEL_RIGHT = -5.3, 5.3
Y_TICK_X = PANEL_LEFT - 0.3
VOLT_AXIS_X = PANEL_LEFT - 0.2

Text.set_default(font="Segoe UI", color=BLACK)


def fit(mob, width):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def label(g):
    """Display name for a subtype key (MFA_ORDEN -> MFA ORDEN)."""
    return g.replace("_", " ")


def nice_row_ticks(n, target=5):
    """Round tick step (10/20/25/40/50/100/200/...) giving ~`target` ticks
    from 0 up to (but not necessarily including) n."""
    candidates = [5, 10, 20, 25, 40, 50, 100, 200, 250, 500]
    step = next((c for c in candidates if n / c <= target), candidates[-1])
    ticks = list(range(0, n, step))
    return ticks


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_spikes(path):
    """{group: [(neuron_id, spike_ms), ...]} over the whole recording."""
    spikes = {g: [] for g in GROUPS}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            _, group, nid, ms = row
            if group in spikes:
                spikes[group].append((int(nid), int(ms)))
    return spikes


def load_voltages(path):
    """{group: (ms_array, voltage_array)}, sorted by ms, whole recording."""
    rows_by_group = {g: [] for g in GROUPS}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            _, _, global_ms, _, group, v = row
            if group in rows_by_group:
                rows_by_group[group].append((int(global_ms), float(v)))
    out = {}
    for g, rows in rows_by_group.items():
        rows.sort(key=lambda r: r[0])
        ms = np.array([r[0] for r in rows])
        v = np.array([r[1] for r in rows])
        out[g] = (ms, v)
    return out


class RasterReveal(Scene):
    def construct(self):
        spikes = load_spikes(SPIKES_CSV)
        voltages = load_voltages(VOLTAGES_CSV)

        t_end = max(ms for g in GROUPS for _, ms in spikes[g]) + 1
        t_start = t_end - WINDOW_MS

        n_rows = {g: max(nid for nid, _ in spikes[g]) + 1 for g in GROUPS}

        # global row offset of each group, stacked bottom (Pyramidal) to top
        offsets, cum = {}, 0
        for g in GROUPS:
            offsets[g] = cum
            cum += n_rows[g]
        total_neurons = cum

        v_min = min(v[(ms >= t_start) & (ms < t_end)].min()
                    for ms, v in voltages.values())
        v_max = max(v[(ms >= t_start) & (ms < t_end)].max()
                    for ms, v in voltages.values())
        v_pad = 0.08 * (v_max - v_min)
        v_min, v_max = v_min - v_pad, v_max + v_pad

        def x_of(t):
            return PANEL_LEFT + (t - t_start) / WINDOW_MS * (PANEL_RIGHT - PANEL_LEFT)

        def y_of_v(v):
            return VOLT_BOTTOM + (v - v_min) / (v_max - v_min) * (VOLT_TOP - VOLT_BOTTOM)

        raster_h = RASTER_TOP - RASTER_BOTTOM

        def row_y(g, local_id):
            frac = (offsets[g] + local_id + 0.5) / total_neurons
            return RASTER_BOTTOM + frac * raster_h

        # ------------------------------------------------------------ header
        title = Text(f"CA3 network activity: last {WINDOW_MS} ms", font_size=32, weight=BOLD)
        title.move_to(UP * TITLE_Y)
        sub = Text(f"{total_neurons} recorded neurons across "
                   f"{len(GROUPS)} groups  ·  t = {t_start}-{t_end} ms",
                   font_size=18, color=MUTED)
        sub.next_to(title, DOWN, buff=0.12)

        self.play(FadeIn(title, shift=DOWN * 0.15), FadeIn(sub))
        self.wait(0.3)

        # ------------------------------------------------------------ frame
        raster_box = Rectangle(width=PANEL_RIGHT - PANEL_LEFT, height=raster_h,
                               stroke_color=MUTED, stroke_width=1.5) \
            .move_to([(PANEL_LEFT + PANEL_RIGHT) / 2, (RASTER_TOP + RASTER_BOTTOM) / 2, 0])
        volt_box = Rectangle(width=PANEL_RIGHT - PANEL_LEFT, height=VOLT_TOP - VOLT_BOTTOM,
                             stroke_color=MUTED, stroke_width=1.5) \
            .move_to([(PANEL_LEFT + PANEL_RIGHT) / 2, (VOLT_TOP + VOLT_BOTTOM) / 2, 0])

        # thin divider between each group's block of rows
        dividers = VGroup(*[
            Line([PANEL_LEFT, RASTER_BOTTOM + offsets[g] / total_neurons * raster_h, 0],
                 [PANEL_RIGHT, RASTER_BOTTOM + offsets[g] / total_neurons * raster_h, 0],
                 stroke_color=MUTED, stroke_width=1, stroke_opacity=0.5)
            for g in GROUPS[1:]
        ])

        row_ticks = nice_row_ticks(total_neurons)
        y_ticks = VGroup(*[
            Text(str(r), font_size=14, color=MUTED)
            .move_to([Y_TICK_X, RASTER_BOTTOM + r / total_neurons * raster_h, 0], aligned_edge=RIGHT)
            for r in row_ticks
        ])
        raster_ylabel = Text("Global Neuron ID", font_size=16, color=MUTED).rotate(PI / 2) \
            .next_to(y_ticks, LEFT, buff=0.2)

        v_ticks = VGroup(*[
            Text(f"{v:.0f}", font_size=14, color=MUTED)
            .move_to([VOLT_AXIS_X, y_of_v(v), 0], aligned_edge=RIGHT)
            for v in np.linspace(v_min + v_pad, v_max - v_pad, 3)
        ])
        volt_ylabel = Text("mV", font_size=15, color=MUTED).rotate(PI / 2) \
            .next_to(v_ticks, LEFT, buff=0.15)

        time_ticks_ms = [t_start, (t_start + t_end) // 2, t_end]
        time_ticks = VGroup(*[
            Text(f"{t}", font_size=15, color=MUTED).move_to([x_of(t), AXIS_LABEL_Y, 0])
            for t in time_ticks_ms
        ])
        time_axis_lbl = Text("Time (ms)", font_size=17, color=MUTED).move_to([0, TIME_LABEL_Y, 0])

        legend = VGroup(*[
            VGroup(Dot(radius=0.07, color=COLORS[g]),
                  Text(f"{label(g)} ({n_rows[g]})", font_size=17, color=MUTED)).arrange(RIGHT, buff=0.1)
            for g in GROUPS
        ]).arrange(RIGHT, buff=0.5).move_to([0, LEGEND_Y, 0])

        self.play(Create(raster_box), Create(volt_box), Create(dividers),
                  FadeIn(y_ticks), FadeIn(raster_ylabel),
                  FadeIn(v_ticks), FadeIn(volt_ylabel),
                  FadeIn(time_ticks), FadeIn(time_axis_lbl), FadeIn(legend))
        self.wait(0.3)

        # ------------------------------------------------------------- dots
        dot_r = min(0.032, raster_h / total_neurons * 0.7)
        all_dots = VGroup()
        for g in GROUPS:
            for nid, ms in spikes[g]:
                if t_start <= ms < t_end:
                    dot = Dot(point=[x_of(ms), row_y(g, nid), 0], radius=dot_r,
                             color=COLORS[g], fill_opacity=0, stroke_width=0)
                    dot.spike_ms = ms
                    all_dots.add(dot)
        self.add(all_dots)

        # ---------------------------------------------------------- voltage
        volt_curves = VGroup()
        volt_updaters = []
        for g in GROUPS:
            ms_arr, v_arr = voltages[g]
            mask = (ms_arr >= t_start) & (ms_arr < t_end)
            ms_w, v_w = ms_arr[mask], v_arr[mask]
            pts = np.array([[x_of(m), y_of_v(v), 0] for m, v in zip(ms_w, v_w)])
            curve = VMobject(stroke_color=COLORS[g], stroke_width=2)
            if len(pts):
                curve.set_points_as_corners(pts[:1].repeat(2, axis=0))
            volt_curves.add(curve)
            volt_updaters.append((curve, ms_w, pts))
        self.add(volt_curves)

        # ----------------------------------------------------------- cursor
        cursor = ValueTracker(t_start)

        def make_dot_updater():
            def updater(group):
                t = cursor.get_value()
                for dot in group:
                    dot.set_fill(opacity=1 if dot.spike_ms <= t else 0)
            return updater

        all_dots.add_updater(make_dot_updater())

        def make_volt_updater(curve, ms_w, pts):
            def updater(mob):
                idx = int(np.searchsorted(ms_w, cursor.get_value(), side="right"))
                if idx < 2:
                    mob.set_points_as_corners(pts[:1].repeat(2, axis=0) if len(pts) else pts)
                else:
                    mob.set_points_as_corners(pts[:idx])
            return updater

        for curve, ms_w, pts in volt_updaters:
            curve.add_updater(make_volt_updater(curve, ms_w, pts))

        playhead = always_redraw(lambda: Line(
            [x_of(cursor.get_value()), RASTER_TOP, 0],
            [x_of(cursor.get_value()), VOLT_BOTTOM, 0],
            stroke_color=PLAYHEAD, stroke_width=1.5, stroke_opacity=0.85,
        ))

        def spike_count():
            t = cursor.get_value()
            return sum(1 for dot in all_dots if dot.spike_ms <= t)

        counter = always_redraw(lambda: Text(
            f"t = {cursor.get_value():.0f} ms   ·   spikes shown: {spike_count()}",
            font_size=16, color=PLAYHEAD,
        ).next_to(sub, DOWN, buff=0.1))

        self.add(playhead, counter)

        # ------------------------------------------------------------- play
        rt = SWEEP_TIME / 2 if FAST_SWEEP else SWEEP_TIME
        self.play(cursor.animate.set_value(t_end - 1), run_time=rt, rate_func=linear)

        all_dots.clear_updaters()
        for curve, *_ in volt_updaters:
            curve.clear_updaters()
        self.wait(2.5)
