"""
periodicity_scene.py

Criterion 2 of the CA3-scaling scoring rubric (arXiv:2609.16429): the
periodicity score P(N). Builds it step by step from the CA3_0_0001_3_group_18
recording:

  1. Approximate the local field potential as the average voltage of the
     Pyramidal cells (voltages.csv, group-mean), over the last 2000 ms of the
     5000 ms recording (the paper skips the first 3 s as transient).
  2. |FFT|^2 of that trace -> power spectral density, then smoothed with a
     10-bin *forward* moving average (each bin = mean of itself and the next
     9, i.e. f .. f+4.5 Hz). Because the window looks ahead, peaks smear
     toward lower frequencies, e.g. the ~14.5 Hz beta peak's rising edge
     lands at 11.5 Hz, inside the theta band.
  3. Keep 0 < f < 100 Hz and re-draw on log-log axes, where the 1/f-like
     falloff of LFP power becomes a straight line.
  4. Fit that line (linear regression in log-log space, over 0 < f < 100 Hz)
     as a baseline.
  5. Divide the spectrum by the baseline -> a power-ratio curve that makes
     peaks comparable regardless of their absolute power.
  6. Within each of three oscillation bands (theta 4-12Hz, beta 13-25Hz,
     gamma 25-100Hz, edges exclusive), find the peak ratio and score it on a
     0-1 scale via the paper's piecewise rule (Eq. 3): 0 at or below
     baseline, (ratio-1)/10 up to 11x baseline, capped at 1.0 from there on.
  7. Combine the three band scores with the CA3 weights
     P(N) = 0.3*theta + 0.5*beta + 0.2*gamma.

This mirrors periodicity_score / freq_range_score in the tuning pipeline.

The ratio panel is built on a log frequency axis so the scan line lines up
with the log-log spectrum above it, then slides to a linear axis (as in the
paper's Supp. Fig. 2) before the oscillation bands are scored.

Render:
    manim -pql periodicity_scene.py PeriodicityScore     # quick preview
    manim -pqh periodicity_scene.py PeriodicityScore     # 1080p
"""

import numpy as np
from manim import *

from raster_scene import MUTED, fit, load_voltages, VOLTAGES_CSV

config.background_color = WHITE

# ---------------------------------------------------------------------------
# Analysis parameters (mirrors the paper's Methods / Eq. 3)
# ---------------------------------------------------------------------------
WINDOW_MS = 2000          # analysis window length
FS = 1000.0               # samples/sec (1 ms sampling in voltages.csv)

SMOOTH_BINS = 10         # forward moving-average width (bins of 0.5 Hz)
FREQ_KEEP = (0.0, 100.0)  # exclusive bounds on the frequencies kept for fit/scoring

FREQ_DISPLAY = (0.5, 100.0)
POWER_LOG_RANGE = (1e2, 1e8)
RATIO_LOG_RANGE = (1e-3, 1e3)

RANGE_DEFS = [
    ("Theta", 4.0, 12.0, "#D7A400"),
    ("Beta", 13.0, 25.0, "#C5003E"),
    ("Gamma", 25.0, 100.0, "#7B4FA6"),
]
RANGE_WEIGHT = {"Theta": 0.3, "Beta": 0.5, "Gamma": 0.2}   # CA3 weights
SCORE_CAP_RATIO = 11.0

ACCENT = "#9C6E0C"
FIT_COLOR = "#C5003E"
TRACE_COLOR = "#000000"
RATIO_COLOR = "#1D6FA4"

# ---------------------------------------------------------------------------
# Layout (manim default frame: x in [-7.11, 7.11], y in [-4, 4])
# boxes are (left, bottom, right, top) of the plotting area itself
# ---------------------------------------------------------------------------
TITLE_Y = 3.72
CAPTION_Y = 3.28

FULL_BOX = (-5.6, -2.2, 5.9, 2.7)
TOP_BOX = (-5.6, 0.95, 5.9, 2.75)
BOTTOM_BOX = (-5.6, -2.65, 5.9, -0.75)

MID_X = (FULL_BOX[0] + FULL_BOX[2]) / 2
ARROW_TOP_Y, ARROW_BOTTOM_Y = 0.18, -0.55
FOOTER_ROWS_Y = (-3.08, -3.4, -3.72)

Text.set_default(font="Segoe UI", color=BLACK)

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup(n):
    return str(n).translate(SUPERSCRIPT)


def pow10_label(e):
    return f"10{sup(e)}"


def minus(s):
    return s.replace("-", "−")


def crisp_text(s, font_size=24, **kwargs):
    # rendering at 2x then scaling down gives noticeably sharper glyphs
    return Text(s, font_size=font_size * 2, **kwargs).scale(0.5)


# ---------------------------------------------------------------------------
# Data / signal processing
# ---------------------------------------------------------------------------
def load_pyramidal_window():
    ms, v = load_voltages(VOLTAGES_CSV)["Pyramidal"]
    t_end = int(ms.max()) + 1
    t_start = t_end - WINDOW_MS
    mask = (ms >= t_start) & (ms < t_end)
    return ms[mask], v[mask], t_start, t_end


def compute_fft(v):
    # one-sided spectrum; |rfft|^2 equals |fft|^2 on the positive bins
    freqs = np.fft.rfftfreq(len(v), d=1.0 / FS)
    power = np.abs(np.fft.rfft(v)) ** 2
    return freqs, power


def forward_moving_average(x, w=SMOOTH_BINS):
    # bin i -> mean(x[i : i+w]); the last w bins are left unsmoothed, as in
    # the pipeline's in-place loop
    out = np.array(x, dtype=float)
    n = len(x) - w
    out[:n] = np.convolve(x, np.ones(w) / w, mode="valid")[:n]
    return out


def fit_loglog(freqs, power):
    slope, intercept = np.polyfit(np.log10(freqs), np.log10(power), 1)
    return slope, intercept


def baseline_power(freqs, slope, intercept):
    return (10 ** intercept) * np.asarray(freqs, dtype=float) ** slope


def range_score(ratio_val):
    if ratio_val >= SCORE_CAP_RATIO:
        return 1.0
    if ratio_val > 1.0:
        return (ratio_val - 1.0) / 10
    return 0.0


def linear_ticks(vmax, target=4):
    """0-based ticks with a round step, labelled as mantissa x 10^e."""
    e = int(np.floor(np.log10(vmax)))
    steps = [k * 10.0 ** p for p in (e - 1, e) for k in (1, 2, 2.5, 5)]
    step = next(s for s in steps if vmax / s <= target)
    ticks = []
    for v in np.arange(0, vmax + 1e-9 * vmax, step):
        label = "0" if v == 0 else f"{v / 10 ** e:g}×{pow10_label(e)}"
        ticks.append((v, label))
    return ticks


# ---------------------------------------------------------------------------
# Panel: a framed plot whose data mapping is defined by its box, so data
# can never leave the frame (everything is clipped to [0, 1] of the box).
# ---------------------------------------------------------------------------
class Panel:
    def __init__(self, box, xspec, yspec, xticks, yticks, xlabel, ylabel):
        self.box = box
        self.left, self.bottom, self.right, self.top = box
        self.w = self.right - self.left
        self.h = self.top - self.bottom
        self.xspec, self.yspec = xspec, yspec          # (lo, hi, is_log)
        self.xtick_defs, self.ytick_defs = xticks, yticks
        self.xlabel_text, self.ylabel_text = xlabel, ylabel

    def moved(self, box):
        return Panel(box, self.xspec, self.yspec, self.xtick_defs, self.ytick_defs,
                     self.xlabel_text, self.ylabel_text)

    @staticmethod
    def norm(v, spec):
        lo, hi, is_log = spec
        v = np.asarray(v, dtype=float)
        if is_log:
            u = (np.log10(np.maximum(v, 1e-300)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        else:
            u = (v - lo) / (hi - lo)
        return np.clip(u, 0.0, 1.0)

    def at(self, u, w):
        u = np.asarray(u, dtype=float)
        w = np.asarray(w, dtype=float)
        return np.stack([self.left + u * self.w, self.bottom + w * self.h, np.zeros_like(u + w)], axis=-1)

    def c2p(self, x, y):
        return self.at(self.norm(x, self.xspec), self.norm(y, self.yspec))

    def x_of(self, x):
        return self.left + float(self.norm(x, self.xspec)) * self.w

    def curve(self, xs, ys, color, width=2.2):
        return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(self.c2p(xs, ys))

    # --------------------------------------------------------------- frame
    def x_tick(self, value, label, u):
        x = self.left + u * self.w
        mark = Line([x, self.bottom, 0], [x, self.bottom - 0.08, 0], stroke_color=MUTED, stroke_width=1.2)
        lbl = crisp_text(label, font_size=14, color=MUTED).move_to([x, self.bottom - 0.26, 0])
        tick = VGroup(mark, lbl)
        tick.value = value
        return tick

    def y_tick(self, value, label, u):
        y = self.bottom + u * self.h
        mark = Line([self.left, y, 0], [self.left - 0.08, y, 0], stroke_color=MUTED, stroke_width=1.2)
        lbl = crisp_text(label, font_size=13, color=MUTED).move_to([self.left - 0.14, y, 0], aligned_edge=RIGHT)
        tick = VGroup(mark, lbl)
        tick.value = value
        return tick

    def build(self):
        self.rect = Rectangle(width=self.w, height=self.h, stroke_color=MUTED, stroke_width=1.2)
        self.rect.move_to([(self.left + self.right) / 2, (self.bottom + self.top) / 2, 0])
        self.xticks = VGroup(*[self.x_tick(v, s, float(self.norm(v, self.xspec))) for v, s in self.xtick_defs])
        self.yticks = VGroup(*[self.y_tick(v, s, float(self.norm(v, self.yspec))) for v, s in self.ytick_defs])
        self.xlabel = crisp_text(self.xlabel_text, font_size=15, color=MUTED)
        self.xlabel.move_to([(self.left + self.right) / 2, self.bottom - 0.55, 0])
        self.ylabel = crisp_text(self.ylabel_text, font_size=15, color=MUTED).rotate(PI / 2)
        label_left = min(t[1].get_left()[0] for t in self.yticks) if len(self.yticks) else self.left
        self.ylabel.move_to([label_left - 0.15 - self.ylabel.width / 2, (self.bottom + self.top) / 2, 0])
        self.frame = VGroup(self.rect, self.xticks, self.yticks, self.xlabel, self.ylabel)
        return self.frame

    def decorations(self):
        return VGroup(self.xticks, self.yticks, self.xlabel, self.ylabel)


# ---------------------------------------------------------------------------
class PeriodicityScore(Scene):
    def construct(self):
        self.prepare_data()
        self.intro()
        self.section_trace()
        self.section_fft()
        self.section_loglog()
        self.section_fit()
        self.section_ratio_scan()
        self.section_bands()
        self.finale()

    # ------------------------------------------------------------- data prep
    def prepare_data(self):
        ms, v, t_start, t_end = load_pyramidal_window()
        self.ms, self.v, self.t_start, self.t_end = ms, v, t_start, t_end

        freqs, power = compute_fft(v)
        power_smooth = forward_moving_average(power)
        keep = (freqs > FREQ_KEEP[0]) & (freqs < FREQ_KEEP[1])
        self.freqs = freqs[keep]
        self.power_raw = power[keep]
        self.power = power_smooth[keep]

        self.slope, self.intercept = fit_loglog(self.freqs, self.power)
        self.fit_vals = baseline_power(self.freqs, self.slope, self.intercept)
        self.ratio = self.power / self.fit_vals

        self.range_results = {}
        for name, lo, hi, _ in RANGE_DEFS:
            band = (self.freqs > lo) & (self.freqs < hi)
            idx = np.argmax(self.ratio[band])
            peak_f, peak_r = self.freqs[band][idx], self.ratio[band][idx]
            self.range_results[name] = (peak_f, peak_r, range_score(peak_r))

        self.periodicity_score = sum(RANGE_WEIGHT[n] * self.range_results[n][2] for n, *_ in RANGE_DEFS)

    # ---------------------------------------------------------- panel specs
    def trace_panel(self, box):
        vmin, vmax = self.v.min(), self.v.max()
        pad = 0.08 * (vmax - vmin)
        lo, hi = np.floor((vmin - pad) / 5) * 5, np.ceil((vmax + pad) / 5) * 5
        yticks = [(y, minus(f"{y:.0f}")) for y in np.arange(np.ceil(lo / 10) * 10, hi + 1e-9, 10)]
        xticks = [(t, str(t)) for t in range(self.t_start, self.t_end + 1, 500)]
        return Panel(box, (self.t_start, self.t_end, False), (lo, hi, False), xticks, yticks,
                     "Time (ms)", "mV")

    def linear_psd_panel(self, box, smoothed=False):
        top = (self.power if smoothed else self.power_raw).max() * 1.08
        xticks = [(f, str(f)) for f in range(0, 101, 20)]
        return Panel(box, (0, 100, False), (0, top, False), xticks, linear_ticks(top),
                     "Frequency (Hz)", "Power")

    def loglog_psd_panel(self, box):
        xticks = [(f, str(f)) for f in (1, 10, 100)]
        yticks = [(10.0 ** e, pow10_label(e)) for e in range(2, 9, 2)]
        return Panel(box, (*FREQ_DISPLAY, True), (*POWER_LOG_RANGE, True), xticks, yticks,
                     "Frequency (Hz), log scale", "Power, log scale")

    def ratio_panel(self, box, log_x=True):
        yticks = [(10.0 ** e, pow10_label(e)) for e in range(-3, 4)]
        if log_x:
            xticks = [(f, str(f)) for f in (1, 10, 100)]
            return Panel(box, (*FREQ_DISPLAY, True), (*RATIO_LOG_RANGE, True), xticks, yticks,
                         "Frequency (Hz), log scale", "Power ÷ baseline")
        xticks = [(f, str(f)) for f in range(0, 101, 20)]
        return Panel(box, (0, FREQ_DISPLAY[1], False), (*RATIO_LOG_RANGE, True), xticks, yticks,
                     "Frequency (Hz)", "Power ÷ baseline")

    # -------------------------------------------------------------- chrome
    def intro(self):
        self.title = crisp_text("Periodicity score", font_size=30, weight=BOLD).move_to(UP * TITLE_Y)
        sub = crisp_text("CA3_0_0001_3_group_18  ·  average voltage of Pyramidal cells (approx. LFP)",
                   font_size=16, color=MUTED).next_to(self.title, DOWN, buff=0.12)
        self.play(FadeIn(self.title, shift=DOWN * 0.15), FadeIn(sub))
        self.wait(0.6)
        self.play(FadeOut(sub))

    def set_caption(self, text_str):
        new_caption = fit(crisp_text(text_str, font_size=20, color=ACCENT), 11.6).move_to(UP * CAPTION_Y)
        if hasattr(self, "caption"):
            self.play(ReplacementTransform(self.caption, new_caption), run_time=0.4)
        else:
            self.play(FadeIn(new_caption), run_time=0.4)
        self.caption = new_caption

    def show_panel(self, panel, curve, run_time=1.4):
        self.play(Create(panel.rect), FadeIn(panel.decorations()))
        self.play(Create(curve), run_time=run_time)

    def down_arrow(self, label_str):
        arrow = Arrow([MID_X, ARROW_TOP_Y, 0], [MID_X, ARROW_BOTTOM_Y, 0], buff=0, color=ACCENT,
                      stroke_width=4, max_tip_length_to_length_ratio=0.3)
        label = crisp_text(label_str, font_size=20, color=ACCENT).next_to(arrow, RIGHT, buff=0.2)
        return arrow, label

    # --------------------------------------------------------- 1. the trace
    def section_trace(self):
        self.set_caption(f"Last {WINDOW_MS} ms of the recording: mean voltage of the Pyramidal group")
        panel = self.trace_panel(FULL_BOX)
        panel.build()
        curve = panel.curve(self.ms, self.v, TRACE_COLOR)
        self.show_panel(panel, curve, run_time=2.0)
        self.wait(0.6)
        self.trace, self.trace_curve = panel, curve

    # ------------------------------------------- 2-3. |FFT|^2 -> linear PSD
    def section_fft(self):
        top = self.trace_panel(TOP_BOX)
        top.build()
        top_curve = top.curve(self.ms, self.v, TRACE_COLOR)
        self.play(ReplacementTransform(self.trace.frame, top.frame),
                  ReplacementTransform(self.trace_curve, top_curve), run_time=1.2)

        self.set_caption("Fourier transform: power spectral density = |FFT|²")
        arrow, arrow_lbl = self.down_arrow("|FFT|²")
        self.play(GrowArrow(arrow), FadeIn(arrow_lbl))

        psd = self.linear_psd_panel(BOTTOM_BOX)
        psd.build()
        psd_curve = psd.curve(self.freqs, self.power_raw, TRACE_COLOR, 2.0)
        self.show_panel(psd, psd_curve)
        self.wait(0.6)

        self.play(FadeOut(top.frame), FadeOut(top_curve), FadeOut(arrow), FadeOut(arrow_lbl))
        full = self.linear_psd_panel(FULL_BOX)
        full.build()
        full_curve = full.curve(self.freqs, self.power_raw, TRACE_COLOR, 2.0)
        self.play(ReplacementTransform(psd.frame, full.frame),
                  ReplacementTransform(psd_curve, full_curve), run_time=1.2)
        self.wait(0.4)

        self.set_caption(f"Smooth: each bin becomes the mean of itself and the next {SMOOTH_BINS - 1} "
                         f"(f to f + {(SMOOTH_BINS - 1) * FS / WINDOW_MS:g} Hz)")
        raw_ghost = full_curve.copy().set_stroke(color=GREY_B, opacity=0.8)
        self.add(raw_ghost, full_curve)
        smoothed = full.curve(self.freqs, self.power, TRACE_COLOR, 2.0)
        self.play(Transform(full_curve, smoothed), run_time=1.6)
        self.set_caption("The look-ahead window lowers and widens peaks, dragging them toward lower frequencies")
        self.wait(1.0)
        self.play(FadeOut(raw_ghost))

        self.set_caption("On linear axes the beta peak dwarfs everything else: the rest is flattened")
        rescaled = self.linear_psd_panel(FULL_BOX, smoothed=True)
        rescaled.build()
        self.morph_axes(full, rescaled, full_curve, self.freqs, self.power, run_time=1.5)
        self.wait(0.8)
        self.psd, self.psd_curve = rescaled, full_curve

    # -------------------------------------------------- 4. linear -> log-log
    def morph_axes(self, src, dst, curve, xs, ys, run_time=3.0):
        """Slide `curve` and the tick marks from src's axis scaling to dst's
        (same box), cross-fading tick sets and axis labels that change. On
        return, dst.frame holds the live mobjects."""
        t = ValueTracker(0.0)

        def blend(v, spec_a, spec_b):
            a = t.get_value()
            return (1 - a) * Panel.norm(v, spec_a) + a * Panel.norm(v, spec_b)

        def bx(v):
            return blend(v, src.xspec, dst.xspec)

        def by(v):
            return blend(v, src.yspec, dst.yspec)

        curve.add_updater(lambda m: m.set_points_as_corners(dst.at(bx(xs), by(ys))))

        def tick_updater(axis, fading_out):
            def update(tick):
                if axis == "x":
                    x = dst.left + float(bx(tick.value)) * dst.w
                    tick.shift(RIGHT * (x - tick[0].get_x()))
                else:
                    y = dst.bottom + float(by(tick.value)) * dst.h
                    tick.shift(UP * (y - tick[0].get_y()))
                tick.set_opacity(1 - t.get_value() if fading_out else t.get_value())
            return update

        x_same = src.xspec == dst.xspec and src.xtick_defs == dst.xtick_defs
        y_same = src.yspec == dst.yspec and src.ytick_defs == dst.ytick_defs
        if x_same:
            dst.xticks = src.xticks
        if y_same:
            dst.yticks = src.yticks

        moving = []
        for axis, same, old, new in (("x", x_same, src.xticks, dst.xticks), ("y", y_same, src.yticks, dst.yticks)):
            if same:
                continue
            for tick in old:
                tick.add_updater(tick_updater(axis, True))
            for tick in new:
                tick.add_updater(tick_updater(axis, False))
                tick.update()
            self.add(new)
            moving += [*old, *new]

        label_anims = []
        for attr in ("xlabel", "ylabel"):
            old_lbl, new_lbl = getattr(src, attr), getattr(dst, attr)
            if getattr(src, attr + "_text") == getattr(dst, attr + "_text"):
                setattr(dst, attr, old_lbl)
            else:
                label_anims += [FadeOut(old_lbl), FadeIn(new_lbl)]

        self.play(t.animate.set_value(1.0), *label_anims, run_time=run_time, rate_func=smooth)

        curve.clear_updaters()
        for tick in moving:
            tick.clear_updaters()
        if not x_same:
            self.remove(src.xticks)
            dst.xticks.set_opacity(1)
        if not y_same:
            self.remove(src.yticks)
            dst.yticks.set_opacity(1)
        self.remove(src.rect)
        self.add(dst.rect)
        dst.frame = VGroup(dst.rect, dst.xticks, dst.yticks, dst.xlabel, dst.ylabel)

    def section_loglog(self):
        self.set_caption("Re-scale both axes to log: the 1/f falloff becomes a straight line")
        log = self.loglog_psd_panel(FULL_BOX)
        log.build()
        self.morph_axes(self.psd, log, self.psd_curve, self.freqs, self.power)
        self.wait(0.6)
        self.psd = log

    # ----------------------------------------------------------- 5. best fit
    def fit_overlay(self, panel):
        line = DashedVMobject(panel.curve(self.freqs, self.fit_vals, FIT_COLOR, 2.5), num_dashes=45)
        eq = VGroup(
            crisp_text("baseline: least-squares line in log-log", font_size=13, color=FIT_COLOR),
            crisp_text(minus(f"log P = {self.slope:.2f} · log f + {self.intercept:.2f}"), font_size=17, color=FIT_COLOR),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.08)
        eq.next_to(panel.rect.get_corner(UR), DL, buff=0.15)
        return line, eq

    def section_fit(self):
        self.set_caption("Linear regression in log-log space = baseline power at every frequency")
        self.fit_line, self.fit_eq = self.fit_overlay(self.psd)
        self.play(Create(self.fit_line), run_time=1.2)
        self.play(FadeIn(self.fit_eq))
        self.wait(0.8)

    # -------------------------------------------------- 6. build the ratio
    def section_ratio_scan(self):
        top = self.loglog_psd_panel(TOP_BOX)
        top.build()
        top_curve = top.curve(self.freqs, self.power, TRACE_COLOR)
        top_fit, top_eq = self.fit_overlay(top)
        self.play(ReplacementTransform(self.psd.frame, top.frame),
                  ReplacementTransform(self.psd_curve, top_curve),
                  ReplacementTransform(self.fit_line, top_fit),
                  ReplacementTransform(self.fit_eq, top_eq), run_time=1.2)

        self.set_caption("Divide the spectrum by its baseline, one frequency at a time")
        arrow, arrow_lbl = self.down_arrow("power ÷ baseline")
        self.play(GrowArrow(arrow), FadeIn(arrow_lbl))

        bottom = self.ratio_panel(BOTTOM_BOX)
        bottom.build()
        self.play(Create(bottom.rect), FadeIn(bottom.decorations()))

        log_f = ValueTracker(np.log10(FREQ_DISPLAY[0]))
        log_freqs = np.log10(self.freqs)

        def f_now():
            return 10 ** log_f.get_value()

        def interp(values):
            return 10 ** np.interp(log_f.get_value(), log_freqs, np.log10(values))

        ratio_curve = VMobject(stroke_color=RATIO_COLOR, stroke_width=2.2)

        def grow_ratio(m):
            n = max(2, int(np.searchsorted(self.freqs, f_now() + 1e-9, side="right")))
            m.set_points_as_corners(bottom.c2p(self.freqs[:n], self.ratio[:n]))

        ratio_curve.add_updater(grow_ratio)
        ratio_curve.update()

        playhead = always_redraw(lambda: DashedLine(
            [top.x_of(f_now()), top.top, 0], [top.x_of(f_now()), bottom.bottom, 0],
            stroke_color=ACCENT, stroke_width=1.5, dash_length=0.06))
        power_dot = always_redraw(lambda: Dot(top.c2p(f_now(), interp(self.power)), radius=0.06, color=TRACE_COLOR))
        fit_dot = always_redraw(lambda: Dot(top.c2p(f_now(), interp(self.fit_vals)), radius=0.06, color=FIT_COLOR))
        ratio_dot = always_redraw(lambda: Dot(bottom.c2p(f_now(), interp(self.ratio)), radius=0.06, color=RATIO_COLOR))

        def readout_text():
            p, b = interp(self.power), interp(self.fit_vals)
            return crisp_text(f"f = {f_now():.1f} Hz:   {p:.1e} ÷ {b:.1e} = {p / b:.2f}×",
                        font_size=16, color=ACCENT).move_to([MID_X - 0.35, (ARROW_TOP_Y + ARROW_BOTTOM_Y) / 2, 0],
                                                            aligned_edge=RIGHT)

        readout = always_redraw(readout_text)

        self.add(ratio_curve, playhead, power_dot, fit_dot, ratio_dot, readout)
        self.play(log_f.animate.set_value(np.log10(FREQ_DISPLAY[1])), run_time=7.0, rate_func=linear)
        ratio_curve.clear_updaters()
        self.play(FadeOut(playhead), FadeOut(power_dot), FadeOut(fit_dot), FadeOut(ratio_dot), FadeOut(readout))
        self.wait(0.4)

        self.play(FadeOut(top.frame), FadeOut(top_curve), FadeOut(top_fit), FadeOut(top_eq),
                  FadeOut(arrow), FadeOut(arrow_lbl))
        full = self.ratio_panel(FULL_BOX)
        full.build()
        full_curve = full.curve(self.freqs, self.ratio, RATIO_COLOR)
        self.play(ReplacementTransform(bottom.frame, full.frame),
                  ReplacementTransform(ratio_curve, full_curve), run_time=1.2)

        self.set_caption("Switch the frequency axis back to linear to read off the oscillation bands")
        linear_view = self.ratio_panel(FULL_BOX, log_x=False)
        linear_view.build()
        self.morph_axes(full, linear_view, full_curve, self.freqs, self.ratio, run_time=2.5)
        self.wait(0.4)
        self.ratio_view, self.ratio_curve = linear_view, full_curve

    # ------------------------------------------------------------- 7. bands
    def section_bands(self):
        self.set_caption("Score the peak ratio in each oscillation band: theta, beta, gamma")
        panel = self.ratio_view
        lo_r, hi_r = RATIO_LOG_RANGE

        bands = VGroup()
        for _, lo, hi, color in RANGE_DEFS:
            p0, p1 = panel.c2p(lo, lo_r), panel.c2p(hi, hi_r)
            bands.add(Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1], stroke_width=0,
                                fill_color=color, fill_opacity=0.14).move_to((p0 + p1) / 2))
        bands.set_z_index(-1)

        legend = VGroup(*[
            VGroup(Square(0.14, stroke_width=0, fill_color=color, fill_opacity=0.5),
                   crisp_text(f"{name} ({lo:g}-{hi:g} Hz)", font_size=14, color=MUTED)).arrange(RIGHT, buff=0.08)
            for name, lo, hi, color in RANGE_DEFS
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        legend.next_to(panel.rect.get_corner(UR), DL, buff=0.15)
        self.play(FadeIn(bands), FadeIn(legend))

        ref_lines = VGroup()
        for value, ratio_str, score_str, color in ((1.0, "ratio = 1", "score 0", MUTED),
                                                   (SCORE_CAP_RATIO, "ratio = 11", "score 1.0", BLACK)):
            y = panel.c2p(FREQ_DISPLAY[1], value)[1]
            line = DashedLine([panel.left, y, 0], [panel.right, y, 0], stroke_color=color,
                              stroke_width=1.5, dash_length=0.08)
            lbl = VGroup(crisp_text(ratio_str, font_size=13, color=color),
                         crisp_text(score_str, font_size=12, color=color)).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
            lbl.next_to([panel.right, y, 0], RIGHT, buff=0.1)
            ref_lines.add(VGroup(line, lbl))
        self.play(LaggedStart(*[Create(g[0]) for g in ref_lines], lag_ratio=0.3),
                  FadeIn(VGroup(*[g[1] for g in ref_lines])))

        rule = crisp_text("score = 0 if ratio ≤ 1  ·  (ratio − 1) / 10 if ratio < 11  ·  1.0 if ratio ≥ 11",
                    font_size=15, color=MUTED).move_to([MID_X, FOOTER_ROWS_Y[0], 0])
        fit(rule, panel.w)
        self.play(FadeIn(rule))
        self.wait(0.8)
        self.play(FadeOut(rule))

        self.peak_dots = VGroup()
        self.score_rows = VGroup()
        for i, (name, lo, hi, color) in enumerate(RANGE_DEFS):
            peak_f, peak_r, score = self.range_results[name]
            dot = Dot(panel.c2p(peak_f, peak_r), radius=0.08, color=color)
            self.play(bands[i].animate.set_fill(opacity=0.34), FadeIn(dot, scale=0.5), run_time=0.6)

            if peak_r >= SCORE_CAP_RATIO:
                verdict = f"≥ 11  →  score {score:.2f}"
            elif peak_r > 1:
                verdict = f"→  score ({peak_r:.2f} − 1) / 10 = {score:.2f}"
            else:
                verdict = f"≤ 1  →  score {score:.2f}"
            row = crisp_text(f"{name} ({lo:g}-{hi:g} Hz):  peak at {peak_f:.1f} Hz  ·  ratio {peak_r:.1f}×  {verdict}",
                       font_size=16, color=color)
            fit(row, panel.w).move_to([MID_X, FOOTER_ROWS_Y[i], 0])
            self.play(FadeIn(row), bands[i].animate.set_fill(opacity=0.14), run_time=0.6)
            self.peak_dots.add(dot)
            self.score_rows.add(row)
            self.wait(0.3)
        self.wait(0.6)

    # ------------------------------------------------------------- finale
    def finale(self):
        self.play(FadeOut(self.score_rows))
        self.set_caption("Combine the three band scores into the periodicity score P(N)")

        sum_parts = " + ".join(
            f"{RANGE_WEIGHT[name]:g}×{self.range_results[name][2]:.2f}" for name, *_ in RANGE_DEFS)
        summary = VGroup(
            crisp_text(f"P(N) = {sum_parts} = {self.periodicity_score:.3f}", font_size=22, weight=BOLD, color=ACCENT),
            crisp_text("CA3 weights: θ 0.3 · β 0.5 · γ 0.2  ·  P(N) is weighted 0.4 in the overall objective O(N)",
                 font_size=14, color=MUTED),
        ).arrange(DOWN, buff=0.15)
        fit(summary, self.ratio_view.w)
        summary.move_to([MID_X, (FOOTER_ROWS_Y[0] + FOOTER_ROWS_Y[2]) / 2, 0])

        self.play(FadeIn(summary),
                  *[Flash(d, color=c, flash_radius=0.25) for d, (_, _, _, c) in zip(self.peak_dots, RANGE_DEFS)])
        self.wait(3)
