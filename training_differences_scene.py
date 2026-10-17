"""
training_differences_scene.py

Extends the "Differences in Training" slides with the story of *why*
spiking networks are hard to train with vanilla backprop, told as three
clips that build on each other:

  * Clip1 - Forward pass. A traditional (sigmoid) and a spiking network,
            same input, same weights, propagate layer by layer. Each
            architecture's activation function is drawn once, below its
            network, and every neuron's pre-activation lands on it as the
            pass proceeds -- the traditional net's values are spread out,
            the spiking net's are mostly silent (0) with the occasional
            spike (1).
  * Clip2 - Weight update. The per-layer chain rule appears for each
            architecture, and each activation panel grows its derivative
            curve, with a marker at the specific value that layer produced
            in Clip1 -- the actual gradient this example contributes.
  * Clip3 - The derivative's area. The networks fade away, the two
            derivative curves take over, and a single arrow at each
            curve's peak looks almost comparable (the spike's even
            taller) -- until the area under each curve is shaded: the
            sigmoid's derivative is usable across a wide band, the
            spike's only in a sliver around threshold. Ends on the same
            problem statement as before -- surrogate gradients are the
            next slide.

Simplification: the spiking side is treated as an instantaneous step
S(v) = H(v - theta) applied once per layer (no leaky/temporal membrane
dynamics), so both architectures share the exact same feed-forward
structure and only their nonlinearity differs.

Render:
    manim -pql training_differences_scene.py Clip1 Clip2 Clip3
    manim -pqh training_differences_scene.py Clip1 Clip2 Clip3
"""

from manim import *
import numpy as np

config.background_color = WHITE
Text.set_default(font="Segoe UI", color=BLACK)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
EDGE = "#3a4556"          # network connections
TRAD_COLOR = "#B8860B"    # traditional / sigmoid
GREEN = "#2E7D32"         # spiking activation S(v)
DELTA_COLOR = "#C5003E"   # spiking derivative / the delta
ACCENT = "#9C6E0C"        # edge-flash highlight
MARKER_COLOR = "#1D6FA4"  # forward-pass markers (distinct from both curve colors)

THETA = 0.5               # spiking threshold

# ---------------------------------------------------------------------------
# Network simulation (shared by all three clips)
# ---------------------------------------------------------------------------
LAYERS = [3, 3, 3, 3]
SEED = 9
W_SCALE = 0.6
X0 = np.array([1.0, 0.4, -0.6])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def heaviside(x, theta=THETA):
    return (x > theta).astype(float) if isinstance(x, np.ndarray) else float(x > theta)


def dirac_bump(x, theta=THETA, width=0.05, height=1.0):
    """A narrow bump standing in for delta(x - theta): tall and thin, not a
    true (infinite) delta, so it can actually be drawn and its finite area
    compared against the sigmoid derivative's."""
    return height * np.exp(-0.5 * ((x - theta) / width) ** 2)


def simulate():
    """Same weights and input, two nonlinearities. Returns, per layer
    transition l (0..2): the pre-activations z and the resulting values a,
    for both architectures."""
    rng = np.random.default_rng(SEED)
    Ws = [rng.normal(0, W_SCALE, (b, a)) for a, b in zip(LAYERS[:-1], LAYERS[1:])]

    trad_layers, spike_layers = [X0], [X0]
    trad_z, spike_z = [], []
    a, v = X0, X0
    for W in Ws:
        z_a = W @ a
        a = sigmoid(z_a)
        trad_z.append(z_a)
        trad_layers.append(a)

        z_v = W @ v
        v = heaviside(z_v)
        spike_z.append(z_v)
        spike_layers.append(v)

    return Ws, trad_layers, trad_z, spike_layers, spike_z


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
TITLE_Y = 3.7
SUBTITLE_Y = 3.32

HEADER_Y = 2.65
NET_CENTER_Y = 2.0
ROW_GAP = 0.4
NEURON_R = 0.17

LEFT_COLS = np.linspace(-6.6, -0.9, 4)
RIGHT_COLS = np.linspace(0.9, 6.6, 4)
ROWS_Y = [NET_CENTER_Y + ROW_GAP, NET_CENTER_Y, NET_CENTER_Y - ROW_GAP]

ACT_AXES_Y = -0.4
ACT_X_LENGTH = 5.7
ACT_Y_LENGTH = 1.9
LEFT_PANEL_X = float(np.mean(LEFT_COLS))
RIGHT_PANEL_X = float(np.mean(RIGHT_COLS))

EQN_Y = -1.85
CAPTION_Y = -2.9


def crisp_text(s, font_size=24, **kwargs):
    return Text(s, font_size=font_size * 2, **kwargs).scale(0.5)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------
def build_network(cols_x, prefix, input_values, weight_matrices):
    """Returns circles[layer][row], labels[layer][row] (None until filled),
    edges[layer] (VGroup of Line per (row_from, row_to) pair), and headers."""
    circles = [[Circle(radius=NEURON_R, stroke_color=BLACK, stroke_width=1.5,
                        fill_color=BLACK, fill_opacity=0)
                for _ in range(3)] for _ in range(4)]
    for row in range(3):
        circles[0][row].set_fill(opacity=1)
        circles[0][row].move_to([cols_x[0], ROWS_Y[row], 0])
    for layer in range(1, 4):
        for row in range(3):
            circles[layer][row].move_to([cols_x[layer], ROWS_Y[row], 0])

    labels = [[None, None, None] for _ in range(4)]
    for row in range(3):
        lbl = crisp_text(f"{input_values[row]:.1f}", font_size=14, color=WHITE)
        lbl.move_to(circles[0][row].get_center())
        labels[0][row] = lbl

    edges = []
    for layer in range(3):
        W = weight_matrices[layer]
        max_w = np.abs(W).max()
        lines = VGroup()
        for i in range(3):       # from-row (layer)
            for j in range(3):   # to-row (layer + 1)
                opacity = float(np.clip(abs(W[j, i]) / max_w, 0.25, 1.0))
                line = Line(
                    circles[layer][i].get_center(), circles[layer + 1][j].get_center(),
                    color=EDGE, stroke_width=1.2, stroke_opacity=opacity,
                )
                lines.add(line)
        edges.append(lines)

    headers = VGroup()
    for layer in range(4):
        h = crisp_text(f"{prefix}{layer}", font_size=20, color=BLACK)
        h.move_to([cols_x[layer], HEADER_Y, 0])
        headers.add(h)

    return circles, labels, edges, headers


def build_activation_panel(center_x, func, color, discontinuities=None):
    axes = Axes(
        x_range=[-2.5, 2.5, 1], y_range=[0, 1.0, 0.5],
        x_length=ACT_X_LENGTH, y_length=ACT_Y_LENGTH,
        axis_config={"color": "#444444", "stroke_width": 1.5,
                     "include_tip": False, "font_size": 14},
    ).move_to([center_x, ACT_AXES_Y, 0])
    kwargs = {"color": color, "stroke_width": 3}
    if discontinuities is not None:
        kwargs["discontinuities"] = discontinuities
    curve = axes.plot(func, x_range=[-2.5, 2.5], **kwargs)
    return axes, curve


def make_marker(axes, z, val, color):
    line = Line(axes.c2p(z, 0), axes.c2p(z, val), color=color, stroke_width=2)
    dot = Dot(axes.c2p(z, val), radius=0.05, color=color)
    return VGroup(line, dot)


def build_filled_network(cols_x, prefix, layer_values, weight_matrices, fmt):
    """Like build_network, but every layer already shows its computed value
    (used when a clip picks up right where the forward pass left off)."""
    circles, labels, edges, headers = build_network(cols_x, prefix, layer_values[0], weight_matrices)
    group = VGroup(*edges)
    for layer in range(4):
        for row in range(3):
            c = circles[layer][row]
            if layer > 0:
                c.set_fill(opacity=1)
                lbl = crisp_text(fmt(layer_values[layer][row]), font_size=14, color=WHITE)
                lbl.move_to(c.get_center())
                labels[layer][row] = lbl
            group.add(c, labels[layer][row])
    group.add(headers)
    return circles, labels, edges, headers, group


class Clip1(Scene):
    def construct(self):
        Ws, trad_layers, trad_z, spike_layers, spike_z = simulate()

        title = crisp_text("Differences in Training", font_size=38, weight=BOLD)
        title.move_to([0, TITLE_Y, 0])
        subtitle = crisp_text("Same input, same weights — different activation function",
                               font_size=22, color="#555555")
        subtitle.move_to([0, SUBTITLE_Y, 0])
        self.play(FadeIn(title, shift=DOWN * 0.1), FadeIn(subtitle))

        trad_c, trad_l, trad_e, trad_h = build_network(LEFT_COLS, "L", X0, Ws)
        spike_c, spike_l, spike_e, spike_h = build_network(RIGHT_COLS, "V", X0, Ws)

        trad_axes, trad_curve = build_activation_panel(LEFT_PANEL_X, sigmoid, TRAD_COLOR)
        trad_func_label = MathTex(r"\sigma(x)", font_size=28, color=TRAD_COLOR)
        trad_func_label.next_to(trad_axes, UP, buff=0.2)

        spike_axes, spike_curve = build_activation_panel(
            RIGHT_PANEL_X, heaviside, GREEN, discontinuities=[THETA])
        spike_func_label = MathTex(r"S(v) = H(v-\theta)", font_size=28, color=GREEN)
        spike_func_label.next_to(spike_axes, UP, buff=0.2)

        all_circles = VGroup(*[c for layer in trad_c + spike_c for c in layer])
        all_edges = VGroup(*trad_e, *spike_e)
        all_headers = VGroup(trad_h, spike_h)
        input_labels = VGroup(*trad_l[0], *spike_l[0])

        self.play(
            Create(all_edges), Create(all_circles), Write(all_headers),
            run_time=1.4,
        )
        self.play(
            *[FadeIn(lbl) for lbl in input_labels],
            Create(trad_axes), Create(spike_axes),
            Write(trad_func_label), Write(spike_func_label),
            run_time=1.0,
        )
        self.play(Create(trad_curve), Create(spike_curve))
        self.wait(0.4)

        for layer in range(3):
            self.play(
                ShowPassingFlash(trad_e[layer].copy().set_stroke(color=ACCENT, width=3), time_width=0.6),
                ShowPassingFlash(spike_e[layer].copy().set_stroke(color=ACCENT, width=3), time_width=0.6),
                run_time=1.0,
            )

            markers = []
            new_labels = []
            for row in range(3):
                z_t, a_t = trad_z[layer][row], trad_layers[layer + 1][row]
                m_t = make_marker(trad_axes, z_t, a_t, MARKER_COLOR)
                markers.append(m_t)
                lbl_t = crisp_text(f"{a_t:.2f}", font_size=14, color=WHITE)
                lbl_t.move_to(trad_c[layer + 1][row].get_center())
                trad_l[layer + 1][row] = lbl_t
                new_labels.append(lbl_t)

                z_s, a_s = spike_z[layer][row], spike_layers[layer + 1][row]
                m_s = make_marker(spike_axes, z_s, a_s, MARKER_COLOR)
                markers.append(m_s)
                lbl_s = crisp_text(f"{a_s:.0f}", font_size=14, color=WHITE)
                lbl_s.move_to(spike_c[layer + 1][row].get_center())
                spike_l[layer + 1][row] = lbl_s
                new_labels.append(lbl_s)

            self.play(*[FadeIn(m) for m in markers], run_time=0.6)
            self.play(
                *[c.animate.set_fill(opacity=1) for c in trad_c[layer + 1] + spike_c[layer + 1]],
                *[Write(lbl) for lbl in new_labels],
                run_time=0.6,
            )
            self.wait(0.2)
            self.play(*[m.animate.set_opacity(0.35) for m in markers], run_time=0.3)

        caption = crisp_text(
            "The traditional net's values spread out; the spiking net is mostly silent.",
            font_size=22, color="#555555",
        )
        caption.move_to([0, CAPTION_Y, 0])
        self.play(FadeIn(caption))
        self.wait(2)


class Clip2(Scene):
    """Picks up where Clip1 left off: the forward pass is done and cached,
    now we update weights. Same chain rule template for both architectures;
    the difference is entirely in the derivative term."""

    SPOTLIGHT = (1, 1)  # (layer, row) of the neuron we zoom in on

    def construct(self):
        Ws, trad_layers, trad_z, spike_layers, spike_z = simulate()

        title = crisp_text("Differences in Training", font_size=38, weight=BOLD)
        title.move_to([0, TITLE_Y, 0])
        subtitle = crisp_text("Same chain rule — different derivative",
                               font_size=22, color="#555555")
        subtitle.move_to([0, SUBTITLE_Y, 0])

        trad_c, trad_l, trad_e, trad_h, trad_group = build_filled_network(
            LEFT_COLS, "L", trad_layers, Ws, lambda v: f"{v:.2f}")
        spike_c, spike_l, spike_e, spike_h, spike_group = build_filled_network(
            RIGHT_COLS, "V", spike_layers, Ws, lambda v: f"{v:.0f}")

        trad_axes, trad_curve = build_activation_panel(LEFT_PANEL_X, sigmoid, TRAD_COLOR)
        trad_func_label = MathTex(r"\sigma(x)", font_size=28, color=TRAD_COLOR)
        trad_func_label.next_to(trad_axes, UP, buff=0.2)

        spike_axes, spike_curve = build_activation_panel(
            RIGHT_PANEL_X, heaviside, GREEN, discontinuities=[THETA])
        spike_func_label = MathTex(r"S(v) = H(v-\theta)", font_size=28, color=GREEN)
        spike_func_label.next_to(spike_axes, UP, buff=0.2)

        self.play(FadeIn(title, shift=DOWN * 0.1), FadeIn(subtitle))
        self.play(
            FadeIn(trad_group), FadeIn(spike_group),
            Create(trad_axes), Create(spike_axes),
            Write(trad_func_label), Write(spike_func_label),
            run_time=1.0,
        )
        self.play(Create(trad_curve), Create(spike_curve))
        self.wait(0.3)

        trad_eqn = MathTex(
            r"\frac{dW_i}{dLoss} \;=\; \frac{dL_{i+1}}{dLoss} \cdot"
            r" \frac{d\sigma(L_i \cdot W_i)}{d(L_i \cdot W_i)}",
            font_size=24, color=BLACK,
        ).move_to([LEFT_PANEL_X, EQN_Y, 0])
        spike_eqn = MathTex(
            r"\frac{dW_i}{dLoss} \;=\; \frac{dV_{i+1}}{dLoss} \cdot"
            r" \frac{dS(V_i \cdot W_i)}{d(V_i \cdot W_i)}",
            font_size=24, color=BLACK,
        ).move_to([RIGHT_PANEL_X, EQN_Y, 0])
        self.play(Write(trad_eqn), Write(spike_eqn))
        self.wait(0.4)

        trad_deriv = trad_axes.plot(sigmoid_prime, x_range=[-2.5, 2.5],
                                     color=DELTA_COLOR, stroke_width=3)
        spike_deriv = spike_axes.plot(dirac_bump, x_range=[-2.5, 2.5, 0.002],
                                       color=DELTA_COLOR, stroke_width=3)
        trad_deriv_label = MathTex(r"\sigma'(x)", font_size=22, color=DELTA_COLOR)
        trad_deriv_label.next_to(trad_axes.c2p(2.5, 0), UR, buff=0.05)
        spike_deriv_label = MathTex(r"\delta(v-\theta)", font_size=22, color=DELTA_COLOR)
        spike_deriv_label.next_to(spike_axes.c2p(-2.5, 0), UR, buff=0.05)

        self.play(
            Create(trad_deriv), Create(spike_deriv),
            Write(trad_deriv_label), Write(spike_deriv_label),
        )
        self.wait(0.3)

        # Spotlight one specific neuron's actual gradient: even the one that
        # fired gets essentially nothing, because it wasn't exactly at
        # threshold.
        layer, row = self.SPOTLIGHT
        z_t, d_t = trad_z[layer][row], sigmoid_prime(trad_z[layer][row])
        z_s, d_s = spike_z[layer][row], dirac_bump(spike_z[layer][row])

        guide_t = DashedLine(trad_axes.c2p(z_t, 0), trad_axes.c2p(z_t, 1.05),
                              color=MARKER_COLOR, stroke_width=1.5, dash_length=0.08)
        dot_t = Dot(trad_axes.c2p(z_t, d_t), radius=0.06, color=MARKER_COLOR)
        label_t = crisp_text(f"{d_t:.2f}", font_size=16, color=MARKER_COLOR)
        label_t.next_to(dot_t, UR, buff=0.05)

        guide_s = DashedLine(spike_axes.c2p(z_s, 0), spike_axes.c2p(z_s, 1.05),
                              color=MARKER_COLOR, stroke_width=1.5, dash_length=0.08)
        dot_s = Dot(spike_axes.c2p(z_s, 0), radius=0.06, color=MARKER_COLOR)
        label_s = crisp_text("≈ 0", font_size=16, color=MARKER_COLOR)
        label_s.next_to(dot_s, UR, buff=0.05)

        self.play(Create(guide_t), Create(guide_s))
        self.play(FadeIn(dot_t), FadeIn(dot_s), Write(label_t), Write(label_s))
        self.wait(0.3)

        caption = crisp_text(
            "Even the neuron that fired gets almost no gradient — "
            "only inputs exactly at threshold do.",
            font_size=22, color="#555555",
        )
        caption.move_to([0, CAPTION_Y, 0])
        self.play(FadeIn(caption))
        self.wait(2)


class Clip3(Scene):
    """The network is gone; only the two derivative curves matter now. A
    single arrow at each curve's peak looks almost comparable -- the spike
    is even taller. Only once we shade the area under each curve does the
    real gap show up: the sigmoid's derivative is usable across a wide
    band, the spike's only in a sliver around threshold."""

    BIG_Y_LENGTH = 3.4
    PANEL_Y = 0.3

    def construct(self):
        title = crisp_text("Differences in Training", font_size=38, weight=BOLD)
        title.move_to([0, TITLE_Y, 0])
        subtitle = crisp_text("How much of that gradient is actually usable?",
                               font_size=22, color="#555555")
        subtitle.move_to([0, SUBTITLE_Y, 0])
        self.play(FadeIn(title, shift=DOWN * 0.1), FadeIn(subtitle))

        trad_axes = Axes(
            x_range=[-2.5, 2.5, 1], y_range=[0, 1.0, 0.5],
            x_length=ACT_X_LENGTH, y_length=self.BIG_Y_LENGTH,
            axis_config={"color": "#444444", "stroke_width": 1.5,
                         "include_tip": False, "font_size": 14},
        ).move_to([LEFT_PANEL_X, self.PANEL_Y, 0])
        trad_curve = trad_axes.plot(sigmoid_prime, x_range=[-2.5, 2.5],
                                     color=DELTA_COLOR, stroke_width=3)
        trad_label = MathTex(r"\sigma'(x)", font_size=28, color=DELTA_COLOR)
        trad_label.next_to(trad_axes, UP, buff=0.2)

        spike_axes = Axes(
            x_range=[-2.5, 2.5, 1], y_range=[0, 1.0, 0.5],
            x_length=ACT_X_LENGTH, y_length=self.BIG_Y_LENGTH,
            axis_config={"color": "#444444", "stroke_width": 1.5,
                         "include_tip": False, "font_size": 14},
        ).move_to([RIGHT_PANEL_X, self.PANEL_Y, 0])
        spike_curve = spike_axes.plot(dirac_bump, x_range=[-2.5, 2.5, 0.002],
                                       color=DELTA_COLOR, stroke_width=3)
        spike_label = MathTex(r"\delta(v-\theta)", font_size=28, color=DELTA_COLOR)
        spike_label.next_to(spike_axes, UP, buff=0.2)

        self.play(
            Create(trad_axes), Create(spike_axes),
            Write(trad_label), Write(spike_label),
        )
        self.play(Create(trad_curve), Create(spike_curve))
        self.wait(0.3)

        # A single arrow at each curve's peak. By height alone the spike
        # even looks bigger -- that's the setup for the twist.
        trad_peak_x, trad_peak_y = 0.0, sigmoid_prime(0.0)
        spike_peak_x, spike_peak_y = THETA, dirac_bump(THETA)

        trad_arrow = Arrow(trad_axes.c2p(trad_peak_x, 0), trad_axes.c2p(trad_peak_x, trad_peak_y),
                            color=MARKER_COLOR, buff=0, stroke_width=5,
                            max_tip_length_to_length_ratio=0.35)
        spike_arrow = Arrow(spike_axes.c2p(spike_peak_x, 0), spike_axes.c2p(spike_peak_x, spike_peak_y),
                             color=MARKER_COLOR, buff=0, stroke_width=5,
                             max_tip_length_to_length_ratio=0.15)
        trad_grad_label = crisp_text("gradient", font_size=18, color=MARKER_COLOR)
        trad_grad_label.next_to(trad_arrow, UR, buff=0.08)
        spike_grad_label = crisp_text("gradient", font_size=18, color=MARKER_COLOR)
        spike_grad_label.next_to(spike_arrow, RIGHT, buff=0.1)

        self.play(GrowArrow(trad_arrow), GrowArrow(spike_arrow))
        self.play(Write(trad_grad_label), Write(spike_grad_label))
        self.wait(0.3)

        twist = crisp_text("By height alone, these look comparable — the spike's even taller.",
                            font_size=22, color="#555555")
        twist.move_to([0, CAPTION_Y, 0])
        self.play(FadeIn(twist))
        self.wait(1.2)
        self.play(FadeOut(twist))

        # The real comparison: area under each curve.
        grid = np.linspace(-2.5, 2.5, 4001)
        trad_area_val = float(np.trapezoid(sigmoid_prime(grid), grid))
        spike_area_val = float(np.trapezoid(dirac_bump(grid), grid))
        ratio = trad_area_val / spike_area_val

        trad_area = trad_axes.get_area(trad_curve, x_range=[-2.5, 2.5],
                                        color=DELTA_COLOR, opacity=0.3)
        spike_area = spike_axes.get_area(spike_curve, x_range=[-2.5, 2.5],
                                          color=DELTA_COLOR, opacity=0.3)

        self.play(
            FadeOut(trad_arrow), FadeOut(spike_arrow),
            FadeOut(trad_grad_label), FadeOut(spike_grad_label),
            FadeIn(trad_area), FadeIn(spike_area),
        )

        trad_area_label = crisp_text(f"area ≈ {trad_area_val:.2f}", font_size=22, color=DELTA_COLOR)
        trad_area_label.move_to([LEFT_PANEL_X, EQN_Y, 0])
        spike_area_label = crisp_text(f"area ≈ {spike_area_val:.2f}", font_size=22, color=DELTA_COLOR)
        spike_area_label.move_to([RIGHT_PANEL_X, EQN_Y, 0])
        self.play(Write(trad_area_label), Write(spike_area_label))
        self.wait(0.3)

        ratio_label = crisp_text(
            f"~{ratio:.0f}× more gradient area to learn from",
            font_size=24, color=BLACK, weight=BOLD,
        )
        ratio_label.move_to([0, CAPTION_Y, 0])
        self.play(FadeIn(ratio_label, shift=UP * 0.15))
        self.wait(1.5)
        self.play(FadeOut(ratio_label))

        banner_text = crisp_text(
            "Gradients vanish almost everywhere — vanilla backprop cannot train spiking networks.",
            font_size=26, color=WHITE, weight=BOLD,
        )
        banner_text.set_width(min(banner_text.width, 12.4))
        banner = RoundedRectangle(
            width=banner_text.width + 0.6, height=banner_text.height + 0.4,
            corner_radius=0.12, color=DELTA_COLOR, fill_color=DELTA_COLOR,
            fill_opacity=1, stroke_width=0,
        )
        banner_group = VGroup(banner, banner_text).move_to([0, CAPTION_Y, 0])
        self.play(FadeIn(banner_group, shift=UP * 0.2))
        self.wait(2)
