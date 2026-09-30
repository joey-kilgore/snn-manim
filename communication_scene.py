"""
Manim animations contrasting how much communication happens in a dense
artificial neural network (ANN) versus a spiking neural network (SNN).

Both scenes stream the same number of ticks through a 5 -> 8 -> 8 -> 4 network
and share a side panel (message counters + a messages-per-tick bar chart on the
same scale), so the two videos can be cut back to back or side by side.

  * ANNCommunication - every tick is a full forward pass: every connection
                       carries a 32-bit value, every tick.
  * SNNCommunication - sparse wiring + diverse leaky integrate-and-fire
                       neurons: a connection only carries a message when its
                       source neuron spikes.
  * ANNvsSNN         - both networks side by side in one video, in lock-step.
  * ANNvsSNNClip1..4 - the same side-by-side run, split into 4 presentation
                       clips per communication_scene.md:
                         1. build both networks (sparsity + SNN diversity)
                         2. a single time step, slow
                         3. the remaining time steps, fast
                         4. the final tally: SNN traffic vs the ANN's

Render:
    manim -pql communication_scene.py ANNCommunication     # quick preview
    manim -pql communication_scene.py SNNCommunication
    manim -pqh communication_scene.py ANNCommunication SNNCommunication  # 1080p
    manim -pqh communication_scene.py ANNvsSNN              # side by side, one video
    manim -pqh communication_scene.py ANNvsSNNClip1 ANNvsSNNClip2 \\
                                       ANNvsSNNClip3 ANNvsSNNClip4  # as 4 separate clips
"""

from dataclasses import dataclass

from manim import *
import numpy as np

config.background_color = WHITE
Text.set_default(color=BLACK)

# ---------------------------------------------------------------------------
# Network / simulation parameters
# ---------------------------------------------------------------------------
LAYERS = [5, 8, 8, 4]
LAYER_NAMES = ["input", "hidden", "hidden", "output"]
N_EDGES = sum(a * b for a, b in zip(LAYERS, LAYERS[1:]))
TICKS = 8
DETAILED_TICKS = 2        # first N ticks are narrated slowly
FAST = 0.45               # run_time multiplier for un-narrated ticks

ANN_SEED = 3

# Sparse LIF network, tuned so activity is sparse but spikes still make it all
# the way to the output layer.
SNN_SEED = 57
CONN_P = 0.35             # P(a connection exists) between adjacent layers
INPUT_RATE = 0.25         # P(input neuron spikes) per tick
W_MEAN, W_STD = 0.8, 0.35
BETA = 0.85               # membrane leak per tick (output neurons)
THETA = 1.0               # firing threshold (output neurons)

# Hidden neuron types: (name, outline colour, leak per tick, threshold).
# Each hidden layer is laid out top-to-bottom in contiguous groups, one group
# per type, sized by NEURON_COUNTS (which must sum to the hidden layer size).
NEURON_TYPES = [
    ("Pyramidal", "#2E7D32", 0.85, 1.0),
    ("MFA ORDEN", PURPLE_D, 0.95, 1.4),
    ("Basket", "#C2185B", 0.6, 0.8),
]
NEURON_COUNTS = [5, 2, 1]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
EDGE = "#3a4556"
NEURON_STROKE = GREY_D
ACT = BLUE_E              # activation / membrane potential
PULSE = "#9C6E0C"         # a message travelling along a connection
MUTED = GREY_D

R = 0.18                  # neuron radius
V_GAP = 0.26              # vertical gap between neurons in a layer
NET_X = (-5.9, 0.3)       # x of first / last layer
NET_Y = -0.45
CAPTION_Y = 2.65

PANEL_X = 2.0             # left edge of the right-hand panel
VAL_X = 6.7               # right edge of panel values
CHART_BASE = -3.2
CHART_H = 2.0


def fit(mob, width):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class SNNRun:
    mask: list                # mask[l][i, j]: connection layer l i -> l+1 j exists
    types: list               # types[h][i]: NEURON_TYPES index, hidden layer h
    theta: list               # theta[l][i]: threshold of layer l+1 neuron i
    spikes: list              # spikes[t][l]: bool array, layer l fires at tick t
    v_next: list              # v_next[t][l]: layer l+1 potential after tick t

    @property
    def n_conn(self):
        return int(sum(m.sum() for m in self.mask))

    def messages(self, t):
        """Messages sent at tick t: each spike travels along its out-edges."""
        return sum(int(m[s].sum()) for m, s in zip(self.mask, self.spikes[t]))


def sparse_mask(rng, a, b):
    """Random a x b connectivity where every neuron keeps >= 1 in/out edge."""
    m = rng.random((a, b)) < CONN_P
    for i in np.where(~m.any(1))[0]:
        m[i, rng.integers(b)] = True
    for j in np.where(~m.any(0))[0]:
        m[rng.integers(a), j] = True
    return m


def hidden_types(n):
    """NEURON_TYPES index of each of n hidden neurons, top to bottom, grouped
    into contiguous blocks of NEURON_COUNTS (Pyramidal first, Basket last)."""
    assert sum(NEURON_COUNTS) == n, f"NEURON_COUNTS must sum to the hidden size {n}"
    return np.repeat(np.arange(len(NEURON_TYPES)), NEURON_COUNTS)


def simulate_snn(seed=SNN_SEED):
    """Sparse layered LIF network with a one-tick synaptic delay per layer."""
    rng = np.random.default_rng(seed)
    mask = [sparse_mask(rng, a, b) for a, b in zip(LAYERS, LAYERS[1:])]
    W = [np.where(m, rng.normal(W_MEAN, W_STD, m.shape), 0) for m in mask]
    types = [hidden_types(n) for n in LAYERS[1:-1]]
    beta = [np.array([NEURON_TYPES[k][2] for k in ty]) for ty in types]
    beta.append(np.full(LAYERS[-1], BETA))
    theta = [np.array([NEURON_TYPES[k][3] for k in ty]) for ty in types]
    theta.append(np.full(LAYERS[-1], THETA))

    v = [np.zeros(n) for n in LAYERS[1:]]
    spikes, v_next = [], []
    for _ in range(TICKS):
        s = [rng.random(LAYERS[0]) < INPUT_RATE] + [vl >= th for vl, th in zip(v, theta)]
        v = [np.maximum(beta[l] * np.where(s[l + 1], 0, v[l])
                        + s[l].astype(float) @ W[l], 0)
             for l in range(len(v))]
        spikes.append(s)
        v_next.append([vl.copy() for vl in v])
    return SNNRun(mask, types, theta, spikes, v_next)


def make_network(x_range, y, r=R, gap=V_GAP, name_size=18):
    """Neurons, edges and layer names for LAYERS laid out over x_range at y.

    edges[l][i][j] is the Line from layer l neuron i to layer l+1 neuron j.
    """
    layers = []
    for x, n in zip(np.linspace(*x_range, len(LAYERS)), LAYERS):
        col = VGroup(*[Circle(radius=r, stroke_color=NEURON_STROKE, stroke_width=2,
                              fill_color=ACT, fill_opacity=0)
                       for _ in range(n)])
        col.arrange(DOWN, buff=gap).move_to([x, y, 0])
        col.set_z_index(2)
        layers.append(col)

    edges = [[[Line(a, b, stroke_width=1.3, color=EDGE) for b in nxt] for a in cur]
             for cur, nxt in zip(layers, layers[1:])]

    bottom = min(col.get_bottom()[1] for col in layers)
    names = VGroup(*[Text(name, font_size=name_size, color=MUTED)
                     .move_to([col.get_x(), bottom - 1.6 * r, 0])
                     for name, col in zip(LAYER_NAMES, layers)])
    return layers, edges, names


def prune(edges, mask):
    """Set edges missing from `mask` to None and return the removed Lines."""
    pruned = []
    for l, m in enumerate(mask):
        for i, j in zip(*np.where(~m)):
            pruned.append(edges[l][i][j])
            edges[l][i][j] = None
    return pruned


def type_legend(font_size=18):
    return VGroup(*[
        VGroup(Circle(radius=0.1, stroke_color=color, stroke_width=3),
               Text(name, font_size=font_size, color=MUTED)).arrange(RIGHT, buff=0.12)
        for name, color, _, _ in NEURON_TYPES
    ]).arrange(RIGHT, buff=0.45)


def type_recolor(layers, types):
    """Animations colouring hidden neuron outlines by NEURON_TYPES index."""
    return [n.animate.set_stroke(NEURON_TYPES[k][1], width=3)
            for col, ty in zip(layers[1:-1], types)
            for n, k in zip(col, ty)]


# ---------------------------------------------------------------------------
# Shared layout
# ---------------------------------------------------------------------------
class NetworkScene(Scene):
    TITLE = ""
    SUBTITLE = ""
    PAYLOAD = ""

    def title_card(self):
        title = Text(self.TITLE, font_size=52, weight=BOLD)
        sub = Text(self.SUBTITLE, font_size=26, color=MUTED)
        fit(sub, 12.5).next_to(title, DOWN, buff=0.3)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

    # --------------------------------------------------------------- network
    def build_network(self, caption, mask=None, prune_caption=None):
        """Draw the dense network; with `mask`, then fade out missing edges."""
        self.header = Text(self.TITLE, font_size=34, weight=BOLD).to_edge(UP, buff=0.3)
        self.caption = None

        self.layers, self.edges, names = make_network(NET_X, NET_Y)
        all_edges = VGroup(*[e for layer in self.edges for row in layer for e in row])

        self.play(FadeIn(self.header))
        self.say(caption)
        self.play(LaggedStart(*[GrowFromCenter(n) for col in self.layers for n in col],
                              lag_ratio=0.04),
                  FadeIn(names), run_time=1.5)
        self.play(LaggedStart(*[Create(e) for e in all_edges], lag_ratio=0.015),
                  run_time=2.5)
        if mask is None:
            return

        pruned = prune(self.edges, mask)
        self.wait(0.5)
        self.say(prune_caption)
        self.play(*[e.animate.set_stroke(RED_E, width=2) for e in pruned], run_time=0.6)
        self.play(*[FadeOut(e) for e in pruned], run_time=1.2)
        self.wait(0.5)

    def say(self, text, rt=0.5):
        new = fit(Text(text, font_size=24), 12.8).move_to(UP * CAPTION_Y)
        if self.caption is not None:
            self.play(FadeOut(self.caption, shift=UP * 0.15),
                      FadeIn(new, shift=UP * 0.15), run_time=rt)
        else:
            self.play(FadeIn(new, shift=UP * 0.15), run_time=rt)
        self.caption = new

    def pulses(self, edges, rt):
        return [ShowPassingFlash(e.copy().set_stroke(PULSE, width=4).set_z_index(1),
                                 time_width=0.6, run_time=rt)
                for e in edges if e is not None]

    # ----------------------------------------------------------------- panel
    def row_value(self, text, y, color=BLACK):
        mob = Text(text, font_size=24, color=color)
        return mob.move_to([VAL_X - mob.width / 2, y, 0])

    def build_panel(self):
        ys = [1.7, 1.15, 0.6, 0.05]
        labels = ["tick", "messages this tick", "total messages", "each message"]
        self.row_y = dict(zip(["tick", "now", "total", "payload"], ys))
        lbls = VGroup(*[Text(t, font_size=24, color=MUTED)
                        .move_to([PANEL_X, y, 0], aligned_edge=LEFT)
                        for t, y in zip(labels, ys)])
        self.tick_lbl = self.row_value(f"0 / {TICKS}", ys[0])
        self.now_lbl = self.row_value("0", ys[1])
        self.total_lbl = self.row_value("0", ys[2])
        payload = self.row_value(self.PAYLOAD, ys[3], color=PULSE)
        self.total = 0

        # messages-per-tick bar chart, normalised to "every connection"
        chart_w = VAL_X - PANEL_X
        base = Line([PANEL_X, CHART_BASE, 0], [VAL_X, CHART_BASE, 0],
                    stroke_width=2, color=MUTED)
        top = DashedLine([PANEL_X, CHART_BASE + CHART_H, 0],
                         [VAL_X, CHART_BASE + CHART_H, 0],
                         stroke_width=1.5, color=MUTED, dash_length=0.08)
        top_lbl = Text(f"every connection ({N_EDGES})", font_size=16, color=MUTED)
        top_lbl.next_to(top, UP, buff=0.08).align_to(top, RIGHT)
        chart_title = Text("messages per tick", font_size=20, color=MUTED)
        chart_title.next_to(top_lbl, UP, buff=0.15).align_to(base, LEFT)
        slot = chart_w / TICKS
        self.bar_x = [PANEL_X + slot * (t + 0.5) for t in range(TICKS)]
        self.bar_w = slot * 0.62
        ticks = VGroup(*[Text(str(t + 1), font_size=16, color=MUTED)
                         .move_to([x, CHART_BASE - 0.22, 0])
                         for t, x in enumerate(self.bar_x)])

        self.play(FadeIn(lbls), FadeIn(self.tick_lbl), FadeIn(self.now_lbl),
                  FadeIn(self.total_lbl), FadeIn(payload),
                  Create(base), Create(top), FadeIn(top_lbl), FadeIn(chart_title),
                  FadeIn(ticks))

    def add_ref_line(self, n, label):
        """Extra dashed line on the chart at n messages per tick."""
        y = CHART_BASE + CHART_H * n / N_EDGES
        line = DashedLine([PANEL_X, y, 0], [VAL_X, y, 0], stroke_width=1.5,
                          color=MUTED, dash_length=0.08)
        lbl = Text(label, font_size=16, color=MUTED)
        lbl.next_to(line, UP, buff=0.08).align_to(line, RIGHT)
        self.play(Create(line), FadeIn(lbl))

    def swap(self, attr, text, y):
        old = getattr(self, attr)
        new = self.row_value(text, y)
        setattr(self, attr, new)
        return ReplacementTransform(old, new)

    def count(self, t, n):
        """Animations updating the counters for tick t sending n messages."""
        self.total += n
        return [self.swap("tick_lbl", f"{t + 1} / {TICKS}", self.row_y["tick"]),
                self.swap("now_lbl", f"{n:,}", self.row_y["now"]),
                self.swap("total_lbl", f"{self.total:,}", self.row_y["total"])]

    def add_bar(self, t, n):
        h = CHART_H * n / N_EDGES
        if h <= 0:
            return Wait(0)
        bar = Rectangle(width=self.bar_w, height=h, stroke_width=0,
                        fill_color=PULSE, fill_opacity=0.85)
        bar.move_to([self.bar_x[t], CHART_BASE, 0], aligned_edge=DOWN)
        return GrowFromEdge(bar, DOWN)

    def summary(self, lines, color=BLACK):
        box = VGroup(*[Text(l, font_size=26, color=color if i == 0 else MUTED)
                       for i, l in enumerate(lines)]).arrange(DOWN, buff=0.15)
        box.move_to(UP * CAPTION_Y)
        self.play(FadeOut(self.caption), FadeIn(box, shift=UP * 0.15))
        self.caption = box


# ---------------------------------------------------------------------------
# ANN: dense communication
# ---------------------------------------------------------------------------
class ANNCommunication(NetworkScene):
    TITLE = "Artificial Neural Network"
    SUBTITLE = "Dense layers: every neuron talks to every neuron in the next layer"
    PAYLOAD = "32-bit value"

    def construct(self):
        rng = np.random.default_rng(ANN_SEED)
        W = [rng.normal(0, 1.5 / np.sqrt(a), (a, b)) for a, b in zip(LAYERS, LAYERS[1:])]

        self.title_card()
        self.build_network(f"Fully connected: {N_EDGES} connections")
        self.build_panel()

        for t in range(TICKS):
            acts = [rng.random(LAYERS[0])]
            for w in W:
                acts.append(sigmoid((acts[-1] - 0.5) @ w * 2))
            self.tick(t, acts, narrate=t < DETAILED_TICKS)

        self.summary([f"All {N_EDGES} connections fire on every tick",
                      f"{self.total:,} messages, whatever the input"])
        self.wait(3)

    def fill(self, layer, values, rt):
        return [n.animate(run_time=rt).set_fill(ACT, opacity=0.15 + 0.85 * a)
                for n, a in zip(self.layers[layer], values)]

    def tick(self, t, acts, narrate):
        f = 1.0 if narrate else FAST
        if t == 0:
            self.say("A new input arrives every tick")
        # clear the previous tick's activity, then load the new input
        clear = [n.animate.set_fill(opacity=0) for col in self.layers[1:] for n in col]
        self.play(*clear, *self.fill(0, acts[0], 0.5 * f), run_time=0.5 * f)

        for l in range(len(LAYERS) - 1):
            n_edges = LAYERS[l] * LAYERS[l + 1]
            if t == 0 and l == 0:
                self.say(f"Every input neuron sends its value to all "
                         f"{LAYERS[1]} hidden neurons: {n_edges} messages")
            elif t == 0 and l == 1:
                self.say("...and every hidden neuron does the same, "
                         "even if its value is tiny")
            edges = [e for row in self.edges[l] for e in row]
            self.play(*self.pulses(edges, 0.9 * f), run_time=0.9 * f)
            self.play(*self.fill(l + 1, acts[l + 1], 0.35 * f), run_time=0.35 * f)

        if t == 1:
            self.say(f"Every tick: all {N_EDGES} connections carry a value")
        self.play(*self.count(t, N_EDGES), self.add_bar(t, N_EDGES), run_time=0.5 * f)
        if narrate:
            self.wait(0.5)


# ---------------------------------------------------------------------------
# SNN: sparse, event-driven communication
# ---------------------------------------------------------------------------
class SNNCommunication(NetworkScene):
    TITLE = "Spiking Neural Network"
    SUBTITLE = "Sparse wiring, and neurons only communicate when they spike"
    PAYLOAD = "1-bit spike"

    def construct(self):
        self.snn = simulate_snn()
        n_conn = self.snn.n_conn

        self.title_card()
        self.build_network(f"A dense network would have {N_EDGES} connections",
                           mask=self.snn.mask,
                           prune_caption=f"The SNN keeps only {n_conn} of them: sparse wiring")
        self.show_diversity()
        self.build_panel()
        self.add_ref_line(n_conn, f"every SNN connection ({n_conn})")

        for t in range(TICKS):
            self.tick(t, narrate=t < DETAILED_TICKS)

        pct = 100 * self.total / (N_EDGES * TICKS)
        self.summary([f"{self.total} messages vs {N_EDGES * TICKS:,} for the dense ANN ({pct:.0f}%)",
                      "Sparse wiring + sparse spikes: most of the network stays silent"])
        self.wait(3)

    def show_diversity(self):
        """Colour hidden neuron outlines by type and add a legend."""
        recolor = type_recolor(self.layers, self.snn.types)
        legend = type_legend().move_to([np.mean(NET_X), -3.65, 0])

        self.say("Hidden neurons are diverse: each type has its own leak and threshold")
        self.play(*recolor, FadeIn(legend, shift=UP * 0.1), run_time=1.2)
        self.wait(1)

    def tick(self, t, narrate):
        f = 1.0 if narrate else FAST
        s, v_next = self.snn.spikes[t], self.snn.v_next[t]
        spiking = [n for l, col in enumerate(self.layers)
                   for n, fired in zip(col, s[l]) if fired]
        out_edges = [e for l in range(len(LAYERS) - 1)
                     for i, fired in enumerate(s[l]) if fired
                     for e in self.edges[l][i]]
        n_msgs = self.snn.messages(t)

        if t == 0:
            self.say("Only a few input neurons spike this tick")
        elif t == 1:
            self.say("A neuron fires only when its potential crosses threshold")

        # 1. spike: flash, then reset the membrane potential
        if spiking:
            self.play(*[n.animate.set_fill(PULSE, opacity=1) for n in spiking],
                      *[Flash(n, color=PULSE, flash_radius=R + 0.1, line_length=0.12,
                              num_lines=10) for n in spiking],
                      run_time=0.45 * f)

        # 2. transmit: only the spiking neurons' connections carry a message
        if t == 0:
            self.say("Each spike is a 1-bit event sent along that neuron's connections only")
        anims = self.pulses(out_edges, 0.9 * f)
        anims += [n.animate.set_fill(ACT, opacity=0) for n in spiking]
        self.play(*(anims or [Wait()]), run_time=0.9 * f)

        # 3. integrate: downstream membrane potentials leak + rise with input
        if t == 0:
            self.say("Receiving neurons integrate the spikes; everything else stays silent")
        fills = [n.animate.set_fill(ACT, opacity=0.9 * min(v / th, 1))
                 for l, col in enumerate(self.layers[1:])
                 for n, v, th in zip(col, v_next[l], self.snn.theta[l])]
        self.play(*fills, run_time=0.4 * f)

        self.play(*self.count(t, n_msgs), self.add_bar(t, n_msgs), run_time=0.5 * f)
        if narrate:
            self.wait(0.8)


# ---------------------------------------------------------------------------
# Side by side: both networks on one clock
# ---------------------------------------------------------------------------
HALF_X = 3.55             # centre x of each half
HALF_W = 4.8              # width of each network / chart
SBS_NET_Y = 0.7
SBS_R, SBS_GAP = 0.14, 0.2
SBS_CAPTION_Y = 2.95
SBS_LEGEND_Y = -1.75
SBS_COUNT_Y = -2.2
SBS_CHART_BASE = -3.65
SBS_CHART_H = 0.95


class Half:
    """One network plus its caption, counters and mini chart."""

    def __init__(self, cx, header):
        self.cx = cx
        self.left, self.right = cx - HALF_W / 2, cx + HALF_W / 2
        self.header = Text(header, font_size=36, weight=BOLD).move_to([cx, 3.5, 0])
        self.caption = None
        self.layers, self.edges, self.names = make_network(
            (self.left + 0.2, self.right - 0.2), SBS_NET_Y, SBS_R, SBS_GAP, name_size=15)
        self.total = 0

    def all_edges(self):
        return [e for layer in self.edges for row in layer for e in row if e is not None]

    def say(self, text):
        """Animations swapping this half's caption."""
        new = fit(Text(text, font_size=20), HALF_W + 1.2).move_to([self.cx, SBS_CAPTION_Y, 0])
        old, self.caption = self.caption, new
        anims = [FadeIn(new, shift=UP * 0.12)]
        if old is not None:
            anims.append(FadeOut(old, shift=UP * 0.12))
        return anims

    # ----------------------------------------------------------- counters
    def value(self, text, right_x):
        mob = Text(text, font_size=20)
        return mob.move_to([right_x - mob.width / 2, SBS_COUNT_Y, 0])

    def build_panel(self):
        lbls = VGroup(
            Text("time", font_size=20, color=MUTED)
            .move_to([self.left, SBS_COUNT_Y, 0], aligned_edge=LEFT),
            Text("total", font_size=20, color=MUTED)
            .move_to([self.cx + 0.3, SBS_COUNT_Y, 0], aligned_edge=LEFT),
        )
        self.now_x, self.total_x = self.cx - 0.25, self.right
        self.now_lbl = self.value("0", self.now_x)
        self.total_lbl = self.value("0", self.total_x)

        base = Line([self.left, SBS_CHART_BASE, 0], [self.right, SBS_CHART_BASE, 0],
                    stroke_width=2, color=MUTED)
        slot = HALF_W / TICKS
        self.bar_x = [self.left + slot * (t + 0.5) for t in range(TICKS)]
        self.bar_w = slot * 0.62
        return [FadeIn(lbls), FadeIn(self.now_lbl), FadeIn(self.total_lbl), Create(base),
                *self.ref_line(N_EDGES, f"every connection ({N_EDGES})")]

    def ref_line(self, n, label):
        """Dashed chart line at n messages per tick (chart scale: N_EDGES)."""
        y = SBS_CHART_BASE + SBS_CHART_H * n / N_EDGES
        line = DashedLine([self.left, y, 0], [self.right, y, 0], stroke_width=1.5,
                          color=MUTED, dash_length=0.08)
        lbl = Text(label, font_size=14, color=MUTED)
        lbl.next_to(line, UP, buff=0.06).align_to(line, RIGHT)
        return [Create(line), FadeIn(lbl)]

    def count(self, t, n, animate=True):
        """Update the counters and add tick t's bar.

        animate=True returns Animations to play. animate=False mutates the
        counters in place (via `become`, so they stay the same Mobjects) and
        returns just the new bar (if any) to `self.add` - used to fast-forward
        earlier ticks instantly before a clip's own animated ticks begin.
        """
        self.total += n
        new_now = self.value(f"{n:,}", self.now_x)
        new_total = self.value(f"{self.total:,}", self.total_x)
        bar = None
        if n > 0:
            bar = Rectangle(width=self.bar_w, height=SBS_CHART_H * n / N_EDGES,
                            stroke_width=0, fill_color=PULSE, fill_opacity=0.85)
            bar.move_to([self.bar_x[t], SBS_CHART_BASE, 0], aligned_edge=DOWN)

        if not animate:
            self.now_lbl.become(new_now)
            self.total_lbl.become(new_total)
            return [bar] if bar else []

        old_now, old_total = self.now_lbl, self.total_lbl
        self.now_lbl, self.total_lbl = new_now, new_total
        anims = [ReplacementTransform(old_now, new_now), ReplacementTransform(old_total, new_total)]
        if bar:
            anims.append(GrowFromEdge(bar, DOWN))
        return anims


def half_legends(ann, spk):
    """Legends: 'identical units' for the ANN half, neuron types for the SNN half."""
    ann_legend = VGroup(Circle(radius=0.1, stroke_color=NEURON_STROKE, stroke_width=3),
                        Text("identical units", font_size=15, color=MUTED))
    ann_legend.arrange(RIGHT, buff=0.12).move_to([ann.cx, SBS_LEGEND_Y, 0])
    spk_legend = type_legend(font_size=15).move_to([spk.cx, SBS_LEGEND_Y, 0])
    return ann_legend, spk_legend


ANN_CAPTION = "Every connection sends a 32-bit value, every tick"
SNN_CAPTION = "Only spiking neurons send a 1-bit event"


class ANNvsSNNBase(NetworkScene):
    """ANN (left) and SNN (right) side by side, ticking in lock-step.

    The ANN is pipelined - every layer forwards its current values every tick -
    so it runs on the same clock as the SNN's one-tick-per-layer delay.

    Shared by the full ANNvsSNN video and the four ANNvsSNNClip* scenes, which
    split the same run into separate clips for a presentation (see
    communication_scene.md). Every clip re-derives the identical run from the
    fixed ANN_SEED / SNN_SEED, so a later clip's `static_setup` + fast-forward
    reproduces exactly what an earlier clip ended on.
    """

    def setup_run(self):
        """The simulation plus the two (still empty) network halves."""
        snn = simulate_snn()
        rng = np.random.default_rng(ANN_SEED)
        W = [rng.normal(0, 1.5 / np.sqrt(a), (a, b)) for a, b in zip(LAYERS, LAYERS[1:])]
        acts = [np.zeros(n) for n in LAYERS]
        ann, spk = Half(-HALF_X, "ANN"), Half(HALF_X, "SNN")
        return ann, spk, snn, rng, W, acts

    def tick_text(self, t):
        return Text(f"tick {t} / {TICKS}", font_size=22, color=MUTED).move_to(UP * 3.5)

    def tick_swap(self, t):
        old, self.tick_lbl = self.tick_lbl, self.tick_text(t)
        return ReplacementTransform(old, self.tick_lbl)

    # ------------------------------------------------------------- Clip 1
    def build(self, ann, spk, snn):
        """Animated: draw both networks, prune the SNN, colour its neurons."""
        divider = DashedLine(UP * 3.2, DOWN * 3.9, stroke_width=1.5, color=EDGE,
                             dash_length=0.12)
        self.play(FadeIn(ann.header), FadeIn(spk.header), Create(divider),
                  *ann.say("Artificial neural network: fully connected"),
                  *spk.say(f"Spiking neural network: same {N_EDGES} connections to start"))
        self.play(*[LaggedStart(*[GrowFromCenter(n) for col in h.layers for n in col],
                                lag_ratio=0.04) for h in (ann, spk)],
                  FadeIn(ann.names), FadeIn(spk.names), run_time=1.5)
        self.play(*[LaggedStart(*[Create(e) for e in h.all_edges()], lag_ratio=0.015)
                    for h in (ann, spk)], run_time=2.5)

        # SNN: sparse wiring
        pruned = prune(spk.edges, snn.mask)
        self.play(*spk.say(f"Sparse wiring: only {snn.n_conn} of {N_EDGES} connections kept"))
        self.play(*[e.animate.set_stroke(RED_E, width=2) for e in pruned], run_time=0.6)
        self.play(*[FadeOut(e) for e in pruned], run_time=1.2)
        self.wait(1)

        # neuron diversity
        ann_legend, spk_legend = half_legends(ann, spk)
        self.play(*ann.say("Every neuron is the same kind of unit"),
                  *spk.say("Diverse neuron types, each with its own leak and threshold"))
        self.play(*type_recolor(spk.layers, snn.types),
                  FadeIn(ann_legend, shift=UP * 0.1), FadeIn(spk_legend, shift=UP * 0.1),
                  run_time=1.2)
        self.wait(1)

        # counters + charts
        self.tick_lbl = self.tick_text(0)
        self.play(*ann.build_panel(), *spk.build_panel(),
                  *spk.ref_line(snn.n_conn, f"every SNN connection ({snn.n_conn})"),
                  FadeIn(self.tick_lbl))
        self.play(*ann.say(ANN_CAPTION), *spk.say(SNN_CAPTION))
        self.wait(0.5)

    # ------------------------------------------------------ Clips 2, 3, 4
    def static_setup(self, ann, spk, snn):
        """Instantly place the scene exactly as Clip 1 leaves it, with no
        animation - so Clips 2-4 can pick up mid-run on their own."""
        divider = DashedLine(UP * 3.2, DOWN * 3.9, stroke_width=1.5, color=EDGE,
                             dash_length=0.12)
        prune(spk.edges, snn.mask)  # drop pruned edges before they're ever added
        for col, ty in zip(spk.layers[1:-1], snn.types):
            for n, k in zip(col, ty):
                n.set_stroke(NEURON_TYPES[k][1], width=3)
        ann_legend, spk_legend = half_legends(ann, spk)

        ann.caption = fit(Text(ANN_CAPTION, font_size=20), HALF_W + 1.2) \
            .move_to([ann.cx, SBS_CAPTION_Y, 0])
        spk.caption = fit(Text(SNN_CAPTION, font_size=20), HALF_W + 1.2) \
            .move_to([spk.cx, SBS_CAPTION_Y, 0])
        self.tick_lbl = self.tick_text(0)

        self.add(divider, ann.header, spk.header,
                 *ann.layers, ann.names, *ann.all_edges(),
                 *spk.layers, spk.names, *spk.all_edges(),
                 ann_legend, spk_legend, ann.caption, spk.caption, self.tick_lbl,
                 *[a.mobject for a in ann.build_panel()],
                 *[a.mobject for a in spk.build_panel()],
                 *[a.mobject for a in spk.ref_line(snn.n_conn, f"every SNN connection ({snn.n_conn})")])

    def tick_step(self, t, ann, spk, snn, rng, W, acts, animate, narrate=False):
        """Advance the simulation by tick t.

        animate=True plays the tick's animation (slow if `narrate`, fast
        otherwise). animate=False applies the tick's end state instantly -
        used to fast-forward earlier ticks before a clip's own animated ticks
        begin.
        """
        acts[0] = rng.random(LAYERS[0])
        s, v_next = snn.spikes[t], snn.v_next[t]
        spiking = [n for l, col in enumerate(spk.layers)
                   for n, fired in zip(col, s[l]) if fired]
        spike_edges = [e for l in range(len(LAYERS) - 1)
                       for i, fired in enumerate(s[l]) if fired
                       for e in spk.edges[l][i]]
        acts[1:] = [sigmoid((acts[l] - 0.5) @ W[l] * 2) for l in range(len(W))]

        if not animate:
            for n, a in zip(ann.layers[0], acts[0]):
                n.set_fill(ACT, opacity=0.15 + 0.85 * a)
            for col, a_l in zip(ann.layers[1:], acts[1:]):
                for n, a in zip(col, a_l):
                    n.set_fill(ACT, opacity=0.15 + 0.85 * a)
            for n in spiking:
                n.set_fill(ACT, opacity=0)
            for col, v_l, th_l in zip(spk.layers[1:], v_next, snn.theta):
                for n, v, th in zip(col, v_l, th_l):
                    n.set_fill(ACT, opacity=0.9 * min(v / th, 1))
            self.tick_lbl.become(self.tick_text(t + 1))
            self.add(*ann.count(t, N_EDGES, animate=False),
                     *spk.count(t, snn.messages(t), animate=False))
            return

        f = 1.0 if narrate else FAST
        # 1. ANN loads a new input frame; SNN neurons over threshold spike
        self.play(*[n.animate.set_fill(ACT, opacity=0.15 + 0.85 * a)
                    for n, a in zip(ann.layers[0], acts[0])],
                  *[n.animate.set_fill(PULSE, opacity=1) for n in spiking],
                  *[Flash(n, color=PULSE, flash_radius=SBS_R + 0.08, line_length=0.1,
                          num_lines=10) for n in spiking],
                  run_time=0.45 * f)

        # 2. transmit: every ANN connection vs only the spiking neurons' ones
        self.play(*self.pulses(ann.all_edges(), 0.9 * f),
                  *self.pulses(spike_edges, 0.9 * f),
                  *[n.animate.set_fill(ACT, opacity=0) for n in spiking],
                  run_time=0.9 * f)

        # 3. receive: ANN layers take their new values, SNN neurons integrate
        self.play(*[n.animate.set_fill(ACT, opacity=0.15 + 0.85 * a)
                    for col, a_l in zip(ann.layers[1:], acts[1:])
                    for n, a in zip(col, a_l)],
                  *[n.animate.set_fill(ACT, opacity=0.9 * min(v / th, 1))
                    for col, v_l, th_l in zip(spk.layers[1:], v_next, snn.theta)
                    for n, v, th in zip(col, v_l, th_l)],
                  run_time=0.4 * f)

        self.play(*ann.count(t, N_EDGES), *spk.count(t, snn.messages(t)),
                  self.tick_swap(t + 1), run_time=0.5 * f)
        if narrate:
            self.wait(0.6)

    # ------------------------------------------------------------- Clip 4
    def summary(self, ann, spk):
        pct = 100 * spk.total / ann.total
        self.play(*ann.say(f"{ann.total:,} messages: all {N_EDGES} connections, every tick"),
                  *spk.say(f"{spk.total} messages: {pct:.0f}% of the ANN's traffic"))
        self.play(Indicate(ann.total_lbl, color=PULSE), Indicate(spk.total_lbl, color=PULSE))


class ANNvsSNN(ANNvsSNNBase):
    """The full run in one continuous video (all four clips back to back)."""

    def construct(self):
        ann, spk, snn, rng, W, acts = self.setup_run()
        self.build(ann, spk, snn)
        for t in range(TICKS):
            self.tick_step(t, ann, spk, snn, rng, W, acts,
                           animate=True, narrate=t < DETAILED_TICKS)
        self.summary(ann, spk)
        self.wait(3)


class ANNvsSNNClip1(ANNvsSNNBase):
    """Clip 1: build both networks, showing the SNN's sparsity and its
    diverse neuron types."""

    def construct(self):
        ann, spk, snn, rng, W, acts = self.setup_run()
        self.build(ann, spk, snn)
        self.wait(1)


class ANNvsSNNClip2(ANNvsSNNBase):
    """Clip 2: a single time step, at a slow, narrated pace."""

    def construct(self):
        ann, spk, snn, rng, W, acts = self.setup_run()
        self.static_setup(ann, spk, snn)
        self.tick_step(0, ann, spk, snn, rng, W, acts, animate=True, narrate=True)
        self.wait(1)


class ANNvsSNNClip3(ANNvsSNNBase):
    """Clip 3: the remaining time steps, at a faster pace."""

    def construct(self):
        ann, spk, snn, rng, W, acts = self.setup_run()
        self.static_setup(ann, spk, snn)
        self.tick_step(0, ann, spk, snn, rng, W, acts, animate=False)  # fast-forward Clip 2
        for t in range(1, TICKS):
            self.tick_step(t, ann, spk, snn, rng, W, acts, animate=True, narrate=False)
        self.wait(1)


class ANNvsSNNClip4(ANNvsSNNBase):
    """Clip 4: the final tally, emphasising how little the SNN communicates
    relative to the ANN."""

    def construct(self):
        ann, spk, snn, rng, W, acts = self.setup_run()
        self.static_setup(ann, spk, snn)
        for t in range(TICKS):
            self.tick_step(t, ann, spk, snn, rng, W, acts, animate=False)
        self.wait(0.5)
        self.summary(ann, spk)
        self.wait(3)
