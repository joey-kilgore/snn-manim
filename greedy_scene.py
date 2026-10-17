"""
Manim clips for the greedy block placement in generate_hw_mappings.py.

Convention: pre-synaptic neurons are COLUMNS (x axis, inputs enter from the
top), post-synaptic neurons are ROWS (y axis, outputs are read on the side).
A K = (pre_K, post_K) kernel is pre_K columns wide and post_K rows tall.

Each clip is its own Scene so it can be dropped into a presentation. A clip
rebuilds the exact state the previous clip ended on (instantly, no animation)
and then animates only its own slice of the run:

    Clip1  pre/post neurons drawn as a block of connections
    Clip2  build the matrix + parameter panel, first block placement
    Clip3  more placements, up to the first rejection
    Clip4  more rejections until a block is finally accepted
    Clip5  continue until the 6th rejection in a row
    Clip6  what happens when the failure budget is exhausted
    Clip7  remaining smaller blocks, then the final blocks pulled out
    Clip8  pack the blocks into one 25 x 25 crossbar tile
    Clip9  spikes -> read voltages -> current for a single block
    GreedyFull    clips 1-8 as one ~1 min video, sampling sped up
    DefaultsCard  (extra) the real defaults from generate_hw_mappings.py

Render:
    manim -ql greedy_scene.py Clip1 Clip2 Clip3 Clip4 Clip5 Clip6 Clip7 Clip8 Clip9
    manim -ql greedy_scene.py GreedyFull           # clips 1-8 in one video
    manim -qh greedy_scene.py Clip1 ...            # 1080p
"""

import numpy as np
from manim import *

from greedy_blocking import greedy_blocking

config.background_color = WHITE

# ---------------------------------------------------------------------------
# Demo parameters (scaled down from the script defaults so every step is
# visible: TILE_SIZE=25, fallback 5, min_accept_frac 0.98, decay 0.995,
# max_tries 5000)
# ---------------------------------------------------------------------------
PRE_N, POST_N = 25, 25
PRE_K, POST_K = 12, 1
FALLBACK_K = 5            # on stall, kernel -> min(K, 5) per dim = 5 x 1
TARGET_DENSITY = 0.45
MIN_ACCEPT_FRAC = 1.0
DECAY = 0.75
MAX_TRIES = 6
SEED = 40
TILE_SIZE = 25            # Daffodil crossbar dimension
DETAILED_TRIES = 3        # first N proposals are narrated slowly
FAST = 0.3                # run_time multiplier for un-narrated proposals

EMPTY = "#EDEFF2"
GRID = "#C7CDD6"
WIRE = "#8A93A3"
EDGE = BLUE_E
OVERLAP = "#B15400"
PROPOSE = "#9C6E0C"
OK = "#2E7D32"
BAD = "#B22222"
MUTED = GREY_D

CELL = 0.19
HIST_LEN = 1.0            # length of the longest in/out-degree bar
PANEL_X = 0.7             # left edge of the right-hand panel
PANEL_W = 6.1


Text.set_default(font="Segoe UI", color=BLACK)


def fit(mob, width=PANEL_W):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def plan_narration(events):
    """Which proposals get the slow, explained treatment. Computed once over
    the whole run so every clip agrees regardless of where it starts."""
    flags, seen, n_try, after_stall = [], set(), 0, False
    for ev in events:
        if ev.kind == "try":
            if not ev.accepted:
                kind = "reject"
            elif ev.new < ev.size:
                kind = "accept-with-duplicates"
            else:
                kind = "accept"
            flags.append(n_try < DETAILED_TRIES or after_stall or kind not in seen)
            seen.add(kind)
            after_stall = False
            n_try += 1
        else:
            flags.append(True)
            after_stall = ev.kind in ("fallback", "decay")
    return flags


def clip_marks(events):
    """Event-index slices [a, b) for clips 2-7, found from the run itself."""
    tries = [(i, e) for i, e in enumerate(events) if e.kind == "try"]
    first_reject = next(i for i, e in tries if not e.accepted)
    accept_after = next(i for i, e in tries if i > first_reject and e.accepted)
    sixth = next(i for i, e in tries if e.tries >= MAX_TRIES)
    assert events[sixth + 1].kind in ("fallback", "decay")
    return {
        "c2": (0, 1),
        "c3": (1, first_reject + 1),
        "c4": (first_reject + 1, accept_after + 1),
        "c5": (accept_after + 1, sixth + 1),
        "c6": (sixth + 1, sixth + 2),
        "c7": (sixth + 2, len(events)),
    }


def first_fit(shapes, size=TILE_SIZE):
    """Illustrative packer: drop each (width, height) block into the first
    free spot of a size x size tile, scanning rows top to bottom."""
    used = np.zeros((size, size), dtype=bool)
    slots = []
    for w, h in shapes:
        for i in range(size - h + 1):
            spot = next((j for j in range(size - w + 1)
                         if not used[i:i + h, j:j + w].any()), None)
            if spot is not None:
                used[i:i + h, spot:spot + w] = True
                slots.append((i, spot))
                break
        else:
            raise ValueError("blocks do not fit in one tile")
    return slots, used


class GreedyBase(Scene):
    def setup(self):
        super().setup()
        self.res = greedy_blocking(
            PRE_N, POST_N, PRE_K, POST_K, TARGET_DENSITY,
            min_accept_frac=MIN_ACCEPT_FRAC, decay=DECAY, max_tries=MAX_TRIES,
            fallback_K=FALLBACK_K, seed=SEED,
        )
        self.events = self.res.events
        self.narrate = plan_narration(self.events)
        self.marks = clip_marks(self.events)

    # ---------------------------------------------------------------- pacing
    pace = 1.0                # multiplies every run_time and wait

    def compile_animations(self, *args, **kwargs):
        anims = super().compile_animations(*args, **kwargs)
        if self.pace != 1:
            # floor at one frame ourselves so manim doesn't warn on every step
            for anim in anims:
                anim.run_time = max(anim.run_time * self.pace, 1 / config.frame_rate)
        return anims

    # ------------------------------------------------------------ geometry
    def cell_center(self, r, c):
        """r = pre-synaptic index -> column (x); c = post-synaptic index -> row (y)."""
        return self.origin + RIGHT * r * CELL + DOWN * c * CELL

    def build_tile(self, center):
        """An empty TILE_SIZE x TILE_SIZE crossbar tile centered at `center`."""
        t0 = center + LEFT * (TILE_SIZE - 1) * CELL / 2 + UP * (TILE_SIZE - 1) * CELL / 2
        cell_pos = lambda i, j: t0 + RIGHT * j * CELL + DOWN * i * CELL
        cells = [[Square(CELL, stroke_color=GRID, stroke_width=1,
                         fill_color=EMPTY, fill_opacity=1).move_to(cell_pos(i, j))
                  for j in range(TILE_SIZE)] for i in range(TILE_SIZE)]
        return cells, cell_pos

    def state_after(self, n):
        """Everything the display shows after the first n events."""
        st = dict(K=(min(PRE_K, PRE_N), min(POST_K, POST_N)), thr=MIN_ACCEPT_FRAC,
                  blocks=0, prog=0, tries=0,
                  counts=np.zeros((PRE_N, POST_N), dtype=int))
        for ev in self.events[:n]:
            st.update(K=ev.K, thr=ev.threshold, blocks=ev.blocks, prog=ev.progress,
                      tries=ev.tries if ev.kind == "try" and not ev.accepted else 0)
            if ev.kind == "try" and ev.accepted:
                st["counts"][np.ix_(ev.pre, ev.post)] += 1
        return st

    # ----------------------------------------------------------- workspace
    def build_workspace(self, animate=True, upto=0):
        """Matrix, histograms and parameter panel. With animate=False the
        display is placed instantly in the state it had after `upto` events."""
        st = self.state_after(upto)
        counts = st["counts"]

        header = Text("Greedy block placement", font_size=34, weight=BOLD)
        header.to_edge(UP, buff=0.3)

        W, H = PRE_N * CELL, POST_N * CELL
        self.W, self.H = W, H
        center = LEFT * 4.15 + UP * 0.3
        self.origin = center + LEFT * (W - CELL) / 2 + UP * (H - CELL) / 2
        self.cells = [[Square(CELL, stroke_color=GRID, stroke_width=1,
                              fill_color=(EMPTY if counts[r][c] == 0 else
                                          EDGE if counts[r][c] == 1 else OVERLAP),
                              fill_opacity=1)
                       .move_to(self.cell_center(r, c))
                       for c in range(POST_N)] for r in range(PRE_N)]
        grid = VGroup(*[sq for row in self.cells for sq in row])
        self.counts = counts.tolist()

        top = Text(f"pre-synaptic neurons ({PRE_N})", font_size=20, color=MUTED)
        top.next_to(grid, UP, buff=0.15)
        left = Text(f"post-synaptic neurons ({POST_N})", font_size=20, color=MUTED)
        left.rotate(PI / 2).next_to(grid, LEFT, buff=0.15)

        # --- right panel
        r = self.res
        self.stat_target = Text(
            f"Target: {TARGET_DENSITY:.0%} of {PRE_N}×{POST_N} = {r.target} synapses",
            font_size=22)
        self.stat_kernel = self.kernel_text(st["K"])
        self.stat_thr = self.thr_text(st["thr"])
        self.stat_blocks = self.blocks_text(st["blocks"])

        tries_lbl = Text(f"Rejections in a row (max_tries = {MAX_TRIES})", font_size=20)
        self.dots = VGroup(*[Circle(0.1, stroke_color=BAD, stroke_width=2,
                                    fill_color=BAD,
                                    fill_opacity=1 if i < st["tries"] else 0)
                             for i in range(MAX_TRIES)]).arrange(RIGHT, buff=0.15)
        tries = VGroup(tries_lbl, self.dots).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        self.prog_lbl = self.prog_text(st["prog"])
        self.bar_frame = Rectangle(width=PANEL_W - 0.4, height=0.25,
                                   stroke_color=MUTED, stroke_width=1.5)
        prog = VGroup(self.prog_lbl, self.bar_frame).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        panel = VGroup(self.stat_target, self.stat_kernel, self.stat_thr,
                       tries, self.stat_blocks, prog)
        panel.arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        panel.next_to(header, DOWN, buff=0.45).align_to(RIGHT * PANEL_X, LEFT)

        self.progress = ValueTracker(st["prog"])
        frame = self.bar_frame
        self.bar = always_redraw(lambda: Rectangle(
            width=max(1e-3, frame.width * min(1, self.progress.get_value() / r.target)),
            height=frame.height, stroke_width=0, fill_color=OK, fill_opacity=0.9,
        ).align_to(frame, LEFT).align_to(frame, UP))

        # readout area below the panel
        self.readout_anchor = RIGHT * PANEL_X + UP * (panel.get_bottom()[1] - 0.45)
        self.readout = VGroup()

        legend = VGroup(
            self.legend_item(EDGE, "synapse"),
            self.legend_item(OVERLAP, "duplicate (placed twice)"),
        ).arrange(RIGHT, buff=0.5)

        hists = self.build_histograms(grid, counts)
        legend.scale(0.85).next_to(self.bottom_title, DOWN, buff=0.12).set_x(grid.get_x())

        if animate:
            self.play(FadeIn(header), FadeIn(top), FadeIn(left),
                      LaggedStart(*[FadeIn(sq) for sq in grid], lag_ratio=0.004, run_time=1.2))
            self.play(FadeIn(hists))
            self.play(FadeIn(panel, shift=LEFT * 0.2), FadeIn(self.bar), FadeIn(legend))
            self.wait(1)
        else:
            self.add(header, top, left, grid, hists, panel, self.bar, legend)
            if upto:
                self.readout = self.place_readout(self.readout_lines(upto - 1))
                self.add(self.readout)

    # ------------------------------------------------------------ histograms
    def build_histograms(self, grid, counts):
        """Per-neuron unique synapse counts. Pre-synaptic neurons are columns,
        so their out-degree bars hang below each column; post-synaptic
        neurons are rows, so their in-degree bars extend right of each row."""
        final = self.res.matrix > 0
        self.hist_max = max(final.sum(axis=1).max(), final.sum(axis=0).max())
        self.out_y0 = grid.get_bottom()[1] - 0.1
        self.in_x0 = grid.get_right()[0] + 0.1
        self.out_deg = [int((counts[r] > 0).sum()) for r in range(PRE_N)]
        self.in_deg = [int((counts[:, c] > 0).sum()) for c in range(POST_N)]
        self.out_bars = VGroup(*[self.out_bar(r, self.out_deg[r]) for r in range(PRE_N)])
        self.in_bars = VGroup(*[self.in_bar(c, self.in_deg[c]) for c in range(POST_N)])

        top, bottom = grid.get_top()[1], grid.get_bottom()[1]
        left, right = grid.get_left()[0], grid.get_right()[0]
        out_axis = Line([left, self.out_y0, 0], [right, self.out_y0, 0],
                        stroke_width=1.5, color=MUTED)
        in_axis = Line([self.in_x0, top, 0], [self.in_x0, bottom, 0],
                       stroke_width=1.5, color=MUTED)

        out_avg = TARGET_DENSITY * POST_N
        in_avg = TARGET_DENSITY * PRE_N
        oy = self.out_y0 - HIST_LEN * out_avg / self.hist_max
        ix = self.in_x0 + HIST_LEN * in_avg / self.hist_max
        out_guide = DashedLine([left, oy, 0], [right, oy, 0], stroke_width=1.5,
                               color=PROPOSE, dash_length=0.06)
        in_guide = DashedLine([ix, top, 0], [ix, bottom, 0], stroke_width=1.5,
                              color=PROPOSE, dash_length=0.06)
        out_guide_lbl = Text(f"target avg {out_avg:g}", font_size=14, color=PROPOSE)
        out_guide_lbl.next_to(out_guide, RIGHT, buff=0.08)
        in_guide_lbl = Text(f"target avg {in_avg:g}", font_size=14, color=PROPOSE)
        in_guide_lbl.next_to(in_guide, UP, buff=0.08)

        out_title = Text("synapses out (per pre neuron)", font_size=16, color=EDGE)
        out_title.next_to(out_axis, DOWN, buff=HIST_LEN + 0.08)
        in_title = Text("synapses in (per post neuron)", font_size=16, color=OK)
        in_title.rotate(-PI / 2).next_to(in_axis, RIGHT, buff=HIST_LEN + 0.15)
        self.bottom_title = out_title

        return VGroup(out_axis, in_axis, out_guide, in_guide, out_guide_lbl,
                      in_guide_lbl, out_title, in_title, self.out_bars, self.in_bars)

    def out_bar(self, r, v):
        h = max(1e-3, HIST_LEN * v / self.hist_max)
        bar = Rectangle(width=CELL * 0.75, height=h, stroke_width=0,
                        fill_color=EDGE, fill_opacity=0.9)
        return bar.move_to([self.cell_center(r, 0)[0], self.out_y0 - h / 2, 0])

    def in_bar(self, c, v):
        w = max(1e-3, HIST_LEN * v / self.hist_max)
        bar = Rectangle(width=w, height=CELL * 0.75, stroke_width=0,
                        fill_color=OK, fill_opacity=0.9)
        return bar.move_to([self.in_x0 + w / 2, self.cell_center(0, c)[1], 0])

    def update_hists(self, pre, post):
        """Recount unique synapses on the touched columns/rows; return bar anims."""
        anims = []
        for r in set(pre):
            v = sum(1 for c in range(POST_N) if self.counts[r][c] > 0)
            if v != self.out_deg[r]:
                self.out_deg[r] = v
                anims.append(Transform(self.out_bars[r], self.out_bar(r, v)))
        for c in set(post):
            v = sum(1 for r in range(PRE_N) if self.counts[r][c] > 0)
            if v != self.in_deg[c]:
                self.in_deg[c] = v
                anims.append(Transform(self.in_bars[c], self.in_bar(c, v)))
        return anims

    # ------------------------------------------------------------ panel text
    def legend_item(self, color, label):
        sq = Square(0.22, stroke_width=0, fill_color=color, fill_opacity=1)
        return VGroup(sq, Text(label, font_size=18, color=MUTED)).arrange(RIGHT, buff=0.12)

    def kernel_text(self, K):
        return Text(f"Kernel: {K[0]} × {K[1]}  ({K[0] * K[1]} synapses per block)",
                    font_size=22)

    def thr_text(self, thr):
        return Text(f"Accept if new / block size ≥ {thr:.2f}", font_size=22)

    def blocks_text(self, n):
        return Text(f"Blocks placed: {n}", font_size=22)

    def prog_text(self, n):
        return Text(f"Unique synapses: {n} / {self.res.target}", font_size=22)

    def swap(self, attr, new):
        """Return an animation replacing a panel line in place."""
        old = getattr(self, attr)
        new.move_to(old, aligned_edge=LEFT)
        setattr(self, attr, new)
        return ReplacementTransform(old, new)

    # --------------------------------------------------------------- readout
    def proposal_lines(self, ev):
        idx = lambda xs: "{" + ", ".join(map(str, xs)) + "}"
        return [
            Text("Random proposal", font_size=22, color=PROPOSE, weight=BOLD),
            Text(f"pre  {idx(ev.pre)}", font_size=20),
            Text(f"post {idx(ev.post)}", font_size=20),
        ]

    def verdict_lines(self, ev, narrate):
        cmp = "≥" if ev.accepted else "<"
        lines = [
            Text(f"new synapses: {ev.new} / {ev.size} = {ev.frac:.2f} {cmp} {ev.threshold:.2f}",
                 font_size=22),
            Text("ACCEPT" if ev.accepted else "REJECT", font_size=26, weight=BOLD,
                 color=OK if ev.accepted else BAD),
        ]
        if narrate and ev.new < ev.size:
            lines.append(Text(f"{ev.size - ev.new} synapse(s) already placed",
                              font_size=18, color=MUTED))
        return lines

    def stall_lines(self, ev):
        action = (f"→ shrink kernel to fallback_K: {ev.K[0]} × {ev.K[1]}"
                  if ev.kind == "fallback" else
                  f"→ relax threshold: × decay ({DECAY}) = {ev.threshold:.2f}")
        return [Text(f"Stalled: {MAX_TRIES} rejections in a row", font_size=24,
                     color=BAD, weight=BOLD),
                Text(action, font_size=22)]

    def done_lines(self, ev):
        n_dup = int(np.maximum(self.res.matrix - 1, 0).sum())
        sizes = {}
        for pre, post in self.res.mappings:
            key = f"{len(pre)}×{len(post)}"
            sizes[key] = sizes.get(key, 0) + 1
        mix = ", ".join(f"{n} of {k}" for k, n in sizes.items())
        return [
            Text("Target reached", font_size=26, color=OK, weight=BOLD),
            Text(f"{ev.progress} / {self.res.target} unique synapses", font_size=22),
            Text(f"{ev.blocks} blocks: {mix}", font_size=22),
            Text(f"{n_dup} duplicate placement(s)", font_size=22, color=OVERLAP),
        ]

    def readout_lines(self, i):
        """The readout the display shows once event i has finished."""
        ev = self.events[i]
        if ev.kind == "try":
            return self.proposal_lines(ev) + self.verdict_lines(ev, self.narrate[i])
        if ev.kind in ("fallback", "decay"):
            return self.stall_lines(ev)
        return self.done_lines(ev)

    def place_readout(self, lines):
        new = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        for m in new:
            fit(m)
        return new.move_to(self.readout_anchor, aligned_edge=UL)

    def set_readout(self, lines, rt=0.4, append=False):
        if append:
            last = self.readout[-1]
            for m in lines:
                fit(m)
                m.next_to(last, DOWN, buff=0.14, aligned_edge=LEFT)
                last = m
            self.play(*[FadeIn(m) for m in lines], run_time=rt)
            self.readout.add(*lines)
            return
        new = self.place_readout(lines)
        # fade lines individually: appended lines were added to the scene
        # on their own, so fading only the parent group would leave them behind
        self.play(FadeIn(new), *[FadeOut(m) for m in self.readout], run_time=rt)
        self.readout = new

    def set_dots(self, n):
        return AnimationGroup(*[d.animate.set_fill(opacity=1 if i < n else 0)
                                for i, d in enumerate(self.dots)])

    # ----------------------------------------------------------------- replay
    def play_events(self, a, b, fast=FAST):
        for i in range(a, b):
            ev = self.events[i]
            if ev.kind == "try":
                narrate = self.narrate[i]
                self.show_try(ev, 1.0 if narrate else fast, narrate)
            elif ev.kind in ("fallback", "decay"):
                self.show_stall(ev, i)
            else:
                self.show_done(ev)

    def show_try(self, ev, f, narrate):
        pre_bands = VGroup(*[Rectangle(width=CELL, height=self.H, stroke_width=0,
                                       fill_color=PROPOSE, fill_opacity=0.13)
                             .move_to(self.cell_center(r, (POST_N - 1) / 2))
                             for r in ev.pre])
        post_bands = VGroup(*[Rectangle(width=self.W, height=CELL, stroke_width=0,
                                        fill_color=PROPOSE, fill_opacity=0.13)
                              .move_to(self.cell_center((PRE_N - 1) / 2, c))
                              for c in ev.post])
        pairs = [(r, c) for r in ev.pre for c in ev.post]
        outlines = VGroup(*[Square(CELL * 0.86, stroke_color=PROPOSE, stroke_width=2)
                            .move_to(self.cell_center(r, c)) for r, c in pairs])

        self.set_readout(self.proposal_lines(ev), rt=0.3 * f)
        self.play(FadeIn(pre_bands), FadeIn(post_bands), run_time=0.5 * f)
        self.play(Create(outlines), run_time=0.5 * f)

        # flag cells that already hold a synapse
        dup = VGroup(*[o for o, (r, c) in zip(outlines, pairs) if self.counts[r][c] > 0])
        if len(dup):
            self.play(dup.animate.set_stroke(BAD), run_time=0.35 * f)

        self.set_readout(self.verdict_lines(ev, narrate), rt=0.35 * f, append=True)
        if narrate:
            self.wait(0.8)

        if ev.accepted:
            fills = []
            for r, c in pairs:
                self.counts[r][c] += 1
                color = EDGE if self.counts[r][c] == 1 else OVERLAP
                fills.append(self.cells[r][c].animate.set_fill(color))
            self.play(*fills, *self.update_hists(ev.pre, ev.post),
                      outlines.animate.set_stroke(OK),
                      self.progress.animate.set_value(ev.progress),
                      self.swap("prog_lbl", self.prog_text(ev.progress)),
                      self.swap("stat_blocks", self.blocks_text(ev.blocks)),
                      self.set_dots(0), run_time=0.6 * f)
        else:
            self.play(outlines.animate.set_stroke(BAD), self.set_dots(ev.tries),
                      Wiggle(outlines, scale_value=1.05, rotation_angle=0.02 * TAU),
                      run_time=0.6 * f)
        self.play(FadeOut(pre_bands), FadeOut(post_bands), FadeOut(outlines),
                  run_time=0.3 * f)

    def show_stall(self, ev, i):
        if ev.kind == "fallback":
            swap_anim = self.swap("stat_kernel", self.kernel_text(ev.K))
            attr = "stat_kernel"
        else:
            swap_anim = self.swap("stat_thr", self.thr_text(ev.threshold))
            attr = "stat_thr"
        self.set_readout(self.stall_lines(ev), rt=0.5)
        self.play(swap_anim, self.set_dots(0), run_time=0.8)
        self.play(Indicate(getattr(self, attr), color=PROPOSE, scale_factor=1.08))
        if ev.kind == "fallback":
            self.morph_kernel(self.events[i - 1].K, ev.K)
        self.wait(1.2)

    def kernel_strip(self, K, color):
        k = VGroup(*[Square(CELL, stroke_color=GRID, stroke_width=1,
                            fill_color=color, fill_opacity=1)
                     for _ in range(K[0] * K[1])])
        return k.arrange_in_grid(K[1], K[0], buff=0)

    def morph_kernel(self, old_K, new_K):
        """Show the block outline shrinking from old_K to new_K."""
        y = self.readout.get_bottom()[1] - 0.6
        old = self.kernel_strip(old_K, EDGE)
        old.move_to([PANEL_X + old.width / 2, y, 0])
        new = self.kernel_strip(new_K, PROPOSE)
        new.move_to([PANEL_X + new.width / 2, y, 0])
        self.play(FadeIn(old))
        self.play(ReplacementTransform(old, new), run_time=1.0)
        self.wait(0.8)
        self.play(FadeOut(new))

    def show_done(self, ev):
        self.set_readout(self.done_lines(ev), rt=0.6)
        self.play(Indicate(self.prog_lbl, color=OK))
        self.wait(2)

    # ------------------------------------------------------ pulled-out blocks
    def panel_items(self):
        return [m for m in self.mobjects if m.get_center()[0] > PANEL_X - 0.2]

    def make_kernels(self):
        """One dense kernel per accepted block (pre columns x post rows),
        laid out in one stack per block shape."""
        maps = self.res.mappings
        kernels = [self.kernel_strip((len(pre), len(post)), EDGE) for pre, post in maps]
        shapes = sorted({(len(p), len(q)) for p, q in maps}, reverse=True)
        stacks = []
        for s in shapes:
            stack = VGroup(*[k for k, (p, q) in zip(kernels, maps) if (len(p), len(q)) == s])
            stacks.append(stack.arrange(DOWN, buff=0.05, aligned_edge=LEFT))
        layout = VGroup(*stacks).arrange(RIGHT, buff=0.5, aligned_edge=UP)
        layout.move_to(RIGHT * 3.9 + DOWN * 0.2)
        cap = fit(Text("Each accepted block is emitted as one dense kernel", font_size=22), 5.8)
        cap.move_to(RIGHT * 3.7 + UP * 2.85)
        self.kernels, self.kernel_layout, self.caption = kernels, layout, cap

    def pull_out_blocks(self):
        self.play(*[FadeOut(m) for m in self.panel_items()], run_time=0.6)
        self.make_kernels()
        self.play(FadeIn(self.caption))
        moves = [ReplacementTransform(
                     VGroup(*[self.cells[r][c].copy() for c in post for r in pre]), k)
                 for k, (pre, post) in zip(self.kernels, self.res.mappings)]
        self.play(LaggedStart(*moves, lag_ratio=0.12), run_time=4)
        # the transforms leave each kernel in the scene on its own; hand them
        # back to their layout so pack_tile can keep the whole group
        self.remove(*self.kernels)
        self.add(self.kernel_layout)
        self.wait(1.5)

    def start_pulled_out(self):
        """Instantly show the frame Clip7 ends on."""
        self.build_workspace(animate=False, upto=len(self.events))
        self.remove(*self.panel_items())
        self.make_kernels()
        self.add(self.caption, self.kernel_layout)

    # ------------------------------------------------- stand-alone sections
    def clear(self, rt=0.8):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=rt)

    def block_intro(self):
        """Clip 1: pre/post neurons drawn as one block of connections."""
        cap = Text("A crossbar kernel wires a set of pre-synaptic neurons to a set "
                   "of post-synaptic neurons, all-to-all", font_size=26)
        cap.to_edge(UP, buff=0.5)
        fit(cap, 12.5)

        sp = min(0.45, 4.8 / max(PRE_K, POST_K))
        fan_x, fan_y = -3.4, -0.1
        pre = VGroup(*[Dot(radius=0.13, color=EDGE) for _ in range(PRE_K)])
        pre.arrange(RIGHT, buff=sp - 0.26).move_to([fan_x, fan_y + 1.6, 0])
        post = VGroup(*[Dot(radius=0.13, color=OK) for _ in range(POST_K)])
        post.arrange(RIGHT, buff=sp - 0.26).move_to([fan_x, fan_y - 1.5, 0])
        pre_lbl = Text(f"pre_K = {PRE_K}", font_size=22, color=EDGE).next_to(pre, UP, buff=0.3)
        post_lbl = Text(f"post_K = {POST_K}", font_size=22, color=OK).next_to(post, RIGHT, buff=0.35)
        # post-major so the lines pair up with the block's row-major squares
        lines = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=2, color=MUTED)
                         for b in post for a in pre])

        self.play(FadeIn(cap))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in [*pre, *post]], lag_ratio=0.08),
                  FadeIn(pre_lbl), FadeIn(post_lbl))
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.04), run_time=1.5)

        side = min(0.4, 4.8 / max(PRE_K, POST_K))
        block = VGroup(*[Square(side, stroke_color=GRID, stroke_width=1.5,
                                fill_color=EDGE, fill_opacity=1)
                         for _ in range(PRE_K * POST_K)])
        block.arrange_in_grid(POST_K, PRE_K, buff=0).move_to([3.6, fan_y, 0])
        arrow = Arrow([-0.5, fan_y, 0], [0.9, fan_y, 0], color=MUTED)
        col_lbl = Text("columns: pre-synaptic (inputs)", font_size=20, color=EDGE)
        col_lbl.next_to(block, UP, buff=0.3)
        row_lbl = Text("rows: post-synaptic (outputs)", font_size=20, color=OK)
        blk_lbl = Text(f"one {PRE_K} × {POST_K} block = {PRE_K * POST_K} synapses",
                       font_size=22)
        VGroup(row_lbl, blk_lbl).arrange(DOWN, buff=0.2).next_to(block, DOWN, buff=0.4)

        self.play(GrowArrow(arrow), ReplacementTransform(lines.copy(), block))
        self.play(FadeIn(col_lbl), FadeIn(row_lbl), FadeIn(blk_lbl))
        self.wait(1.5)

        goal = VGroup(
            Text("Goal: reach a target connection density by stacking blocks,",
                 font_size=26),
            Text("while avoiding duplicate synapses", font_size=26),
        ).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.5)
        self.play(FadeOut(cap), FadeIn(goal, shift=UP * 0.2))
        self.wait(2)

    def pack_tile(self):
        """Clip 8: from the pulled-out kernels, pack them into one tile."""
        keep = (self.caption, self.kernel_layout)
        self.play(*[FadeOut(m) for m in self.mobjects
                    if all(m is not k for k in keep)], run_time=0.8)

        tile_cells, cell_pos = self.build_tile(LEFT * 3.4 + DOWN * 0.35)
        tile = VGroup(*[sq for row in tile_cells for sq in row])
        title = Text(f"One crossbar tile: {TILE_SIZE} × {TILE_SIZE} devices",
                     font_size=26, weight=BOLD).next_to(tile, UP, buff=0.55)
        cols = Text("columns: pre-synaptic inputs", font_size=18, color=EDGE)
        cols.next_to(tile, UP, buff=0.12)
        rows = Text("rows: post-synaptic outputs", font_size=18, color=OK)
        rows.rotate(PI / 2).next_to(tile, LEFT, buff=0.12)
        cap = fit(Text("Blocks are packed into the tile", font_size=24), 6.2)
        cap.move_to(self.caption)

        self.play(FadeOut(self.caption), FadeIn(cap),
                  LaggedStart(*[FadeIn(sq) for sq in tile], lag_ratio=0.002, run_time=1.2),
                  FadeIn(title), FadeIn(cols), FadeIn(rows))
        self.bring_to_front(self.kernel_layout)

        shapes = [(len(pre), len(post)) for pre, post in self.res.mappings]
        slots, used = first_fit(shapes)
        moves = [k.animate.move_to(cell_pos(i + (h - 1) / 2, j + (w - 1) / 2))
                 for k, (w, h), (i, j) in zip(self.kernels, shapes, slots)]
        self.play(LaggedStart(*moves, lag_ratio=0.1), run_time=5)

        outlines = VGroup(*[
            Rectangle(width=w * CELL, height=h * CELL, stroke_color=BLACK,
                     stroke_width=1.5, fill_opacity=0)
            .move_to(cell_pos(i + (h - 1) / 2, j + (w - 1) / 2))
            for (w, h), (i, j) in zip(shapes, slots)
        ])
        self.play(LaggedStart(*[Create(o) for o in outlines], lag_ratio=0.06), run_time=1.5)

        n_used = int(used.sum())
        total = TILE_SIZE * TILE_SIZE
        sizes = {}
        for w, h in shapes:
            sizes[f"{w}×{h}"] = sizes.get(f"{w}×{h}", 0) + 1
        mix = ", ".join(f"{n} of {k}" for k, n in sizes.items())
        stats = VGroup(
            Text(f"Blocks packed: {len(shapes)}", font_size=24),
            Text(mix, font_size=20, color=MUTED),
            Text(f"Devices used: {n_used} / {total}  ({100 * n_used / total:.1f}%)",
                 font_size=24, color=OK),
            Text(f"{total - n_used} devices left idle", font_size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 3.6 + UP * 0.4)
        self.play(FadeIn(stats, shift=LEFT * 0.2))
        self.wait(2.5)

    def read_block(self):
        shapes = [(len(pre), len(post)) for pre, post in self.res.mappings]
        slots, used = first_fit(shapes)
        # prefer a wide block (more spikes to show), then the one with the
        # most tile around it on every side
        margin = lambda k: min(slots[k][1], slots[k][0],
                               TILE_SIZE - slots[k][0] - shapes[k][1],
                               TILE_SIZE - slots[k][1] - shapes[k][0])
        target = max(range(len(shapes)), key=lambda k: (shapes[k][0], margin(k)))
        pre_idx, post_idx = self.res.mappings[target]
        w, h = shapes[target]           # h is always 1: post_K = 1 for every block
        i, j = slots[target]

        title = Text("Reading one block inside the tile: spikes → voltages → current",
                     font_size=25, weight=BOLD).to_edge(UP, buff=0.3)
        sub = Text("every other row and column is disabled for this read", font_size=17,
                   color=MUTED).next_to(title, DOWN, buff=0.1)

        tile_cells, cell_pos = self.build_tile(LEFT * 0.9 + DOWN * 0.55)
        grid = VGroup(*[sq for row in tile_cells for sq in row])
        for r in range(TILE_SIZE):
            for c in range(TILE_SIZE):
                if not (r == i and j <= c < j + w):
                    tile_cells[r][c].set_stroke(opacity=0.3)
        devs = VGroup(*[tile_cells[i][c] for c in range(j, j + w)])
        devs.set_fill(EDGE, opacity=1)
        block_outline = Rectangle(width=w * CELL, height=CELL, stroke_color=PROPOSE,
                                  stroke_width=2.5).move_to(cell_pos(i, j + (w - 1) / 2))

        top_y = cell_pos(0, 0)[1] + CELL / 2 + 0.6
        right_x = cell_pos(0, TILE_SIZE - 1)[0] + CELL / 2 + 0.35
        xs = [cell_pos(i, c)[0] for c in range(j, j + w)]
        row_y = cell_pos(i, 0)[1]

        col_dots = VGroup(*[Dot(radius=0.045, color=MUTED).move_to([cell_pos(0, c)[0], top_y, 0])
                            for c in range(TILE_SIZE)])
        for c in range(j, j + w):
            col_dots[c].set_color(EDGE).scale(1.4)
        row_dots = VGroup(*[Dot(radius=0.045, color=MUTED).move_to([right_x, cell_pos(r, 0)[1], 0])
                            for r in range(TILE_SIZE)])
        row_dots[i].set_color(OK).scale(1.4)

        cols_lbl = Text("columns: pre-synaptic inputs", font_size=17, color=EDGE)
        cols_lbl.next_to(col_dots, UP, buff=0.32)

        wires = VGroup(*[Line([x, top_y - 0.06, 0], [x, row_y + CELL / 2, 0],
                              stroke_width=2.5, color=WIRE) for x in xs])
        row_wire = Line([xs[0], row_y, 0], [right_x, row_y, 0], stroke_width=2.5, color=WIRE)

        idx_lbls = VGroup(*[Text(str(p), font_size=11, color=MUTED)
                            .next_to(col_dots[c], UP, buff=0.06 if a % 2 == 0 else 0.19)
                            for a, (c, p) in enumerate(zip(range(j, j + w), pre_idx))])
        post_lbl = Text(f"post-synaptic neuron {post_idx[0]}\n(row {i})", font_size=15,
                        color=OK, line_spacing=0.9)
        post_lbl.next_to(row_dots[i], RIGHT, buff=0.2)

        # --- just the mechanism: active columns' contributions travel down
        # their wires and sum into the row's current. No plot, no side panel.
        self.play(FadeIn(title), FadeIn(sub))
        self.play(LaggedStart(*[FadeIn(sq) for row in tile_cells for sq in row],
                              lag_ratio=0.0008, run_time=1.2))
        self.play(Create(block_outline), FadeIn(cols_lbl))
        self.play(FadeIn(col_dots), FadeIn(row_dots), FadeIn(idx_lbls), FadeIn(post_lbl))
        self.play(Create(wires), Create(row_wire), run_time=1.0)
        self.wait(0.5)

        rng = np.random.default_rng(7)
        end = np.array([right_x, row_y, 0])
        for k in [max(1, round(w * 0.25)), max(1, round(w * 0.6)), max(1, round(w * 0.9))]:
            active = sorted(rng.choice(w, k, replace=False).tolist())

            # spikes set the active columns to the read voltage
            self.play(*[col_dots[j + a].animate.set_color(PROPOSE) for a in active],
                      run_time=0.5)
            self.play(*[wires[a].animate.set_color(PROPOSE) for a in active],
                      *[devs[a].animate.set_fill(PROPOSE) for a in active], run_time=0.6)

            # each active device sends its contribution down onto the row,
            # where they all sum into a single current
            carriers = VGroup(*[Dot(radius=0.06, color=PROPOSE).move_to(devs[a].get_center())
                                for a in active])
            self.play(FadeIn(carriers), run_time=0.3)
            self.play(LaggedStart(*[c.animate.move_to(end) for c in carriers],
                                  lag_ratio=0.12), run_time=1.2)
            self.play(FadeOut(carriers), Indicate(row_dots[i], color=OK, scale_factor=1.4),
                      run_time=0.3)

            sum_lbl = Text(f"I_post = {k}", font_size=22, color=OK)
            sum_lbl.next_to(post_lbl, DOWN, buff=0.3, aligned_edge=LEFT)
            self.play(FadeIn(sum_lbl, scale=1.15), run_time=0.4)
            self.wait(1.0)

            self.play(*[col_dots[j + a].animate.set_color(EDGE) for a in active],
                      *[wires[a].animate.set_color(WIRE) for a in active],
                      *[devs[a].animate.set_fill(EDGE) for a in active],
                      FadeOut(sum_lbl), run_time=0.5)
        self.wait(1)


# ============================================================== clip 1
class Clip1(GreedyBase):
    """Pre-synaptic -> post-synaptic neurons drawn as a block of connections."""

    def construct(self):
        self.block_intro()


# ============================================================== clips 2-7
class Clip2(GreedyBase):
    """Build the matrix and parameter display; first block placement."""

    def construct(self):
        a, b = self.marks["c2"]
        self.build_workspace(animate=True)
        self.play_events(a, b)
        self.wait(1.5)


class _ReplayClip(GreedyBase):
    mark = None
    fast = 0.5

    def construct(self):
        a, b = self.marks[self.mark]
        self.build_workspace(animate=False, upto=a)
        self.wait(0.5)
        self.play_events(a, b, fast=self.fast)
        self.after()
        self.wait(1.5)

    def after(self):
        pass


class Clip3(_ReplayClip):
    """More placements, stopping at the first rejected block."""
    mark = "c3"


class Clip4(_ReplayClip):
    """More rejections, until a block is finally accepted."""
    mark = "c4"


class Clip5(_ReplayClip):
    """Continue until the 6th rejection in a row."""
    mark = "c5"


class Clip6(_ReplayClip):
    """The failure budget is exhausted: the kernel shrinks."""
    mark = "c6"


class Clip7(_ReplayClip):
    """Remaining smaller blocks, then the final blocks pulled out."""
    mark = "c7"
    fast = FAST

    def after(self):
        self.pull_out_blocks()


# ============================================================== clip 8
class Clip8(GreedyBase):
    """Pack the chosen blocks into one 25 x 25 crossbar tile."""

    def construct(self):
        self.start_pulled_out()
        self.wait(0.5)
        self.pack_tile()


# ============================================================== clip 9
class Clip9(GreedyBase):
    """Spikes set the input columns to read voltages; the block's devices
    turn those voltages into a current summed on the post-synaptic row.
    Shown at the block's real position inside the packed 25 x 25 tile: every
    row and column outside the block is greyed out as disabled for this read."""

    def construct(self):
        self.read_block()


# ============================================================== full run
class GreedyFull(GreedyBase):
    """Clips 1-8 as one continuous video (about a minute). The intro, build,
    pull-out and packing play at normal speed; the random sampling of clips
    2-7 runs straight through, accelerating from REPLAY_PACE_START on the
    first proposal to REPLAY_PACE_END on the last. Clip 9's read is left out."""

    PACE = {"c2": FAST, "c3": 0.5, "c4": 0.5, "c5": 0.5, "c6": 0.5, "c7": FAST}
    REPLAY_PACE_START = 0.3   # run_time multiplier on the first event
    REPLAY_PACE_END = 0.03    # ... and on the last; geometric ramp in between

    def construct(self):
        self.block_intro()
        self.clear()

        self.build_workspace(animate=True)
        n = len(self.events) - 1
        ratio = self.REPLAY_PACE_END / self.REPLAY_PACE_START
        for mark, fast in self.PACE.items():
            a, b = self.marks[mark]
            for i in range(a, b):
                self.pace = self.REPLAY_PACE_START * ratio ** (i / n)
                self.play_events(i, i + 1, fast=fast)
        self.pace = 1.0
        self.wait(1.0)
        self.pull_out_blocks()

        self.wait(0.5)
        self.pack_tile()


# ============================================================== extra
class DefaultsCard(Scene):
    """Not one of the nine clips: the real defaults from generate_hw_mappings.py."""

    def construct(self):
        title = Text("In generate_hw_mappings.py", font_size=34, weight=BOLD)
        rows = [
            ("Kernel (Pyr → Pyr)", "pre_k = post_k = TILE_SIZE = 25"),
            ("Kernel (Pyr → NonPyr)", "post_K = 1"),
            ("Kernel (NonPyr → Pyr)", "inh_exc_pre_k = 1, inh_exc_post_k = 25"),
            ("Fallback kernel", "fallback_k = 5  (or after --blocks-to-fallback)"),
            ("Acceptance threshold", "min_accept_frac = 0.98, × decay 0.995 per stall"),
            ("Stall", "max_tries = 5000 rejections"),
            ("Dense connections", "density ≥ 0.97 → full_connection_generator"),
            ("Progress counts", "unique synapses (--include-overlap counts all)"),
        ]
        table = VGroup(*[
            VGroup(Text(k, font_size=21, color=PROPOSE), Text(v, font_size=21))
            for k, v in rows
        ])
        for row in table:
            row[1].next_to(row[0], RIGHT, buff=0)
        table.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        keys_w = max(r[0].width for r in table)
        for row in table:
            row[1].next_to(row[0], RIGHT, buff=0).align_to(
                row[0].get_left() + RIGHT * (keys_w + 0.5), LEFT)
        body = VGroup(title, table).arrange(DOWN, buff=0.5)
        fit(body, 12.8)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in table], lag_ratio=0.15))
        self.wait(4)
