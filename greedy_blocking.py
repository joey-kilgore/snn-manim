"""
Reference re-implementation of the greedy block placement used by
generate_hw_mappings.py (greedy_connection_generator), instrumented to emit an
event log that the Manim scene replays.

Reconstructed from the CLI help / call site in map.py, since
connection_generator.py is not in this repo:

  * target      = round(target_density * pre_N * post_N) synapses
  * a block     = pre_K random pre neurons x post_K random post neurons,
                  wired all-to-all (one dense crossbar kernel)
  * accept      if (new, non-duplicate edges / block size) >= threshold
  * stall       = max_tries proposals in a row rejected. On a stall the kernel
                  shrinks to fallback_K (first time), otherwise the threshold
                  is multiplied by `decay`
  * blocks_to_fallback forces the shrink after N accepted blocks
  * progress counts unique edges (default) or every placed edge
                  (include_overlap=True)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    kind: str                     # "try" | "fallback" | "decay" | "done"
    pre: tuple = ()
    post: tuple = ()
    new: int = 0
    size: int = 0
    frac: float = 0.0
    threshold: float = 0.0
    accepted: bool = False
    progress: int = 0             # counted edges after this event
    tries: int = 0                # consecutive rejections so far
    K: tuple = (0, 0)
    blocks: int = 0


@dataclass
class GreedyResult:
    target: int
    matrix: np.ndarray
    mappings: list = field(default_factory=list)
    events: list = field(default_factory=list)


def greedy_blocking(pre_N, post_N, pre_K, post_K, target_density,
                    min_accept_frac=0.98, decay=0.995, max_tries=5000,
                    fallback_K=None, include_overlap=False,
                    blocks_to_fallback=None, seed=None):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((pre_N, post_N), dtype=int)
    target = round(target_density * pre_N * post_N)
    threshold = min_accept_frac
    K = (min(pre_K, pre_N), min(post_K, post_N))
    fell_back = False
    progress = 0
    tries = 0
    res = GreedyResult(target=target, matrix=matrix)

    def shrink():
        nonlocal K, fell_back
        K = (min(K[0], fallback_K), min(K[1], fallback_K))
        fell_back = True
        res.events.append(Event("fallback", K=K, threshold=threshold,
                                progress=progress, blocks=len(res.mappings)))

    while progress < target:
        pre = tuple(sorted(rng.choice(pre_N, K[0], replace=False).tolist()))
        post = tuple(sorted(rng.choice(post_N, K[1], replace=False).tolist()))
        sub = matrix[np.ix_(pre, post)]
        size = sub.size
        new = int((sub == 0).sum())
        frac = new / size
        accepted = frac >= threshold
        if accepted:
            matrix[np.ix_(pre, post)] += 1
            progress += size if include_overlap else new
            res.mappings.append((pre, post))
            tries = 0
        else:
            tries += 1
        res.events.append(Event("try", pre, post, new, size, frac, threshold,
                                accepted, progress, tries, K, len(res.mappings)))

        if accepted and blocks_to_fallback and fallback_K and not fell_back \
                and len(res.mappings) >= blocks_to_fallback:
            shrink()
        elif tries >= max_tries:
            tries = 0
            if fallback_K and not fell_back and max(K) > fallback_K:
                shrink()
            else:
                threshold *= decay
                res.events.append(Event("decay", K=K, threshold=threshold,
                                        progress=progress,
                                        blocks=len(res.mappings)))

    res.events.append(Event("done", K=K, threshold=threshold,
                            progress=progress, blocks=len(res.mappings)))
    return res


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    r = greedy_blocking(25, 25, 12, 1, 0.45, min_accept_frac=1.0, decay=0.75,
                        max_tries=6, fallback_K=5, seed=seed)
    print(f"target={r.target} blocks={len(r.mappings)} events={len(r.events)}")
    for e in r.events:
        if e.kind == "try":
            print(f"  try K={e.K} new={e.new}/{e.size} frac={e.frac:.2f} "
                  f"thr={e.threshold:.2f} {'ACCEPT' if e.accepted else 'reject'} "
                  f"prog={e.progress}")
        else:
            print(f"  {e.kind.upper()} K={e.K} thr={e.threshold:.3f}")
    print(r.matrix)
