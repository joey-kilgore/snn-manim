"""
generate_hw_mappings.py

Loads a CARLsim network .dat file, computes connection probabilities for every
group pair, then generates Daffodil hardware mapping CSV files for every
connection that involves at least one Pyramidal cell group (identified by
'pyr' appearing in the group name, case-insensitive).

Greedy block placement is used for sparse connections; full connection
generation is used for dense connections (density >= FULL_CONNECT_THRESHOLD).

Output is written to:
    data/<network_stem>_hw_mappings_seed<seed>/
        <pre>_<post>_mapping.csv   — one file per Pyramidal connection
        mapping_notes.txt           — stats and notes for all connections
"""

import sys
import os
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np

# Allow importing from the scripts directory when run directly
sys.path.insert(0, os.path.dirname(__file__))
from simulation import Simulation
from connection_generator import (
    greedy_connection_generator,
    full_connection_generator,
    exact_connection_generator,
    grouped_exact_connection_generator,
    format_mappings_to_csv,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TILE_SIZE = 25                  # Daffodil crossbar dimension
FALLBACK_TILE_SIZE = 5           # smaller tile size for fallback when greedy fails
FULL_CONNECT_THRESHOLD = 0.97   # density at/above which full connect is used
DEFAULT_MIN_ACCEPT_FRAC = 0.98
DEFAULT_DECAY = 0.995
DEFAULT_MAX_TRIES = 5000
DEFAULT_SEED = 17


def _group_size(group):
    return group.sizeX * group.sizeY * group.sizeZ


def _is_pyramidal(group):
    return "pyr" in group.groupNameClean.lower()


def _safe_name(group_name):
    """Strip characters unsafe for filenames."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in group_name)


def _kernel_sizes(pre_is_pyr, post_is_pyr, pre_N, pre_k=TILE_SIZE, post_k=TILE_SIZE,
                  inh_exc_pre_k=1, inh_exc_post_k=TILE_SIZE):
    """
    Choose starting kernel sizes based on group type.

    Pyramidal groups use pre_k / post_k (defaulting to TILE_SIZE).
    For NonPyr->Pyr (inhibitory->excitatory) connections, inh_exc_pre_k and
    inh_exc_post_k control both dimensions independently, avoiding collapse
    to 1x1 when pre_k/post_k are set small for Pyr->Pyr connections.
    inh_exc_pre_k is clamped to pre_N so it never exceeds the actual
    inhibitory population size. Pyr->NonPyr post dimension is always 1.
    """
    if pre_is_pyr:
        pre_K = pre_k
    else:
        pre_K = min(inh_exc_pre_k, pre_N) if post_is_pyr else 1  # NonPyr->Pyr vs NonPyr->NonPyr

    if not post_is_pyr:
        post_K = 1
    elif not pre_is_pyr:
        post_K = inh_exc_post_k  # NonPyr->Pyr
    else:
        post_K = post_k           # Pyr->Pyr
    return pre_K, post_K


def _fallback_K(pre_K, post_K, fallback_tile_size=FALLBACK_TILE_SIZE):
    """
    Return fallback_tile_size when at least one kernel dimension is larger
    than the fallback (i.e. there is room to shrink). Returns None when both
    dimensions are already at or below the fallback — the min() in the stall
    logic would be a no-op, so no fallback is needed.
    """
    if pre_K > fallback_tile_size or post_K > fallback_tile_size:
        return fallback_tile_size
    return None


def _write_tile_summary(all_used, notes_lines, total_blocks=0, tile_size=TILE_SIZE):
    total_cells = tile_size * tile_size
    total_tiles = len(all_used)
    total_used = sum(int(u.sum()) for u in all_used)
    total_available = total_tiles * total_cells
    total_unused = total_available - total_used
    pct = 100 * total_used / total_available if total_available > 0 else 0.0

    notes_lines.append("\n" + "=" * 70)
    notes_lines.append("--- Tile Usage Summary ---")
    notes_lines.append(f"  Total blocks emitted : {total_blocks}")
    for k, used in enumerate(all_used):
        used_k = int(used.sum())
        unused_k = total_cells - used_k
        pct_k = 100 * used_k / total_cells
        notes_lines.append(
            f"  Tile {k:>3} : {used_k:>4} / {total_cells} used  "
            f"({unused_k:>4} unused,  {pct_k:5.1f}% utilized)"
        )
    notes_lines.append(
        f"  Total   : {total_tiles} tile(s),  "
        f"{total_used} / {total_available} devices used  "
        f"({total_unused} unused,  {pct:.1f}% utilized)"
    )


def _write_notes_header(network_file, seed, extra_label=""):
    lines = []
    lines.append("=" * 70)
    lines.append(f"Daffodil Hardware Mapping Notes  {extra_label}".strip())
    lines.append(f"Generated : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Network   : {network_file}")
    lines.append(f"Seed      : {seed}")
    lines.append(f"Tile size : {TILE_SIZE}")
    return lines


def _write_group_summary(sim, group_sizes, conn_counts, conn_densities, notes_lines):
    notes_lines.append("\n--- Groups ---")
    for i, g in enumerate(sim.groups):
        notes_lines.append(
            f"  [{i}] {g.groupNameClean}  "
            f"({g.sizeX} x {g.sizeY} x {g.sizeZ} = {group_sizes[i]} neurons)"
            + ("  [PYRAMIDAL]" if _is_pyramidal(g) else "")
        )
    notes_lines.append("\n--- All connection densities ---")
    for i, gPre in enumerate(sim.groups):
        for j, gPost in enumerate(sim.groups):
            d = conn_densities[(i, j)]
            if d > 0:
                notes_lines.append(
                    f"  {gPre.groupNameClean} -> {gPost.groupNameClean}: {d:.6f}"
                    f"  ({conn_counts[(i,j)]} synapses,"
                    f"  {group_sizes[i]}x{group_sizes[j]})"
                )


def _run_greedy_pass(sim, pyr_connections, non_pyr_connections,
                     group_sizes, conn_counts, conn_densities,
                     conn_avg_weights, conn_max_weights,
                     out_dir, seed, network_file, include_overlap=False,
                     fallback_k=FALLBACK_TILE_SIZE, pre_k=TILE_SIZE, post_k=TILE_SIZE,
                     inh_exc_pre_k=1, inh_exc_post_k=TILE_SIZE, exact_inhibitory=False,
                     min_accept_frac=DEFAULT_MIN_ACCEPT_FRAC, decay=DEFAULT_DECAY,
                     max_tries=DEFAULT_MAX_TRIES, blocks_to_fallback=None):
    """Write greedy/full-connect CSVs and random non-Pyr .npy files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Greedy pass] Output: {out_dir}")

    notes_lines = _write_notes_header(network_file, seed, extra_label="[greedy]")
    notes_lines.append(f"Full-connect threshold: {FULL_CONNECT_THRESHOLD}")
    notes_lines.append(f"Initial kernel (pyr)  : pre_k={pre_k}  post_k={post_k}")
    notes_lines.append(f"Initial kernel (inh->exc): inh_exc_pre_k={inh_exc_pre_k}  inh_exc_post_k={inh_exc_post_k}")
    notes_lines.append(f"Fallback kernel size  : {fallback_k}")
    notes_lines.append(f"Blocks to fallback    : {blocks_to_fallback}")
    notes_lines.append(f"Include overlap in count: {include_overlap}")
    notes_lines.append("=" * 70)
    _write_group_summary(sim, group_sizes, conn_counts, conn_densities, notes_lines)
    notes_lines.append("\n--- Per-connection mapping details ---")

    np.random.seed(seed)
    all_used = []
    total_blocks = 0

    for i, j in pyr_connections:
        pre_name = sim.groups[i].groupNameClean
        post_name = sim.groups[j].groupNameClean
        pre_N = group_sizes[i]
        post_N = group_sizes[j]
        density = conn_densities[(i, j)]

        csv_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_mapping.csv"

        notes_lines.append(f"\n[{pre_name} -> {post_name}]")
        notes_lines.append(f"  Pre neurons  : {pre_N}  (group {i})")
        notes_lines.append(f"  Post neurons : {post_N}  (group {j})")
        notes_lines.append(f"  Density      : {density:.6f}")
        notes_lines.append(f"  Avg weight   : {conn_avg_weights.get((i, j), 0.0):.6f}")
        notes_lines.append(f"  Max weight   : {conn_max_weights.get((i, j), 0.0):.6f}")
        notes_lines.append(f"  Tiles allocated before: {len(all_used)}")

        if density >= FULL_CONNECT_THRESHOLD:
            notes_lines.append(f"  Generator    : full_connection_generator")
            notes_lines.append(f"  Post chunk   : {TILE_SIZE}")
            t0 = time.perf_counter()
            mappings, conn_matrix = full_connection_generator(
                pre_N=pre_N, post_N=post_N, post_chunk_size=TILE_SIZE,
            )
            t_gen = time.perf_counter() - t0
            notes_lines.append(f"  Blocks placed: {len(mappings)}")
            notes_lines.append(f"  Time (gen)   : {t_gen:.3f}s")
        else:
            pre_K, post_K = _kernel_sizes(
                _is_pyramidal(sim.groups[i]),
                _is_pyramidal(sim.groups[j]),
                pre_N=pre_N,
                pre_k=pre_k, post_k=post_k, inh_exc_pre_k=inh_exc_pre_k, inh_exc_post_k=inh_exc_post_k,
            )
            fb_K = _fallback_K(pre_K, post_K, fallback_tile_size=fallback_k)
            notes_lines.append(f"  Generator    : greedy_connection_generator")
            notes_lines.append(f"  pre_K={pre_K}  post_K={post_K}  fallback_K={fb_K}")
            notes_lines.append(
                f"  min_accept_frac={min_accept_frac}  "
                f"decay={decay}  max_tries={max_tries}"
            )
            notes_lines.append(f"  include_overlap={include_overlap}")
            print(f"  Generating: {pre_name} -> {post_name}")
            t0 = time.perf_counter()
            mappings, conn_matrix = greedy_connection_generator(
                pre_N=pre_N, post_N=post_N, pre_K=pre_K, post_K=post_K,
                target_density=density, min_accept_frac=min_accept_frac,
                decay=decay, max_tries=max_tries,
                fallback_K=fb_K, seed=None, include_overlap=include_overlap,
                blocks_to_fallback=blocks_to_fallback,
            )
            t_gen = time.perf_counter() - t0
            unique_edges = int(np.sum(conn_matrix > 0))
            total_mapped = int(np.sum(conn_matrix))
            overlap_count = total_mapped - unique_edges
            unique_density = unique_edges / (pre_N * post_N)
            total_density = total_mapped / (pre_N * post_N)
            notes_lines.append(f"  Blocks placed       : {len(mappings)}")
            notes_lines.append(f"  Unique connections  : {unique_edges}  (density {unique_density:.6f} vs target {density:.6f})")
            notes_lines.append(f"  Total mapped edges  : {total_mapped}  (density incl. overlaps {total_density:.6f})")
            notes_lines.append(f"  Duplicate placements: {overlap_count}")
            notes_lines.append(f"  Termination counted : {'total mapped edges (overlap mode)' if include_overlap else 'unique connections only (default)'}")
            notes_lines.append(f"  Time (gen)          : {t_gen:.3f}s")

        total_blocks += len(mappings)
        t0 = time.perf_counter()
        csv_str, all_used = format_mappings_to_csv(mappings, all_used=all_used, tile_size=TILE_SIZE)
        t_map = time.perf_counter() - t0
        with open(csv_path, "w") as f:
            f.write(csv_str)
        notes_lines.append(f"  Tiles allocated after : {len(all_used)}")
        notes_lines.append(f"  Time (mapping)        : {t_map:.3f}s")
        notes_lines.append(f"  Saved to       : {csv_path.name}")
        print(f"  Saved: {csv_path}")

    if non_pyr_connections:
        if exact_inhibitory:
            notes_lines.append("\n--- Non-Pyramidal connections (exact CSV mappings) ---")
            for i, j in non_pyr_connections:
                pre_name = sim.groups[i].groupNameClean
                post_name = sim.groups[j].groupNameClean
                pre_N = group_sizes[i]
                post_N = group_sizes[j]
                csv_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_mapping.csv"
                notes_lines.append(f"\n[{pre_name} -> {post_name}]")
                notes_lines.append(f"  Pre neurons    : {pre_N}  (group {i})")
                notes_lines.append(f"  Post neurons   : {post_N}  (group {j})")
                notes_lines.append(f"  Generator      : exact_connection_generator")
                notes_lines.append(f"  Tiles allocated before: {len(all_used)}")
                print(f"  Generating: {pre_name} -> {post_name}")
                t0 = time.perf_counter()
                mappings, conn_matrix = exact_connection_generator(sim, pre_group_id=i, post_group_id=j)
                t_gen = time.perf_counter() - t0
                actual_density = np.sum(conn_matrix) / (pre_N * post_N)
                notes_lines.append(f"  Exact connections: {int(np.sum(conn_matrix))}")
                notes_lines.append(f"  Actual density   : {actual_density:.6f}")
                notes_lines.append(f"  Blocks emitted   : {len(mappings)}")
                notes_lines.append(f"  Time (gen)       : {t_gen:.3f}s")
                total_blocks += len(mappings)
                t0 = time.perf_counter()
                csv_str, all_used = format_mappings_to_csv(mappings, all_used=all_used, tile_size=TILE_SIZE)
                t_map = time.perf_counter() - t0
                with open(csv_path, "w") as f:
                    f.write(csv_str)
                notes_lines.append(f"  Tiles allocated after : {len(all_used)}")
                notes_lines.append(f"  Time (mapping)        : {t_map:.3f}s")
                notes_lines.append(f"  Saved to       : {csv_path.name}")
                print(f"  Saved: {csv_path}")
        else:
            notes_lines.append("\n--- Non-Pyramidal connections (random binary numpy arrays) ---")
            random.seed(seed)
            for i, j in non_pyr_connections:
                pre_name = sim.groups[i].groupNameClean
                post_name = sim.groups[j].groupNameClean
                pre_N = group_sizes[i]
                post_N = group_sizes[j]
                density = conn_densities[(i, j)]
                target_connections = round(density * pre_N * post_N)
                matrix = np.zeros((pre_N, post_N), dtype=int)
                all_indices = [(pre, post) for pre in range(pre_N) for post in range(post_N)]
                chosen = random.sample(all_indices, target_connections)
                for pre, post in chosen:
                    matrix[pre][post] = 1
                actual_density = np.sum(matrix) / (pre_N * post_N)
                npy_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_matrix.npy"
                np.save(npy_path, matrix)
                notes_lines.append(f"\n[{pre_name} -> {post_name}]")
                notes_lines.append(f"  Pre neurons    : {pre_N}  (group {i})")
                notes_lines.append(f"  Post neurons   : {post_N}  (group {j})")
                notes_lines.append(f"  Target density : {density:.6f}")
                notes_lines.append(f"  Avg weight     : {conn_avg_weights.get((i, j), 0.0):.6f}")
                notes_lines.append(f"  Max weight     : {conn_max_weights.get((i, j), 0.0):.6f}")
                notes_lines.append(f"  Actual density : {actual_density:.6f}  ({int(np.sum(matrix))} connections)")
                notes_lines.append(f"  Saved to       : {npy_path.name}")
                print(f"  Saved: {npy_path}")

    _write_tile_summary(all_used, notes_lines, total_blocks=total_blocks)

    notes_path = out_dir / "mapping_notes.txt"
    with open(notes_path, "w") as f:
        f.write("\n".join(notes_lines) + "\n")
    print(f"  Notes saved: {notes_path}")


def _run_exact_blocked_pass(sim, pyr_connections, non_pyr_connections,
                            group_sizes, conn_counts, conn_densities,
                            conn_avg_weights, conn_max_weights,
                            out_dir, seed, network_file, **_):
    """Write grouped-exact CSVs (1xN blocks per pre-neuron) for all connection types."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Exact-blocked pass] Output: {out_dir}")

    notes_lines = _write_notes_header(network_file, seed, extra_label="[exact-blocked]")
    notes_lines.append(f"Block size : {TILE_SIZE}")
    notes_lines.append("=" * 70)
    _write_group_summary(sim, group_sizes, conn_counts, conn_densities, notes_lines)
    notes_lines.append("\n--- Per-connection mapping details ---")

    all_used = []
    total_blocks = 0

    for i, j in pyr_connections:
        pre_name = sim.groups[i].groupNameClean
        post_name = sim.groups[j].groupNameClean
        pre_N = group_sizes[i]
        post_N = group_sizes[j]
        density = conn_densities[(i, j)]

        csv_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_mapping.csv"

        notes_lines.append(f"\n[{pre_name} -> {post_name}]")
        notes_lines.append(f"  Pre neurons  : {pre_N}  (group {i})")
        notes_lines.append(f"  Post neurons : {post_N}  (group {j})")
        notes_lines.append(f"  Density      : {density:.6f}")
        notes_lines.append(f"  Avg weight   : {conn_avg_weights.get((i, j), 0.0):.6f}")
        notes_lines.append(f"  Max weight   : {conn_max_weights.get((i, j), 0.0):.6f}")
        notes_lines.append(f"  Tiles allocated before: {len(all_used)}")
        notes_lines.append(f"  Generator    : grouped_exact_connection_generator (1xN blocks per pre-neuron)")

        print(f"  Generating: {pre_name} -> {post_name}")
        t0 = time.perf_counter()
        mappings, conn_matrix = grouped_exact_connection_generator(
            sim, pre_group_id=i, post_group_id=j, block_size=TILE_SIZE,
        )
        t_gen = time.perf_counter() - t0
        exact_conns = int(np.sum(conn_matrix))
        actual_density = np.sum(conn_matrix) / (pre_N * post_N)
        pre_neurons_with_conns = int(np.any(conn_matrix, axis=1).sum())
        notes_lines.append(f"  Exact connections: {exact_conns}")
        notes_lines.append(f"  Actual density   : {actual_density:.6f}")
        notes_lines.append(f"  Pre neurons with connections: {pre_neurons_with_conns}")
        notes_lines.append(f"  Blocks emitted   : {len(mappings)}")
        notes_lines.append(f"  Time (gen)       : {t_gen:.3f}s")

        total_blocks += len(mappings)
        t0 = time.perf_counter()
        csv_str, all_used = format_mappings_to_csv(mappings, all_used=all_used, tile_size=TILE_SIZE)
        t_map = time.perf_counter() - t0
        with open(csv_path, "w") as f:
            f.write(csv_str)
        notes_lines.append(f"  Tiles allocated after : {len(all_used)}")
        notes_lines.append(f"  Time (mapping)        : {t_map:.3f}s")
        notes_lines.append(f"  Saved to       : {csv_path.name}")
        print(f"  Saved: {csv_path}")

    if non_pyr_connections:
        notes_lines.append("\n--- Non-Pyramidal connections (grouped exact CSV mappings) ---")
        for i, j in non_pyr_connections:
            pre_name = sim.groups[i].groupNameClean
            post_name = sim.groups[j].groupNameClean
            pre_N = group_sizes[i]
            post_N = group_sizes[j]
            csv_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_mapping.csv"
            notes_lines.append(f"\n[{pre_name} -> {post_name}]")
            notes_lines.append(f"  Pre neurons    : {pre_N}  (group {i})")
            notes_lines.append(f"  Post neurons   : {post_N}  (group {j})")
            notes_lines.append(f"  Avg weight     : {conn_avg_weights.get((i, j), 0.0):.6f}")
            notes_lines.append(f"  Max weight     : {conn_max_weights.get((i, j), 0.0):.6f}")
            notes_lines.append(f"  Generator      : grouped_exact_connection_generator (1xN blocks per pre-neuron)")
            notes_lines.append(f"  Tiles allocated before: {len(all_used)}")
            print(f"  Generating: {pre_name} -> {post_name}")
            t0 = time.perf_counter()
            mappings, conn_matrix = grouped_exact_connection_generator(
                sim, pre_group_id=i, post_group_id=j, block_size=TILE_SIZE,
            )
            t_gen = time.perf_counter() - t0
            actual_density = np.sum(conn_matrix) / (pre_N * post_N)
            notes_lines.append(f"  Exact connections: {int(np.sum(conn_matrix))}")
            notes_lines.append(f"  Actual density   : {actual_density:.6f}")
            notes_lines.append(f"  Blocks emitted   : {len(mappings)}")
            notes_lines.append(f"  Time (gen)       : {t_gen:.3f}s")
            total_blocks += len(mappings)
            t0 = time.perf_counter()
            csv_str, all_used = format_mappings_to_csv(mappings, all_used=all_used, tile_size=TILE_SIZE)
            t_map = time.perf_counter() - t0
            with open(csv_path, "w") as f:
                f.write(csv_str)
            notes_lines.append(f"  Tiles allocated after : {len(all_used)}")
            notes_lines.append(f"  Time (mapping)        : {t_map:.3f}s")
            notes_lines.append(f"  Saved to       : {csv_path.name}")
            print(f"  Saved: {csv_path}")

    _write_tile_summary(all_used, notes_lines, total_blocks=total_blocks)

    notes_path = out_dir / "mapping_notes.txt"
    with open(notes_path, "w") as f:
        f.write("\n".join(notes_lines) + "\n")
    print(f"  Notes saved: {notes_path}")


def _run_exact_pass(sim, pyr_connections, non_pyr_connections,
                    group_sizes, conn_counts, conn_densities,
                    conn_avg_weights, conn_max_weights,
                    out_dir, seed, network_file, exact_inhibitory=False):
    """Write exact-synapse CSVs and exact non-Pyr .npy files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Exact pass] Output: {out_dir}")

    notes_lines = _write_notes_header(network_file, seed, extra_label="[exact]")
    notes_lines.append("=" * 70)
    _write_group_summary(sim, group_sizes, conn_counts, conn_densities, notes_lines)
    notes_lines.append("\n--- Per-connection mapping details ---")

    all_used = []
    total_blocks = 0

    for i, j in pyr_connections:
        pre_name = sim.groups[i].groupNameClean
        post_name = sim.groups[j].groupNameClean
        pre_N = group_sizes[i]
        post_N = group_sizes[j]
        density = conn_densities[(i, j)]

        csv_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_mapping.csv"

        notes_lines.append(f"\n[{pre_name} -> {post_name}]")
        notes_lines.append(f"  Pre neurons  : {pre_N}  (group {i})")
        notes_lines.append(f"  Post neurons : {post_N}  (group {j})")
        notes_lines.append(f"  Density      : {density:.6f}")
        notes_lines.append(f"  Avg weight   : {conn_avg_weights.get((i, j), 0.0):.6f}")
        notes_lines.append(f"  Max weight   : {conn_max_weights.get((i, j), 0.0):.6f}")
        notes_lines.append(f"  Tiles allocated before: {len(all_used)}")
        notes_lines.append(f"  Generator    : exact_connection_generator (one entry per synapse)")

        print(f"  Generating: {pre_name} -> {post_name}")
        t0 = time.perf_counter()
        mappings, conn_matrix = exact_connection_generator(
            sim, pre_group_id=i, post_group_id=j,
        )

        t_gen = time.perf_counter() - t0
        actual_density = np.sum(conn_matrix) / (pre_N * post_N)
        notes_lines.append(f"  Exact connections: {int(np.sum(conn_matrix))}")
        notes_lines.append(f"  Actual density   : {actual_density:.6f}")
        notes_lines.append(f"  Blocks emitted   : {len(mappings)}")
        notes_lines.append(f"  Time (gen)       : {t_gen:.3f}s")

        total_blocks += len(mappings)
        t0 = time.perf_counter()
        csv_str, all_used = format_mappings_to_csv(mappings, all_used=all_used, tile_size=TILE_SIZE)
        t_map = time.perf_counter() - t0
        with open(csv_path, "w") as f:
            f.write(csv_str)
        notes_lines.append(f"  Tiles allocated after : {len(all_used)}")
        notes_lines.append(f"  Time (mapping)        : {t_map:.3f}s")
        notes_lines.append(f"  Saved to       : {csv_path.name}")
        print(f"  Saved: {csv_path}")

    if non_pyr_connections:
        if exact_inhibitory:
            notes_lines.append("\n--- Non-Pyramidal connections (exact CSV mappings) ---")
            for i, j in non_pyr_connections:
                pre_name = sim.groups[i].groupNameClean
                post_name = sim.groups[j].groupNameClean
                pre_N = group_sizes[i]
                post_N = group_sizes[j]
                csv_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_mapping.csv"
                notes_lines.append(f"\n[{pre_name} -> {post_name}]")
                notes_lines.append(f"  Pre neurons    : {pre_N}  (group {i})")
                notes_lines.append(f"  Post neurons   : {post_N}  (group {j})")
                notes_lines.append(f"  Generator      : exact_connection_generator")
                notes_lines.append(f"  Tiles allocated before: {len(all_used)}")
                print(f"  Generating: {pre_name} -> {post_name}")
                t0 = time.perf_counter()
                mappings, conn_matrix = exact_connection_generator(sim, pre_group_id=i, post_group_id=j)
                t_gen = time.perf_counter() - t0
                actual_density = np.sum(conn_matrix) / (pre_N * post_N)
                notes_lines.append(f"  Exact connections: {int(np.sum(conn_matrix))}")
                notes_lines.append(f"  Actual density   : {actual_density:.6f}")
                notes_lines.append(f"  Blocks emitted   : {len(mappings)}")
                notes_lines.append(f"  Time (gen)       : {t_gen:.3f}s")
                total_blocks += len(mappings)
                t0 = time.perf_counter()
                csv_str, all_used = format_mappings_to_csv(mappings, all_used=all_used, tile_size=TILE_SIZE)
                t_map = time.perf_counter() - t0
                with open(csv_path, "w") as f:
                    f.write(csv_str)
                notes_lines.append(f"  Tiles allocated after : {len(all_used)}")
                notes_lines.append(f"  Time (mapping)        : {t_map:.3f}s")
                notes_lines.append(f"  Saved to       : {csv_path.name}")
                print(f"  Saved: {csv_path}")
        else:
            notes_lines.append("\n--- Non-Pyramidal connections (exact binary numpy arrays) ---")
            for i, j in non_pyr_connections:
                pre_name = sim.groups[i].groupNameClean
                post_name = sim.groups[j].groupNameClean
                pre_N = group_sizes[i]
                post_N = group_sizes[j]
                matrix = np.zeros((pre_N, post_N), dtype=int)
                for syn in sim.synapses[0]:
                    if syn.gGrpIdPre == i and syn.gGrpIdPost == j:
                        matrix[syn.grpNIdPre][syn.grpNIdPost] = 1
                actual_density = np.sum(matrix) / (pre_N * post_N)
                npy_path = out_dir / f"{_safe_name(pre_name)}_{_safe_name(post_name)}_matrix.npy"
                np.save(npy_path, matrix)
                notes_lines.append(f"\n[{pre_name} -> {post_name}]")
                notes_lines.append(f"  Pre neurons    : {pre_N}  (group {i})")
                notes_lines.append(f"  Post neurons   : {post_N}  (group {j})")
                notes_lines.append(f"  Exact density  : {actual_density:.6f}  ({int(np.sum(matrix))} connections)")
                notes_lines.append(f"  Saved to       : {npy_path.name}")
                print(f"  Saved: {npy_path}")

    _write_tile_summary(all_used, notes_lines, total_blocks=total_blocks)

    notes_path = out_dir / "mapping_notes.txt"
    with open(notes_path, "w") as f:
        f.write("\n".join(notes_lines) + "\n")
    print(f"  Notes saved: {notes_path}")


def generate_mappings(network_file, seed=DEFAULT_SEED, out_dir=None, include_overlap=False,
                      fallback_k=FALLBACK_TILE_SIZE, pre_k=TILE_SIZE, post_k=TILE_SIZE,
                      inh_exc_pre_k=1, inh_exc_post_k=TILE_SIZE, exact_inhibitory=False,
                      min_accept_frac=DEFAULT_MIN_ACCEPT_FRAC, decay=DEFAULT_DECAY,
                      max_tries=DEFAULT_MAX_TRIES, blocks_to_fallback=None,
                      skip_greedy=False):
    # ------------------------------------------------------------------
    # Load network
    # ------------------------------------------------------------------
    sim = Simulation(network_file)
    net_stem = Path(network_file).stem

    # ------------------------------------------------------------------
    # Resolve base output directory
    # ------------------------------------------------------------------
    if out_dir is None:
        net_dir = Path(network_file).parent
        base_dir = net_dir / f"{net_stem}_hw_mappings_seed{seed}"
    else:
        base_dir = Path(out_dir)

    # ------------------------------------------------------------------
    # Compute group sizes and connection densities
    # ------------------------------------------------------------------
    group_sizes = {i: _group_size(g) for i, g in enumerate(sim.groups)}
    conn_counts = defaultdict(int)
    conn_weight_sums = defaultdict(float)
    conn_max_weights = defaultdict(float)
    for syn in sim.synapses[0]:
        key = (syn.gGrpIdPre, syn.gGrpIdPost)
        conn_counts[key] += 1
        conn_weight_sums[key] += syn.weight
        conn_max_weights[key] = max(conn_max_weights[key], syn.maxWeight)

    conn_avg_weights = {
        k: conn_weight_sums[k] / conn_counts[k]
        for k in conn_counts
        if conn_counts[k] > 0
    }

    conn_densities = {}
    for i in range(sim.numGroups):
        for j in range(sim.numGroups):
            conn_densities[(i, j)] = conn_counts[(i, j)] / (group_sizes[i] * group_sizes[j])

    # ------------------------------------------------------------------
    # Identify connections
    # ------------------------------------------------------------------
    pyr_connections = [
        (i, j)
        for i in range(sim.numGroups)
        for j in range(sim.numGroups)
        if conn_densities[(i, j)] > 0
        and (_is_pyramidal(sim.groups[i]) or _is_pyramidal(sim.groups[j]))
    ]
    non_pyr_connections = [
        (i, j)
        for i in range(sim.numGroups)
        for j in range(sim.numGroups)
        if conn_densities[(i, j)] > 0
        and not _is_pyramidal(sim.groups[i])
        and not _is_pyramidal(sim.groups[j])
    ]

    if not pyr_connections:
        print("No connections involving Pyramidal groups found.")
        return

    print(f"Found {len(pyr_connections)} Pyramidal connection(s):")
    for i, j in pyr_connections:
        print(f"  {sim.groups[i].groupNameClean} -> {sim.groups[j].groupNameClean}"
              f"  density={conn_densities[(i,j)]:.6f}")

    # ------------------------------------------------------------------
    # Run both passes
    # ------------------------------------------------------------------
    shared = dict(
        sim=sim,
        pyr_connections=pyr_connections,
        non_pyr_connections=non_pyr_connections,
        group_sizes=group_sizes,
        conn_counts=conn_counts,
        conn_densities=conn_densities,
        conn_avg_weights=conn_avg_weights,
        conn_max_weights=conn_max_weights,
        seed=seed,
        network_file=network_file,
    )

    if not skip_greedy:
        _run_greedy_pass(**shared, out_dir=base_dir / "greedy", include_overlap=include_overlap,
                         fallback_k=fallback_k, pre_k=pre_k, post_k=post_k,
                         inh_exc_pre_k=inh_exc_pre_k, inh_exc_post_k=inh_exc_post_k,
                         exact_inhibitory=exact_inhibitory,
                         min_accept_frac=min_accept_frac, decay=decay, max_tries=max_tries,
                         blocks_to_fallback=blocks_to_fallback)
    _run_exact_pass(**shared, out_dir=base_dir / "exact", exact_inhibitory=exact_inhibitory)
    _run_exact_blocked_pass(**shared, out_dir=base_dir / "exact-blocked", exact_inhibitory=exact_inhibitory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Daffodil hardware mapping CSVs for Pyramidal connections."
    )
    parser.add_argument(
        "network_file",
        nargs="?",
        default=str(Path(__file__).parent.parent / "data" / "CA3_0001_network.dat"),
        help="Path to CARLsim .dat network file (default: data/CA3_0001_network.dat)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: data/<stem>_hw_mappings_seed<seed>/)",
    )
    parser.add_argument(
        "--pre-k",
        type=int,
        default=TILE_SIZE,
        help=(
            f"Initial kernel size for the pre-synaptic dimension of Pyramidal groups "
            f"(default: {TILE_SIZE}). Non-pyramidal groups always use 1 regardless of "
            f"this value."
        ),
    )
    parser.add_argument(
        "--post-k",
        type=int,
        default=TILE_SIZE,
        help=(
            f"Initial kernel size for the post-synaptic dimension of Pyramidal groups "
            f"(default: {TILE_SIZE}). Non-pyramidal groups always use 1 regardless of "
            f"this value."
        ),
    )
    parser.add_argument(
        "--inh-exc-pre-k",
        type=int,
        default=1,
        help=(
            f"Initial pre-synaptic kernel size for inhibitory->excitatory (NonPyr->Pyr) "
            f"connections (default: 1). Applies independently of --pre-k, which only "
            f"controls Pyr->Pyr pre dimensions."
        ),
    )
    parser.add_argument(
        "--inh-exc-post-k",
        type=int,
        default=TILE_SIZE,
        help=(
            f"Initial post-synaptic kernel size for inhibitory->excitatory (NonPyr->Pyr) "
            f"connections (default: {TILE_SIZE}). Applies independently of --post-k, which "
            f"only controls Pyr->Pyr post dimensions. Use this to avoid inh->exc blocks "
            f"collapsing to 1x1 when --post-k is set small."
        ),
    )
    parser.add_argument(
        "--fallback-k",
        type=int,
        default=FALLBACK_TILE_SIZE,
        help=(
            f"Kernel size to fall back to when the greedy algorithm stalls "
            f"(default: {FALLBACK_TILE_SIZE}). Only applies to connections where "
            f"at least one kernel dimension starts at the full tile size ({TILE_SIZE}). "
            f"A smaller value means finer-grained blocks during the fallback phase, "
            f"which can help reach the target density at the cost of more placement attempts."
        ),
    )
    parser.add_argument(
        "--exact-inhibitory",
        action="store_true",
        default=False,
        help=(
            "Generate exact CSV tile mappings for non-pyramidal (inhibitory) connections "
            "instead of saving random binary numpy arrays. Uses exact_connection_generator "
            "to emit one block per synapse, packed into the shared tile set alongside "
            "pyramidal connections. Without this flag (default), non-pyramidal connections "
            "are saved as .npy matrix files and do not consume hardware tiles."
        ),
    )
    parser.add_argument(
        "--min-accept-frac",
        type=float,
        default=DEFAULT_MIN_ACCEPT_FRAC,
        help=(
            f"Initial minimum fraction of available (non-duplicate) edges required "
            f"to accept a proposed block (default: {DEFAULT_MIN_ACCEPT_FRAC}). "
            f"Higher values produce cleaner blocks with fewer overlaps but may "
            f"require more attempts or threshold decay to reach the target density."
        ),
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=DEFAULT_DECAY,
        help=(
            f"Multiplicative factor applied to --min-accept-frac each time no block "
            f"can be placed within --max-tries attempts (default: {DEFAULT_DECAY}). "
            f"Values closer to 1.0 decay more slowly; lower values relax the "
            f"threshold faster at the cost of more overlapping blocks."
        ),
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=DEFAULT_MAX_TRIES,
        help=(
            f"Maximum placement attempts per block before the acceptance threshold "
            f"is decayed (default: {DEFAULT_MAX_TRIES}). Increase for denser "
            f"connections where finding a valid block takes longer."
        ),
    )
    parser.add_argument(
        "--blocks-to-fallback",
        type=int,
        default=None,
        help=(
            "Drop to the fallback kernel size after this many successful block placements, "
            "even if the greedy algorithm has not stalled. Useful for front-loading large "
            "blocks early in the run and switching to finer-grained blocks for the remainder. "
            "Only applies when --fallback-k is also set and the initial kernel is larger than "
            "the fallback. Default: None (fallback is only triggered by stalls)."
        ),
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        default=False,
        help="Skip the greedy mapping pass and only run exact and exact-blocked.",
    )
    parser.add_argument(
        "--include-overlap",
        action="store_true",
        default=False,
        help=(
            "Count duplicate (overlapping) connections toward the target density. "
            "Without this flag (default), the greedy algorithm only counts unique "
            "connections — it keeps placing blocks until the target density of "
            "distinct edges is reached, even if many blocks overlap. "
            "With this flag, every edge in an accepted block counts toward the "
            "target regardless of whether it was already placed, so the algorithm "
            "terminates sooner but the actual unique-connection density will be "
            "lower than the target."
        ),
    )
    args = parser.parse_args()

    generate_mappings(
        network_file=args.network_file,
        seed=args.seed,
        out_dir=args.out_dir,
        include_overlap=args.include_overlap,
        fallback_k=args.fallback_k,
        pre_k=args.pre_k,
        post_k=args.post_k,
        inh_exc_pre_k=args.inh_exc_pre_k,
        inh_exc_post_k=args.inh_exc_post_k,
        exact_inhibitory=args.exact_inhibitory,
        min_accept_frac=args.min_accept_frac,
        decay=args.decay,
        max_tries=args.max_tries,
        blocks_to_fallback=args.blocks_to_fallback,
        skip_greedy=args.skip_greedy,
    )