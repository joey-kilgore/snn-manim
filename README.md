# snn-manim

[Manim](https://www.manim.community/) animations for the scaled CA3 spiking neural network work: *Scaled Hippocampus-inspired Neural Networks on Neuromorphic Memristive Hardware* and the CA3-scaling scoring rubric (arXiv:2609.16429).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install manim numpy
```

Most scenes read the 3-group CA3 recording in `data/CA3_0_0001_3_group_18/` (`detailed_spikes.csv`, `voltages.csv`, `network.dat`). The scoring scenes also import from `sim-loader/` (`networkParams.py`, `scripts/simulation.py`). Figure 1 uses `assets/fig1_hardware_photo.png`.

## Rendering

```powershell
manim -pql <file>.py <SceneName>   # quick 480p preview, opens when done
manim -pqh <file>.py <SceneName>   # 1080p
```

You can list several scene names in one command to render them all. Output goes to `media/videos/<file>/<quality>/<SceneName>.mp4`.

The commands below use `-pql` for previews. Swap in `-qh` for final 1080p renders.

| Flag | Resolution | Output folder |
| --- | --- | --- |
| `-ql` | 854x480, 15 fps | `media/videos/<file>/480p15/` |
| `-qm` | 1280x720, 30 fps | `media/videos/<file>/720p30/` |
| `-qh` | 1920x1080, 60 fps | `media/videos/<file>/1080p60/` |

Rendered videos are not tracked in git (`/media/` is ignored), so use the recipes in [Rendering every video](#rendering-every-video) to regenerate them.

## Scenes

### Background

| Scene | Description | Command |
| --- | --- | --- |
| `RasterReveal` | Raster of the last ~1 s of the 3-group CA3 recording, built spike by spike, with each group's mean voltage trace drawn underneath. | `manim -pql raster_scene.py RasterReveal` |

### Figure 1: `figure1_scene.py`

Animates Figure 1 of the paper, going from the full-scale CA3 network to the hardware. The clip notes are in [figure1_scene.md](figure1_scene.md).

| Scene | Description |
| --- | --- |
| `Clip1` | Full-scale CA3 network: 8 neuron types, 89,226 neurons, 250M synapses |
| `Clip2` | Scale network size: full-scale → reduced-scale (still 8 types) |
| `Clip3` | Scale neuron diversity: 8 types → 3 types (Pyramidal / Basket / MFA ORDEN) |
| `Clip4` | Implementation: 3-type network → a 25×25 memristor crossbar tile |
| `Clip5` | Closing shot: the full-scale network pointing at the FPGA + memristor hardware |

```powershell
manim -pql figure1_scene.py Clip1 Clip2 Clip3 Clip4 Clip5
```

### Scoring rubric

There is one scene for each criterion used to score a scaled network against the full-scale target.

| Criterion | Scene | Description | Command |
| --- | --- | --- | --- |
| 1. Activity A(N) | `ActivityScore` | The last 1 s is split into 15 bins of ~66 ms per subtype. A(N) is the squared fraction of (subtype, bin) cells with at least one spike. | `manim -pql activity_score_scene.py ActivityScore` |
| 2. Periodicity P(N) | `PeriodicityScore` | Pyramidal LFP → PSD → log-log baseline fit → power ratio → theta/beta/gamma band scores, weighted into P(N). | `manim -pql periodicity_scene.py PeriodicityScore` |
| 3. Firing rate F(N) | `FiringRateScore` | Each subtype's mean firing rate is scored with a normalized Gaussian around its full-scale target rate, then averaged. | `manim -pql firing_rate_score_scene.py FiringRateScore` |
| 4. Connection C(N) | `ConnectionScore` | The scaling law gives target connection probabilities. The tuned probabilities from `network.dat` are scored with a Gaussian across all 9 connections. | `manim -pql connection_score_scene.py ConnectionScore` |

`ActivityScore` is also split into two clips (`ActivityScoreClip1`, `ActivityScoreClip2`): clip 1 ends as the first cell is highlighted as active, and clip 2 opens on that same frame and runs to the end.

```powershell
manim -pql activity_score_scene.py ActivityScoreClip1 ActivityScoreClip2
```

`PeriodicityScore` is also split into three clips (`Clip1`-`Clip3`): `Clip1` ends with the FFT PSD on the bottom panel, `Clip2` ends on the best-fit line in log-log space, `Clip3` covers the ratio scan, band scores, and P(N).

```powershell
manim -pql periodicity_scene.py Clip1 Clip2 Clip3
```

`FiringRateScore` is also split into three clips (`Clip1`-`Clip3`), each rebuilding the previous clip's final state instantly before animating its own slice: `Clip1` ends on the firing-rate bar chart, `Clip2` ends on the worked Gaussian for Pyramidal, `Clip3` covers all three subtypes' curves and the combined F(N).

```powershell
manim -pql firing_rate_score_scene.py Clip1 Clip2 Clip3
```

### ANN vs SNN communication: `communication_scene.py`

Compares message traffic in a dense ANN with a sparse spiking network (5 → 8 → 8 → 4). The clip notes are in [communication_scene.md](communication_scene.md).

| Scene | Description |
| --- | --- |
| `ANNCommunication` | ANN only: every connection carries a value every tick |
| `SNNCommunication` | SNN only: a connection carries a message only when its source neuron spikes |
| `ANNvsSNN` | Both networks side by side in one video |
| `ANNvsSNNClip1` | Build both networks, showing SNN sparsity and neuron diversity |
| `ANNvsSNNClip2` | One slow time step |
| `ANNvsSNNClip3` | The remaining time steps, faster |
| `ANNvsSNNClip4` | Final tally: SNN traffic vs ANN traffic |

```powershell
manim -pql communication_scene.py ANNCommunication SNNCommunication
manim -pql communication_scene.py ANNvsSNN
manim -pql communication_scene.py ANNvsSNNClip1 ANNvsSNNClip2 ANNvsSNNClip3 ANNvsSNNClip4
```

### Why backprop fails on spikes: `training_differences_scene.py`

Follows on from the "Differences in Training" slides. A traditional
(sigmoid) and a spiking network -- same input, same weights, only the
nonlinearity differs -- propagate forward, then get their weights updated,
then lose the network entirely so the two derivative curves can be
compared directly. The spiking side is treated as an instantaneous step
`S(v) = H(v-theta)` applied once per layer (no leaky/temporal membrane
dynamics), so both architectures share the exact same feed-forward
structure. Ends on the same problem statement each time: surrogate
gradients are the next slide, not this one.

| Scene | Description | Command |
| --- | --- | --- |
| `Clip1` | Forward pass: values propagate layer by layer, landing on each architecture's activation function as they go -- the traditional net's values spread out, the spiking net is mostly silent. | `manim -pql training_differences_scene.py Clip1` |
| `Clip2` | Weight update: the per-layer chain rule appears for both, then each activation panel grows its derivative curve, with a marker at the value one specific neuron actually produced -- even the neuron that fired gets almost no gradient. | `manim -pql training_differences_scene.py Clip2` |
| `Clip3` | The derivative's area: networks fade away, a single arrow at each curve's peak looks comparable (the spike's even taller), until the area under each curve is shaded and compared -- the sigmoid's derivative is usable across a wide band, the spike's only in a sliver. | `manim -pql training_differences_scene.py Clip3` |

```powershell
manim -pql training_differences_scene.py Clip1 Clip2 Clip3
```

### Greedy block placement: `greedy_scene.py`

Shows how connections are packed onto the memristor crossbar with greedy block placement. The algorithm is replayed from [greedy_blocking.py](greedy_blocking.py), and the clip notes are in [greedy_scene.md](greedy_scene.md). Each clip rebuilds the state the previous clip ended on, so every clip can be rendered on its own.

| Scene | Description |
| --- | --- |
| `Clip1` | Pre/post-synaptic neurons drawn as a block of connections |
| `Clip2` | Build the matrix and parameter panel, then place the first block |
| `Clip3` | Place more blocks, up to the first rejection |
| `Clip4` | More rejections until a block is accepted |
| `Clip5` | Continue until the 6th rejection in a row |
| `Clip6` | What changes when the failure budget runs out |
| `Clip7` | Place the remaining smaller blocks, then pull out the final blocks |
| `Clip8` | Pack the blocks into one 25×25 crossbar tile |
| `Clip9` | Spikes → read voltages → current for a single block |
| `GreedyFull` | Clips 1–8 as one video of about a minute: the random sampling (clips 2–7) runs at 5× speed and the block read (clip 9) is left out |
| `DefaultsCard` | Extra card showing the real defaults from `generate_hw_mappings.py` |

```powershell
manim -pql greedy_scene.py Clip1 Clip2 Clip3 Clip4 Clip5 Clip6 Clip7 Clip8 Clip9
manim -pql greedy_scene.py GreedyFull
manim -pql greedy_scene.py DefaultsCard
```

## Other files

- [greedy_blocking.py](greedy_blocking.py): a re-implementation of the greedy block placement that records an event log for `greedy_scene.py` and `figure1_scene.py` to replay.
- [map.py](map.py): a standalone script, not a scene. It generates Daffodil hardware-mapping CSVs from a CARLsim `.dat` network. It needs `simulation.py` and `connection_generator.py` on the import path.
- [sim-loader/](sim-loader/): loaders for CARLsim networks and network parameters. The scoring scenes import from it.

## Rendering every video

Each block renders one quality. Run the low-res block for quick previews and the high-res block for final output. Drop `-p` if you do not want each video to open when it finishes. Scenes that share a file are listed in one command, so the file is only loaded once.

### Low resolution (480p15)

```powershell
# Full single-video versions
manim -ql raster_scene.py RasterReveal
manim -ql activity_score_scene.py ActivityScore
manim -ql periodicity_scene.py PeriodicityScore
manim -ql firing_rate_score_scene.py FiringRateScore
manim -ql connection_score_scene.py ConnectionScore
manim -ql communication_scene.py ANNCommunication SNNCommunication ANNvsSNN
manim -ql greedy_scene.py GreedyFull DefaultsCard

# Multi-clip versions
manim -ql figure1_scene.py Clip1 Clip2 Clip3 Clip4 Clip5
manim -ql activity_score_scene.py ActivityScoreClip1 ActivityScoreClip2
manim -ql periodicity_scene.py Clip1 Clip2 Clip3
manim -ql firing_rate_score_scene.py Clip1 Clip2 Clip3
manim -ql communication_scene.py ANNvsSNNClip1 ANNvsSNNClip2 ANNvsSNNClip3 ANNvsSNNClip4
manim -ql greedy_scene.py Clip1 Clip2 Clip3 Clip4 Clip5 Clip6 Clip7 Clip8 Clip9
manim -ql training_differences_scene.py Clip1 Clip2 Clip3
```

### High resolution (1080p60)

```powershell
# Full single-video versions
manim -qh raster_scene.py RasterReveal
manim -qh activity_score_scene.py ActivityScore
manim -qh periodicity_scene.py PeriodicityScore
manim -qh firing_rate_score_scene.py FiringRateScore
manim -qh connection_score_scene.py ConnectionScore
manim -qh communication_scene.py ANNCommunication SNNCommunication ANNvsSNN
manim -qh greedy_scene.py GreedyFull DefaultsCard

# Multi-clip versions
manim -qh figure1_scene.py Clip1 Clip2 Clip3 Clip4 Clip5
manim -qh activity_score_scene.py ActivityScoreClip1 ActivityScoreClip2
manim -qh periodicity_scene.py Clip1 Clip2 Clip3
manim -qh firing_rate_score_scene.py Clip1 Clip2 Clip3
manim -qh communication_scene.py ANNvsSNNClip1 ANNvsSNNClip2 ANNvsSNNClip3 ANNvsSNNClip4
manim -qh greedy_scene.py Clip1 Clip2 Clip3 Clip4 Clip5 Clip6 Clip7 Clip8 Clip9
manim -qh training_differences_scene.py Clip1 Clip2 Clip3
```

To render everything at once for a given quality, run both blocks back to back. Swap `-ql`/`-qh` for `-qm` to get 720p30.

### Combining clips

The clip scenes are meant to be played back to back (each one starts on the frame the previous one ended on, except where the notes say otherwise), so a combined video is just the clips concatenated in order. With [ffmpeg](https://ffmpeg.org/) installed, make a list file and join without re-encoding. Use the same quality folder for every clip, and swap `1080p60` for `480p15` for the low-res version:

```powershell
# example: the three PeriodicityScore clips -> one video
$dir = "media/videos/periodicity_scene/1080p60"
"file 'Clip1.mp4'", "file 'Clip2.mp4'", "file 'Clip3.mp4'" | Set-Content "$dir/clips.txt" -Encoding ascii
ffmpeg -f concat -safe 0 -i "$dir/clips.txt" -c copy "$dir/PeriodicityScore_combined.mp4"
```

The same pattern works for the other sets, changing the folder and the clip names:

| Combined video | Folder (under `media/videos/`) | Clips, in order |
| --- | --- | --- |
| Figure 1 | `figure1_scene` | `Clip1` ... `Clip5` |
| Activity score | `activity_score_scene` | `ActivityScoreClip1`, `ActivityScoreClip2` |
| Periodicity score | `periodicity_scene` | `Clip1`, `Clip2`, `Clip3` |
| Firing rate score | `firing_rate_score_scene` | `Clip1`, `Clip2`, `Clip3` |
| ANN vs SNN | `communication_scene` | `ANNvsSNNClip1` ... `ANNvsSNNClip4` |
| Greedy placement | `greedy_scene` | `Clip1` ... `Clip8` (or render `GreedyFull`); add `Clip9` for the block read |
| Training differences | `training_differences_scene` | `Clip1`, `Clip2`, `Clip3` |

If you would rather not concatenate, `ActivityScore`, `PeriodicityScore`, `FiringRateScore`, `ANNvsSNN` and `GreedyFull` render each clip set as a single video.

### Notebook

`optuna_final_0_01_network.ipynb` breaks down the network from Optuna trial 1647. Run it from the repo root; it reads `data/optuna_final_0_01.db` and the generated-network pickle, which are not tracked here, and needs `pip install optuna pandas`.
