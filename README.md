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
| `DefaultsCard` | Extra card showing the real defaults from `generate_hw_mappings.py` |

```powershell
manim -pql greedy_scene.py Clip1 Clip2 Clip3 Clip4 Clip5 Clip6 Clip7 Clip8 Clip9
manim -pql greedy_scene.py DefaultsCard
```

## Other files

- [greedy_blocking.py](greedy_blocking.py): a re-implementation of the greedy block placement that records an event log for `greedy_scene.py` and `figure1_scene.py` to replay.
- [map.py](map.py): a standalone script, not a scene. It generates Daffodil hardware-mapping CSVs from a CARLsim `.dat` network. It needs `simulation.py` and `connection_generator.py` on the import path.
- [sim-loader/](sim-loader/): loaders for CARLsim networks and network parameters. The scoring scenes import from it.
