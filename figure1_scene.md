# Clip 1
Full-scale CA3 network: 8 neuron types in a circle, dense blue (excitatory,
from Pyramidal) and gold (inhibitory) connections, real totals from the paper
(89,226 neurons, 250,078,223 synapses).

# Clip 2
"Scale network size": shift the full-scale network left, arrow, a smaller
version of the same 8-type network fades in on the right.

# Clip 3
"Scale neuron diversity": remove the full-scale network, move the reduced
8-type network left, arrow, it becomes the 3-type triangle (Pyramidal /
Basket / MFA ORDEN) on the right - drawn with the real spiking neuron counts
from data/CA3_0_0001_3_group_18 (160 / 6 / 12).

# Clip 4
"Implementation": remove the reduced 8-type network, move the 3-type triangle
left, arrow, a 25x25 memristor crossbar tile packs in on the right (reuses
greedy_blocking.py's real greedy algorithm - see greedy_scene.py for the full
narrated version of this same mechanism).

# Clip 5
Close the loop: remove the 3-type triangle, bring the full-scale network back
in on the left pointing at the crossbar + the real hardware photo (FPGA +
mixed-signal memristor daughterboard) on the right.
