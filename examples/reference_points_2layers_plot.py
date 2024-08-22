import json
from pathlib import Path

import matplotlib.pyplot as plt
from optirustic import plot_reference_points

# Plot a 3D chart with the reference points
result_folder = Path(__file__).parent / "results"
with open(result_folder / "ref_points_2layers_3obj_5gaps.json") as fid:
    data = json.load(fid)
    plot_reference_points(data)
    plt.savefig(result_folder / "ref_points_2layers_3obj_5gaps.png")
