from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from optirustic import NSGA3

# Generate a 3D Pareto front charts and objective vs. reference point charts
file = Path(__file__).parent / "results" / "DTLZ1_3obj_NSGA3_gen400.json"
data = NSGA3(file.as_posix())

# Generate Pareto front chart
data.plot()
plt.savefig(file.parent / f"{file.stem}_Pareto_front.png")

# Generate a chart with the normalised objectives against the reference points
# The plane is limited in the [0, 1] range.
normalised_objectives = [ind.data["normalised_objectives"] for ind in data.individuals]
normalised_objectives = np.array(normalised_objectives)
obj_names = data.problem.objective_names

ref_points = data.additional_data["reference_points"]
ref_points = np.array(ref_points)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter(
    ref_points[:, 0],
    ref_points[:, 1],
    ref_points[:, 2],
    color="r",
    marker="x",
    s=40,
    label="Reference points objectives",
)
ax.scatter(
    normalised_objectives[:, 0],
    normalised_objectives[:, 1],
    normalised_objectives[:, 2],
    color="k",
    marker=".",
    label="Normalised objectives",
)
ax.set_xlabel(obj_names[0])
ax.set_ylabel(obj_names[1])
ax.set_zlabel(obj_names[2])
ax.view_init(azim=10)

plt.legend()
plt.title(
    f"Normalised objectives vs. reference points \n"
    f"for {data.algorithm} @ generation={data.generation}"
)
plt.savefig(file.parent / f"{file.stem}_obj_vs_ref_points.png")
