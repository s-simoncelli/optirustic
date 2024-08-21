from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from optirustic import NSGA2

file = Path(__file__).parent / "results" / "SCH_2obj_NSGA2_gen250.json"
data = NSGA2(file.as_posix())

# get the list of [variable, f1, f2] values
individuals_data = [[ind.variables["x"], ind.objectives["x^2"], ind.objectives["(x-2)^2"]] for ind in data.individuals]
individuals_data = np.array(individuals_data)

plt.figure()
x = np.arange(-6, 6, 0.1)
f1 = np.pow(x, 2)
f2 = np.pow(x - 2, 2)

# theoretical curves
plt.plot(x, f1, color="b", label="Objective f1")
plt.plot(x, f2, color="g", label="Objective f2")

# data from algorithm
plt.plot(individuals_data[:, 0], individuals_data[:, 1], "r.")
plt.plot(individuals_data[:, 0], individuals_data[:, 2], "r.", label="Solution")

plt.legend()
plt.grid()

plt.xticks(range(-5, 5))
plt.xlim(-3, 5)
plt.xlabel("x")

plt.ylabel("Objective")
plt.ylim(-1, 20)
plt.title("SCH problem solved with NSGA2")

plt.savefig(file.parent / "SCH_2obj_NSGA2_solutions.png")
