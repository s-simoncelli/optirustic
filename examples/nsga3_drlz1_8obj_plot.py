from pathlib import Path

from matplotlib import pyplot as plt
from optirustic import NSGA3

# Generate Pareto front charts and objective vs. reference point charts
# A Parallel coordinate chart will be generated
file = Path(__file__).parent / "results" / "DTLZ1_8obj_NSGA3_gen750.json"
data = NSGA3(file.as_posix())

# Generate Pareto front chart
data.plot()
plt.savefig(file.parent / f"{file.stem}_Pareto_front.png")
