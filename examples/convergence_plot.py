from pathlib import Path

from matplotlib import pyplot as plt
from optirustic import NSGA2

folder = Path(__file__).parent / "results" / "convergence"
NSGA2.plot_convergence(folder.as_posix(), [10000, 10000])
plt.show()
