# Optirustic Py

This is a Python package that let users import serialised data from JSON files
exported with the `optirustic` crate. It lets you:

- import data into Python classes for easy manipulation;
- calculate the population hyper-volume;
- plot 2D, 3D or parallel coordinate charts of the Pareto front.

# Installation

The package can be installed from [PyPi](https://pypi.org/project/optirustic/):

```
pip install optirustic_py
```

# Usage

These are two example scripts to fetch data and plot the Pareto fronts of the `optirustic` example
files

## Python API

All Python API are available in your editor via type hints:

```python
from optirustic import NSGA3

# Load the NSGA3 data first
data = NSGA3(r"../examples/results/DTLZ1_3obj_NSGA3_gen400.json")

# Fetch the problem data
p = data.problem
print(p.number_of_variables)
print(p.variables)
# Fetch the lower bound of X!
print(p.variables["x1"].min_value)

# Get the objective stored into the problem
print(p.objectives)
# Fetch the direction of objective f1
print(p.objectives["f1"].direction)

# Fetch the problem constraints
print(p.constraints)

# Fetch other data such as the algorithm name or generation
print(f"Algorithm name: {data.algorithm}")
print(f"Population reached generation: {data.generation}")
print(f"Algorithm took: {data.took}")
print(f"JSOn file exported on: {data.exported_on}")

# Fetch data for the first individual
print(data.individuals[0])
print(data.individuals[0].constraint_violation)
print(f"Objective values: {data.individuals[0].objectives}")
print(f"Objective f2 value is: {data.individuals[0].get_objective_value("f2")}")
print(f"Variable values: {data.individuals[0].variables}")
print(f"Additional stored data: {data.individuals[0].data}")

# Calculate the hyper-volume
print(f"Hyper-volume is: {data.hyper_volume(reference_point=[100, 100, 100])}")
```

## Generate Pareto front chart

```python
import matplotlib.pyplot as plt
from optirustic import NSGA2, NSGA3

# Plot a 2D charts for a 2-objective problem
NSGA2(r"../examples/results/SCH_2obj_NSGA2_gen250.json").plot()
plt.show()

# Plot a 3D charts for a 3-objective problem
NSGA3(r"../examples/results/DTLZ1_3obj_NSGA3_gen400.json").plot()
plt.show()

# Plot a parallel coordinate chart for an 8-objective problem
NSGA3(r"../examples/results/DTLZ1_8obj_NSGA3_gen750.json").plot()
plt.show()

```

## Generate convergence chart

This template script plots the algorithm convergence by calculating
the hyper-volumes at different generations:

```python
import matplotlib.pyplot as plt
from optirustic import NSGA2

# provide the folder where optimistic exported the JSON files
# and a reference point to use in the hyper-volume calculation
NSGA2.plot_convergence(
    folder=r"../examples/results/convergence",
    reference_point=[10000, 10000]
)
plt.show()
```

## Generate reference points

To generate, plot and inspect the reference points for the `NSGA3` algorithm you can us:

### One layer

```python
import matplotlib.pyplot as plt
from optirustic import DasDarren1998

ds = DasDarren1998(number_of_objectives=3, number_of_partitions=5)
points = ds.calculate()
ds.plot(points)
plt.show()

```

### Two layers

```python
import matplotlib.pyplot as plt
from optirustic import DasDarren1998

two_layers = dict(
    boundary_layer=3,
    inner_layer=4,
    scaling=None,
)
ds = DasDarren1998(number_of_objectives=3, number_of_partitions=two_layers)
points = ds.calculate()
ds.plot(points)
plt.show()
```