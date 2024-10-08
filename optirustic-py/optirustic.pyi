from datetime import timedelta, datetime
from enum import Enum
from typing import TypedDict

import matplotlib.pyplot as plt

class ObjectiveDirection(Enum):
    """
    A class describing the objective direction.
    """

    Minimise = "minimise"
    """ The objective is minimised """
    Maximise = "maximise"
    """ The objective is maximised """

class Objective:
    """
    An objective set on the solved problem.
    """

    name: str
    """ The objective name. """
    direction: ObjectiveDirection
    """ Whether the objective should be minimised or maximised. """

class RelationalOperator(Enum):
    """
    Operator used to check a bounded constraint.
    """

    EqualTo = "=="
    NotEqualTo = "!="
    LessOrEqualTo = "<="
    LessThan = "<"
    GreaterOrEqualTo = ">="
    GreaterThan = ">"

class Constraint:
    """
    A constraint set on the solved problem.
    """

    name: str
    """ The constraint name """
    operator: RelationalOperator
    """ The relational operator that's used to compare a value against the constraint 
    target value """
    target: float
    """ The constraint target """

class VariableType(Enum):
    """
    The type of variable
    """

    Real = "real"
    Integer = "integer"
    Boolean = "boolean"
    Choice = "choice"

class Variable:
    """
    A variable set on the solved problem.
    """

    name: str
    """ The variable name """
    var_type: VariableType
    """ An enumerator class identifying the type of variable """
    min_value: float | None
    """ The minimum bound. This is None if the variable
    does not support bounds. """
    max_value: float | None
    """ The maximum bound. This is None if the variable
    does not support bounds. """

class Problem:
    """
    Class holding information about the solved problem.
    """

    objectives: dict[str, Objective]
    """ The problem objectives. The list contains classes of Objective 
    instances that describe how each objective was configured. """
    constraints: dict[str, Constraint]
    """ The problem constraints. The list contains classes of Constraint 
    instances that describe how each constraint was configured. This 
    is an empty list if no constraints were defined in the solved problem. """
    variables: dict[str, Variable]
    """ The problem variables. The list contains classes of Variable instances
    that describe the type of each variable and how this was configured."""
    constraint_names: list[str]
    """ The constraint names. """
    variable_names: list[str]
    """ The variable names. """
    objective_names: list[str]
    """ The objective names. """
    number_of_objectives: int
    """ The number of objectives """
    number_of_constraints: int
    """ The number of constraints. """
    number_of_variables: int
    """ The number of variables. """

type VariableType = float | int | bool | str
type DataType = float | int | list[DataType] | dict[str, DataType]

class Individual:
    """
    An individual in the population containing the problem solution, and the objective
    and constraint values.
    """

    variables: dict[str, VariableType]
    """ A dictionary with the variable names and values for the individual """
    objectives: dict[str, float]
    """ A dictionary with the objective names and values for the individual """
    constraints: dict[str, float]
    """ A dictionary with the constraint names and values for the individual """
    constraint_violation: float
    """  The overall amount of violation of the solution constraints. This is a measure
     about how close (or far) the individual meets the constraints. If the solution is feasible,
     then the violation is 0.0. Otherwise, a positive number is stored.
    """
    is_feasible: bool
    """ Whether the solution meets all the problem constraints """
    data: dict[str, DataType]
    """ Additional numeric data stores in the individuals (such as crowding distance or rank)
     depending on the algorithm the individuals are derived from """

    def get_objective_value(self, name: str) -> float:
        """
        Get the objective value by name. This returns an error if the objective does not exist.
        :param name: The objective name.
        :return: The objective value.
        """

    def get_constraint_value(self, name: str) -> float:
        """
        Get the constraint value by name. This return an error if the constraint name does not exist.
        :param name: The constraint name.
        :return: The constraint value.
        """

    def get_variable_value(self, name: str) -> VariableType:
        """
        Get the variable value by name. This return an error if the variable name does not exist.
        :param name: The variable name.
        :return: The variable value. The type depends how the variable was configured when the
        problem was optimised.
        """

class AlgorithmData:
    """
    Class holding the algorithm data.
    """

    problem: Problem
    """ The problem. This class holds information about the solved problem. """
    generation: int
    """ The generation the export was collected at """
    algorithm: str
    """ Get the algorithm name used to evolve the individuals """
    took: timedelta
    """ The time took to reach the generation """
    individuals: list[Individual]
    """ The list with the individuals. An individual in the population contains
     the problem solution, and the objective and constraint values. """
    objectives: dict[str, list[float]]
    """ The objective values grouped by objective name """
    additional_data: dict[str, DataType] | None
    """ Any additional data exported by the algorithm (such as the distance to
    the reference point for NSGA3) """
    exported_on: datetime
    """  The date and time when the parsed JSON file was exported """

    def __init__(self, file: str):
        """
        Initialise the NSGA2 file reader.
        :param file: The path to the JSON file exported from optirustic Rust library.
        """

    def hyper_volume(self, reference_point: list[float]) -> float:
        """
        Calculate the exact hyper-volume metric. Depending on the number of problem
        objectives, a different method is used to ensure a correct and fast calculation:
         - with 2 objectives: by calculating the rectangle areas between each objective
           point and the reference point.
         - with 3 objectives: by using the algorithm proposed by Fonseca et al. (2006)
           (https://dx.doi.org/10.1109/CEC.2006.1688440).
         - with 4 or more objectives:  by using the algorithm proposed by While et al.
           (2012) (https://dx.doi.org/10.1109/TEVC.2010.2077298).
        :param reference_point: The reference or anti-optimal point to use in the
        calculation. If you are not sure about the point to use you could pick the worst
        value of each objective from the individual's values. The size of this point must
        match the number of problem objectives.
        :return: The hyper volume.
        """

    def estimate_reference_point(self, offset: list[float | None]) -> list[float]:
        """
        Calculates a reference point by taking the maximum of each objective (or minimum
        if the objective is maximised) from the calculated individual's objective
        values, so that the point will be dominated by all other points. An optional
        offset for all objectives could also be added or removed to enforce strict
        dominance (if the objective is minimised the offset is added to the calculated
        reference point, otherwise it is subtracted).
        :param offset: The offset to add to each objective coordinate of the calculated
        reference point. This must have a size equal to the number of objectives in the
        problem (`self.problem.number_of_objectives`).
        :return: The reference point. This returns an error if there are no individuals
        or the size of the offset does not match `self.problem.number_of_objectives`.
        """

    @staticmethod
    def estimate_reference_point_from_files(
        folder: str, offset: list[float | None]
    ) -> list[float]:
        """
        Calculates a reference point by taking the maximum of each objective (or minimum
        if the objective is maximised) from the objective values exported in a JSON
        files. This may be use to estimate the reference point when convergence is being
        tracked and one dominated reference point is needed.
        :param folder: The path to the folder with the JSON files.
        :param offset: The offset to add to each objective coordinate of the calculated
        reference point. This must have a size equal to the number of objectives in the
        problem (`self.problem.number_of_objectives`).
        :return: The reference point. This returns an error if there are no individuals,
        the folder does not exist or the size of the offset does not match
        `self.problem.number_of_objectives`.
        """

    def plot(self) -> plt.Figure:
        """
        Plot the Pareto front with the objective values. With 2 or 3 objectives, a 2D or
        3D chart is rendered respectively. With multi-objective problem a parallel
        coordinate chart is generated.
        This function only generates the matplotlib chart; you can manipulate the figure,
        save it (using `self.plot().savefig("figure.png")`) or show it (using `plt.show()`).
        :return: The `matplotlib`  figure object.
        """

    @staticmethod
    def convergence_data(
        folder: str, reference_point: list[float]
    ) -> tuple[list[int], list[datetime], list[float]]:
        """
        Calculate the hyper-volume at different generations (using the serialised
        objective values in JSON files exported at different generations).
        :param folder: The folder with the JSON files exported by the algorithm.
        :param reference_point: The reference or anti-optimal point to use in the
        calculation. The size of this point must match the number of problem objectives
        and must be dominated by all objectives at all generations.
        :return: A tuple containing the list of generation numbers, datetime objects,
        when the JSOn files were exported, and the hyper-olume values.
        """

    @staticmethod
    def plot_convergence(folder: str, reference_point: list[float]) -> plt.Figure:
        """
        Calculate the hyper-volume at different generations (using the serialised
        objective values in JSON files exported at different generations) and shows
        a convergence chart.
        :param folder: The folder with the JSON files exported by the algorithm.
        :param reference_point: The reference or anti-optimal point to use in the
        calculation. The size of this point must match the number of problem objectives
        and must be dominated by all objectives at all generations.
        :return: The figure object.
        """

class NSGA2(AlgorithmData):
    """
    Class to parse data exported with the NSGA2 algorithm.
    """

    pass

class NSGA3(AlgorithmData):
    """
    Class to parse data exported with the NSGA3 algorithm.
    """

    def plot_reference_points(self, reference_points: list[list[float]]) -> plt.Figure:
        """
        Generate a chart showing the reference point locations used by the algorithm and
        generated with the Das & Darren (2019) method.
        :param reference_points: The reference points.
        :return: The figure object.
        """

class TwoLayerPartitions(TypedDict):
    boundary_layer: int
    """ This is the number of partitions to use in the boundary layer. """
    inner_layer: int
    """ This is the number of partitions to use in the inner layer. """
    scaling: float | None
    """ Control the size of the inner layer. This defaults to 0.5 which means that the
    maximum points on each objectives axis will be located at 0.5 instead of 1 (as in 
    the boundary layer). """

class DasDarren1998:
    """
    Derive the reference points or weights using the methodology suggested in Section 5.2 in the
    Das & Dennis (1998) paper (https://doi.org/10.1137/S1052623496307510)
    """

    def __init__(
        self, number_of_objectives: int, number_of_partitions: int | TwoLayerPartitions
    ):
        """
        Derive the reference points or weights using the methodology suggested by
        Das & Dennis (1998).
        :param number_of_objectives: The number of problem objectives.
        :param number_of_partitions: The number of uniform gaps between two consecutive
        points along all objective axis on the hyperplane. With this option you can
        create one or two layer of points with different spacing: To create:
          - 1 layer or set of points with a constant uniform gaps use a 'int'.
          - 2 layers of points with each layer having a different gap use a
            dictionary with the following keys: inner_layer (the number of partitions
            to use in the inner layer), boundary_layer (the number of partitions to
            use in the boundary layer) and scaling (to control the size of the inner)
            layer. This defaults to 0.5 which means that the maximum points on each
            objectives axis will be located at 0.5 instead of 1 (as in the boundary layer
        Use the 2nd approach if you are trying to solve a problem with many objectives
        (4 or more) and want to reduce the number of reference points to use. Using two
        layers allows  (1) setting a smaller number of reference points, (2) controlling
        the point density in the inner area and (3) ensure a well-spaced point distribution.
        """

    def calculate(self) -> list[list[float]]:
        """
        Generate the vector of weights of reference points.
        :return: The vector of weights of size `number_of_points`. Each  nested list,
        of size equal to `number_of_objectives`, contains the relative coordinates
        (between 0 and 1) of the points for each objective.
        """

    @staticmethod
    def plot(reference_points: list[list[float]]) -> plt.Figure:
        """
        Generate a chart showing the reference point locations (for example using the Das
        & Darren (2019) method).
        :param reference_points: The reference points.
        :return: The figure object.
        """
