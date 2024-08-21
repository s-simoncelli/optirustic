import numpy as np
from matplotlib import pyplot as plt


def plot_2d(
    objectives: dict[str, list[float]], algorithm: str, generation: int, pop_size: int
) -> plt.Figure:
    """
    Plot the objective chart for a problem with 2 objectives.
    :param objectives: The dictionary with the objective names and values.
    :param algorithm: The algorithm name.
    :param generation: The generation number.
    :param pop_size: The population size.
    :return: The figure object.
    """
    fig = plt.figure()
    ax = plt.subplot()
    names = list(objectives.keys())
    values = list(objectives.values())

    ax.plot(values[0], values[1], "k.")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    plt.title(
        f"Results for {algorithm} @ generation={generation} \n"
        f"Population size={pop_size}"
    )

    return fig


def plot_3d(
    objectives: dict[str, list[float]],
    algorithm: str,
    generation: int,
    pop_size: int,
) -> plt.Figure:
    """
    Plot the objective chart for a problem with 2 objectives.
    :param objectives: The dictionary with the objective names and values.
    :param algorithm: The algorithm name.
    :param generation: The generation number.
    :param pop_size: The population size.
    :return: The figure object.
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    names = list(objectives.keys())
    values = list(objectives.values())

    ax.scatter(values[0], values[1], values[2], color="k", marker=".")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    plt.title(
        f"Results for {algorithm} @ generation={generation} \n"
        f"Population size={pop_size}"
    )

    return fig


def plot_parallel(
    objectives: dict[str, list[float]],
    algorithm: str,
    generation: int,
    pop_size: int,
):
    names = list(objectives.keys())
    values = list(objectives.values())
    fig = parallel_coordinate_plot(values, names)
    plt.title(
        f"Results for {algorithm} @ generation={generation} \n"
        f"Population size={pop_size}"
    )

    return fig


def parallel_coordinate_plot(
    data: list[list[float]],
    objective_names: list[str],
    color="red",
) -> plt.Figure:
    """
    Render a parallel coordinate plot. This code was edited from
    https://github.com/gregornickel/pcp.
    :param data: The data. Each list represents an individual and each nested
    list contains the objective values.
    :param objective_names: The objective names to use for the axis.
    :param color: The line colour.
    :return: The figure object.
    """
    # Check data
    if len(data) != len(objective_names):
        raise ValueError(
            f"The length of the objectives in data ({len(data)}) must "
            f"match the number of axis labels ({len(objective_names)})"
        )

    y_labels = [[]] * len(objective_names)
    y_type = ["linear"] * len(objective_names)
    data = np.array(data)  # .transpose()

    # Set the limits for each objective axis
    y_lim = [[]] * len(objective_names)
    for i in range(len(y_lim)):
        y_lim[i] = [np.min(data[i, :]), np.max(data[i, :])]
        if y_lim[i][0] == y_lim[i][1]:
            y_lim[i] = [y_lim[i][0] * 0.95, y_lim[i][1] * 1.05]
        if y_lim[i] == [0.0, 0.0]:
            y_lim[i] = [0.0, 1.0]

    # Rescale the data to apply transformation function and to scale each
    # axis appropriately.
    min0 = y_lim[0][0]
    max0 = y_lim[0][1]
    scale = max0 - min0
    for i in range(1, len(y_lim)):
        mini = y_lim[i][0]
        maxi = y_lim[i][1]
        if y_type[i] == "log":
            logmin = np.log10(mini)
            logmax = np.log10(maxi)
            span = logmax - logmin
            data[i, :] = ((np.log10(data[i, :]) - logmin) / span) * scale + min0
        else:
            data[i, :] = ((data[i, :] - mini) / (maxi - mini)) * scale + min0

    # Create figure
    fig = plt.figure()
    ax0 = fig.add_axes((0.125, 0.1, 0.75, 0.8))
    axes = [ax0] + [ax0.twinx() for _ in range(data.shape[0] - 1)]

    # Plot curves
    for i in range(data.shape[1]):
        ax0.plot(data[:, i], color=color, clip_on=False)

    # Format x-axis
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position("none")
    ax0.set_xlim([0, data.shape[0] - 1])
    ax0.set_xticks(range(data.shape[0]))
    ax0.set_xticklabels(objective_names)

    # Format y-axis
    for i, ax in enumerate(axes):
        ax.spines["left"].set_position(("axes", 1 / (len(objective_names) - 1) * i))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.yaxis.set_ticks_position("left")
        ax.set_ylim(y_lim[i])

        if y_labels[i]:
            ax.set_yticklabels(y_labels[i])

    return fig


def plot_convergence(generations: list[int], values: list[float]) -> plt.Figure:
    """
    Plot the hyper-volume as function of the generation number to track the algorithm
    convergence.
    :param generations: The generation numbers.
    :param values: The hyper-volume values.
    :return: The figure object.
    """
    fig = plt.figure()
    ax = plt.subplot()

    ax.plot(generations, values, "k.-")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hyper-volume")
    plt.title(f"Convergence from {min(generations)} to {max(generations)} generations")

    return fig


def plot_reference_points(reference_points: list[list[float]]) -> plt.Figure:
    if len(reference_points) == 0:
        raise ValueError("The reference point vector is empty")

    number_of_objectives = len(reference_points[0])
    names = [f"Objective #{i + 1}" for i in range(number_of_objectives)]
    fig = plt.figure()

    if number_of_objectives == 2:
        ax = plt.subplot()

        for point in reference_points:
            ax.plot(*point, "k.")
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
    elif number_of_objectives == 3:
        ax = plt.axes(projection="3d")
        for point in reference_points:
            ax.scatter(*point, color="k", marker=".")
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])
        ax.view_init(azim=10)
    else:
        parallel_coordinate_plot(reference_points, names, "black")

    plt.title(
        "Reference points - Das & Darren (2019)\n" f"{number_of_objectives} objectives"
    )
    return fig
