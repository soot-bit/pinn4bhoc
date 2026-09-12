try:
    from IPython.display import clear_output

    HAS_CLEAR_OUTPUT = True
except ImportError:
    HAS_CLEAR_OUTPUT = False


def plot_cost_curves(iterations, train_costs, val_costs, best_val_costs, lrs, ax):
    """
    Plot training diagnostics (both left and right axes on log scale)

    - Left axis: Learning rate (log scale)
    - Right axis: Train, validation, and best-so-far validation costs (log scale)
    """
    fig = ax.get_figure()

    # x-axis
    ax.set_xlabel("Iterations")

    # Left y-axis: Learning rate
    ax.plot(iterations, lrs, color="green", linestyle="dashed", label="Learning Rate")
    ax.set_ylabel("Learning Rate", color="green")
    ax.tick_params(axis="y", labelcolor="green")
    ax.set_yscale("log")

    # Right y-axis: Cost
    ax_twin = ax.twinx()
    ax_twin.plot(iterations, train_costs, label="Train Cost", color="blue")
    ax_twin.plot(iterations, val_costs, label="Val Cost", color="red")
    ax_twin.plot(
        iterations, best_val_costs, label="Best Val Cost", color="red", linestyle="--"
    )
    ax_twin.set_ylabel("Cost", color="black")
    ax_twin.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_twin.tick_params(axis="y", labelcolor="black")
    ax_twin.set_yscale("log")

    # Title and legend
    ax.set_title(f"Training Progress - Iteration {iterations[-1]}")
    lines = ax.get_lines() + ax_twin.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        frameon=False,
    )
    ax.grid(True, linestyle=":", linewidth=0.5, color="grey")


def plot_cost_curves(iterations, train_costs, val_costs, best_val_costs, lrs, ax):
    """
    Plot training diagnostics (both left and right axes on log scale)

    - Left axis: Learning rate (log scale)
    - Right axis: Train, validation, and best-so-far validation costs (log scale)
    """
    fig = ax.get_figure()

    # x-axis
    ax.set_xlabel("Iterations")

    # Left y-axis: Learning rate
    ax.plot(iterations, lrs, color="green", linestyle="dashed", label="Learning Rate")
    ax.set_ylabel("Learning Rate", color="green")
    ax.tick_params(axis="y", labelcolor="green")
    ax.set_yscale("log")

    # Right y-axis: Cost
    ax_twin = ax.twinx()
    ax_twin.plot(iterations, train_costs, label="Train Cost", color="blue")
    ax_twin.plot(iterations, val_costs, label="Val Cost", color="red")
    ax_twin.plot(
        iterations, best_val_costs, label="Best Val Cost", color="red", linestyle="--"
    )
    ax_twin.set_ylabel("Cost", color="black")
    ax_twin.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_twin.tick_params(axis="y", labelcolor="black")
    ax_twin.set_yscale("log")

    # Title and legend
    ax.set_title(f"Training Progress - Iteration {iterations[-1]}")
    lines = ax.get_lines() + ax_twin.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        frameon=False,
    )
    ax.grid(True, linestyle=":", linewidth=0.5, color="grey")
