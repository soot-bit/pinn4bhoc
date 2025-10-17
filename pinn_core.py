# pinn_core.py
# -------------------------------------------------------------------------
import os
import csv
import time as tm

import numpy as np
import pandas as pd
from scipy.stats import qmc

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------------------------------------------------------
# Optional live display (works in Jupyter/Colab only)
# ---------------------------------------------------------
try:
    from IPython.display import clear_output

    HAS_CLEAR_OUTPUT = True
except ImportError:
    HAS_CLEAR_OUTPUT = False


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_elapsed_time(now_fn, start_time: float):
    """
    Computes and formats elapsed time.

    Parameters:
    - now_fn: function returning current time (e.g. time.time)
    - start_time: float timestamp from now_fn()

    Returns:
    - etime_str: formatted string "HH:MM:SS"
    - etime: total elapsed seconds
    - (h, m, s): tuple of hours, minutes, seconds
    """
    etime = now_fn() - start_time
    h = int(etime // 3600)
    m = int((etime % 3600) // 60)
    s = etime % 60
    return f"{h:02}:{m:02}:{int(s):02}", etime, (h, m, s)


def ensure_dir_exists(filepath):
    """
    Ensure the directory for the given filepath exists.
    If not, create it.
    """
    import os

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# --------------------------------------------------------------------------
# Sampling Strategies and custom Dataset
# --------------------------------------------------------------------------


class SobolSample:
    """
    Generates a Sobol sequence of points in a D-dimensional cube.

    The points are sampled quasi-randomly in the unit D-cube [0,1]^D,
    then scaled to the specified lower and upper bounds.

    Parameters:
        lower_bounds (array-like): Lower bounds for each dimension.
        upper_bounds (array-like): Upper bounds for each dimension.
        num_points_exp (int): Number of points is 2 ** num_points_exp.
        verbose (int): If nonzero, prints sample information.

    Attributes:
        sample (ndarray): Array of shape (2 ** num_points_exp, D),
            containing points in the bounded domain.
    """

    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        num_points_exp=16,  # of points = 2^num_points_exp
        verbose=1,
    ):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Generate Sobol points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        sampler = qmc.Sobol(d=D, scramble=True)
        self.sample = sampler.random_base2(m=num_points_exp)
        self.sample = qmc.scale(self.sample, lower_bounds, upper_bounds)

        if verbose:
            print(f"  SobolSample")
            print(f"  {2**num_points_exp} Sobol points created.")

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]


class UniformSample:
    """
    Generates a uniform random sample in a D-dimensional cube.

    The points are sampled uniformly in the unit D-cube [0,1]^D,
    then scaled to the specified lower and upper bounds.

    Parameters:
        lower_bounds (array-like): Lower bounds for each dimension.
        upper_bounds (array-like): Upper bounds for each dimension.
        num_points (int): Total number of points to generate.
        verbose (int): If nonzero, prints sample information.

    Attributes:
        sample (ndarray): Array of shape (num_points, D),
            containing points in the bounded domain.
    """

    def __init__(self, lower_bounds, upper_bounds, num_points, verbose=1):
        # Generate points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        self.sample = np.random.uniform(0, 1, D * num_points).reshape((num_points, D))
        self.sample = qmc.scale(self.sample, lower_bounds, upper_bounds)

        if verbose:
            print(f"  UniformSample")
            print(f"  {num_points} uniformly sampled points created.")

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]


class Dataset(Dataset):
    """
    Tensor dataset for PINN training from SobolSample or UniformSample.

    Takes the .sample ndarray (shape N x D) from sampling classes,
    and selects either a contiguous slice or a random subset. Splits data into:
      - `phi_vals`: first column, with gradient tracking,
      - `init_conds`: remaining columns.

    Parameters:
    ------------
    data : SobolSample or UniformSample
    start, end : int
        Index range for slicing `.sample`.
    random_sample_size : int, optional
        If set, draws this many random samples from the [start, end) range.
    device : torch.device
        Device to store tensors (default CPU).
    dtype : torch.dtype (default torch.float32)
        Data type of tensors.
    verbose : int
        Print tensor shapes and info if > 0.
    name : str, optional
        Optional name to tag this dataset instance (for logging/debugging).
    """

    def __init__(
        self,
        data,
        start,
        end,
        random_sample_size=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        verbose=1,
        name=None,
    ):
        super().__init__()

        self.name = name or "UnnamedDataset"
        self.verbose = verbose

        # Check that we have the right data types
        if not isinstance(data, (SobolSample, UniformSample)):
            raise TypeError(
                "/!\\ 'data' must be a sampling strategy instance "
                "(like SobolSample or UniformSample)."
            )

        if random_sample_size == None:
            tdata = torch.Tensor(data[start:end])
        else:
            # Create a random sample from items in the specified range (start, end)
            assert isinstance(random_sample_size, int)
            length = end - start
            assert length > 0
            indices = torch.randint(0, length - 1, size=(random_sample_size,))
            tdata = torch.Tensor(data[indices])

        self.phi_vals = tdata[:, 0].reshape(-1, 1).requires_grad_().to(device)
        self.init_conds = tdata[:, 1:].to(device)

        if verbose:
            print(f"  Type               : {self.__class__.__name__}")
            print(f"  Shape of phi_vals  : {self.phi_vals.shape}")
            print(f"  Shape of init_conds: {self.init_conds.shape}")

    def __len__(self):
        return len(self.phi_vals)

    def __getitem__(self, idx):
        return self.phi_vals[idx], self.init_conds[idx]


# -------------------------------------------------------------------------
# Model Definition
# -------------------------------------------------------------------------


class Sin(nn.Module):
    """Sine activation function for neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class FCNN(nn.Module):
    """Fully-connected neural network with sine activation function.

    The three input features are:
      - phi: the value of the solution at the current point,
      - q: the value of the solution at the initial points,
      - dphi/dx: the derivative of the solution with respect to x.

    Parameters:
        n_inputs (int): Number of input features.
        n_hidden (int): Number of hidden layers.
        n_width (int): Number of neurons per hidden layer.
        nonlinearity (nn.Module): Activation function (default Sin).
    """

    def __init__(self, n_inputs=3, n_hidden=3, n_width=75, nonlinearity=Sin):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_width = n_width

        # Build the MLP: input → hidden layers → output
        layers = [nn.Linear(n_inputs, n_width), nonlinearity()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_width, n_width), nonlinearity()]
        self.model = nn.ModuleList(layers)

        # Output layer (No nonlinearity needed)
        self.output_layer = nn.Linear(n_width, 1)

    def forward(self, phi_vals, init_conds):
        # Force shape to tensor [batchsize, 1] for phi values
        x = phi_vals.view(-1, 1)

        # Reshape initial conditions to [batchsize, n_init_conds] if necessary
        q = init_conds.repeat(len(x), 1) if init_conds.ndim == 1 else init_conds
        x = torch.concat((x, q), dim=1)

        # Pass through hidden layers
        for layer in self.model:
            x = layer(x)

        # Output layer
        out = self.output_layer(x)
        return out


# -------------------------------------------------------------------------
# Data Utilities
# -------------------------------------------------------------------------


class DataLoader:
    """
    Custom data loader that is much faster than the default PyTorch DataLoader.

    Notes:
    - If num_iterations is specified, it is assumed that this is the
      desired maximum number of iterations (maxiter) per for-loop.
      The flag shuffle is set to True, and an internal count, defined by

            shuffle_step = floor(len(dataset) / batch_size)

      is computed. The indices for accessing items from the dataset
      are shuffled every time the following condition is True:

            itnum % shuffle_step == 0,

      where itnum is an internal counter that keeps track of the iteration
      number
    - If num_iterations is not specified (the default), then
            maxiter = shuffle_step.

    - This data loader does not return the last batch if it is shorter
      than batch_size.
    """

    def __init__(
        self,
        dataset,
        batch_size=None,
        num_iterations=None,
        verbose=1,
        debug=0,
        shuffle=False,
    ):
        self.dataset = dataset
        self.size = batch_size
        self.niterations = num_iterations
        self.verbose = verbose
        self.debug = debug
        self.shuffle = shuffle

        if self.size is None:
            raise ValueError("You must specify batch_size")

        self.shuffle_step = int(len(dataset) / self.size)

        if self.niterations is not None:
            assert isinstance(self.niterations, int) and self.niterations > 0
            self.maxiter = self.niterations
            self.shuffle = True  # force shuffle mode

        elif len(self.dataset) > self.size:
            self.maxiter = self.shuffle_step

        else:
            # Note: this could be = 2 for a 2-tuple of tensors!
            self.size = len(self.dataset)
            self.shuffle_step = 1
            self.maxiter = self.shuffle_step

        if self.verbose:
            print("PINN DataLoader")
            if self.niterations is not None:
                print(
                    f"  ** Maximum number of iterations specified to: {self.niterations} **"
                )
            print(f"  maxiter:      {self.maxiter:10d}")
            print(f"  batch_size:   {self.size:10d}")
            print(f"  shuffle_step: {self.shuffle_step:10d}")

        assert self.maxiter > 0
        self.itnum = 0

    def __iter__(self):
        self.itnum = 0  # reset at start of new iteration loop
        return self

    def __next__(self):
        if self.itnum >= self.maxiter:
            raise StopIteration

        if self.shuffle:
            if self.itnum % self.shuffle_step == 0:
                if self.debug > 0:
                    print(f"PINN DataLoader/shuffling indices @ iteration {self.itnum}")
                self.indices = torch.randperm(len(self.dataset))

            start = (self.itnum % self.shuffle_step) * self.size
            end = start + self.size
            batch = self.dataset[self.indices[start:end]]
        else:
            start = self.itnum * self.size
            end = start + self.size
            batch = self.dataset[start:end]

        self.itnum += 1
        return batch

    def __len__(self):
        return self.maxiter


# ---------------------------------------------------------------------------
# PINN COMPONENTS
# ---------------------------------------------------------------------------


class Objective(nn.Module):
    """
    Defines the physics-based loss for the PINN.
    Computes the ODE residual: u'' + u - 1.5 * u²,
    and returns either the mean squared residual or the residuals themselves.
    """

    def __init__(self, solution, return_residuals=False):
        super().__init__()
        self.solution = solution
        self.return_residuals = return_residuals

    def eval(self):
        self.solution.eval()

    def train(self):
        self.solution.train()

    def save(self, paramsfile):
        self.solution.save(paramsfile)

    def forward(self, phi_vals, init_conds):
        # Assumes inputs come with shape (batch_size, D); no need to squeeze.

        # Compute u from the solution ansatz
        u = self.solution(phi_vals, init_conds)

        # First derivative du/dphi
        du = torch.autograd.grad(
            u, phi_vals, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        # Second derivative d²u/dphi²
        d2u = torch.autograd.grad(
            du, phi_vals, grad_outputs=torch.ones_like(du), create_graph=True
        )[0]

        # Residuals of the ODE
        residuals = d2u + u - 1.5 * u**2

        return residuals if self.return_residuals else torch.mean(residuals**2)


class Solution(nn.Module):
    """
    Solution ansatz using Theory of Connections (ToC) for 2nd-order ODEs:
        u(φ) = u₀ + g(φ) - g(0) + φ * (v₀ - dg/dφ|_{φ=0})

    Corresponding code variables:
        u     = u0 + gphi - g0 + phi * (v0 - gdot0)

    Enforces exact initial conditions:
        u(0)  = u0
        du/dφ|_{φ=0} = v0
    """

    def __init__(self, net):
        super().__init__()
        self.g = net  # FCNN model

    def train(self):
        self.g.train()

    def eval(self):
        self.g.eval()

    def save(self, dictfile):
        # Save model parameters
        torch.save(self.g.state_dict(), dictfile)

    def load(self, dictfile):
        # Load model parameters and set to eval mode
        self.g.load_state_dict(
            torch.load(dictfile, weights_only=True, map_location=torch.device("cpu"))
        )
        self.eval()

    def forward(self, phi, init_conds):
        """
        Applies the ToC-based ansatz to enforce initial/boundary conditions:
        u(φ) = u₀ + g(φ) - g(0) + φ * (v₀ - dg/dφ|_{φ=0})

        Assumes:
        - phi: shape (batch_size, 1)
        - init_conds: shape (batch_size, 2) or (2,)
        """

        # Zero input with gradient tracking for computing dg/dphi at phi=0
        zeros = torch.zeros_like(phi, requires_grad=True)
        g0 = self.g(zeros, init_conds)
        gphi = self.g(phi, init_conds)

        # Compute dg/dφ evaluated at φ=0
        gdot0 = torch.autograd.grad(
            g0, zeros, grad_outputs=torch.ones_like(g0), create_graph=True
        )[0]

        # Handle batched or unbatched initial conditions
        if init_conds.ndim == 1:
            u0 = init_conds[0].view(-1, 1)
            v0 = init_conds[1].view(-1, 1)
        else:
            u0 = init_conds[:, 0].view(-1, 1)
            v0 = init_conds[:, 1].view(-1, 1)

        # ToC: apply the ansatz to enforce exact boundary constraints
        u = u0 + gphi - g0 + phi * (v0 - gdot0)
        return u

    def diff(self, u, phi):
        """
        Computes du/dφ using autograd
        """
        du = torch.autograd.grad(
            u, phi, grad_outputs=torch.ones_like(phi), create_graph=True
        )[0]
        return du


# -------------------------------------------------------------------------
# Training Utilities
# -------------------------------------------------------------------------


def print_milestones_and_lrs(
    base_lr, n_steps, milestones, gamma, n_max_iterations=None
):
    """
    Print a table of learning rates and milestones for each training step.

    Parameters
    ----------
    base_lr : float
        Initial learning rate.
    n_steps : int
        Number of learning rate steps.
    milestones : list or array-like
        List of iteration numbers at which milestones occur.
    gamma : float
        Multiplicative factor for learning rate decay at each step.
    n_max_iterations : int, optional
        Total number of training iterations (for display only).
    """
    lrs = [base_lr * gamma**i for i in range(n_steps)]

    print("Step | Milestone | LR")
    print("-----------------------------")
    for i in range(n_steps):
        print(f"{i:>4} | {milestones[i]:>9} | {lrs[i]:<10.1e}")
        if i < 1:
            print("-----------------------------")

    if n_max_iterations is not None:
        print(f"\nTotal number of iterations: {n_max_iterations:10d}\n")


# -------------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------------


def train_pinn(
    train_loader,
    val_loader,
    train_valsize_loader,
    optimizer,
    scheduler,
    pinn_obj,
    display_costs=True,
    model_filename=None,
    log_filename=None,
    plot_filename=None,
    monitor_every_n_iterations=100,
    save_model=True,
    drop_threshold=0.005,
):
    """
    Train a Physics-Informed Neural Network (PINN).

    Parameters
    ----------
    train_loader : DataLoader
        Loader for the training dataset.
    val_loader : DataLoader
        Loader for the validation dataset.
    train_valsize_loader : DataLoader
        Loader for a fixed-size subset of the training data, used to compute
        training cost with matched statistics to validation.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler (default: None).
    pinn_obj : object
        PINN model object containing the network and loss function.
    display_costs : bool, default=True
        If True, plot cost curves during training (requires Jupyter/Colab).
    model_filename : str or None, default=None
        Path to save model weights. Ignored if None.
    log_filename : str or None, default=None
        Path to CSV log file. Ignored if None.
    plot_filename : str or None, default=None
        Path to save final training plots. Ignored if None.
    monitor_every_n_iterations : int, default=100
        Frequency (in iterations) to log and monitor costs.
    save_model : bool, default=True
        If True, save the model whenever validation cost significantly improves.
    drop_threshold : float, default=0.01
        Relative threshold (fractional drop) that validation cost must improve
        upon `best_val_cost` before saving the model.

    Returns
    -------
    None

    Notes:
        - Tracks training and validation costs, LR, and runtime.
        - Saves best model if validation cost improves.
        - Writes logs/plots if filenames are provided.
    """

    # Monitoring
    iterations = []
    train_costs = []
    val_costs = []
    best_val_costs = []
    lrs = []
    best_val_cost = None

    start_time = tm.time()

    if not model_filename:
        save_model = False
        print("Warning: Model filename not provided, model saving disabled.")
    else:
        ensure_dir_exists(model_filename)

    # --------------------
    #  Training Loop
    # --------------------
    for i, (phi_batch, init_conds_batch) in enumerate(train_loader):
        pinn_obj.train()

        optimizer.zero_grad()
        cost = pinn_obj(phi_batch, init_conds_batch)
        cost.backward()
        optimizer.step()

        # -----------------------
        # Learning rate update
        # -----------------------
        scheduler.step()

        # -----------------------
        # Real-time monitoring
        # -----------------------
        if i % monitor_every_n_iterations == 0:
            iterations.append(i)
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)

            # -----------------------
            # Compute costs
            # -----------------------
            pinn_obj.eval()

            train_cost = compute_avg_loss(pinn_obj, train_valsize_loader)
            train_costs.append(train_cost)

            val_cost = compute_avg_loss(pinn_obj, val_loader)
            val_costs.append(val_cost)

            # -----------------------
            # csv logging
            # -----------------------
            if log_filename:
                write_header = (
                    not os.path.isfile(log_filename)
                    or os.path.getsize(log_filename) == 0
                )
                with open(log_filename, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow(
                            [
                                "iteration",
                                "train_cost",
                                "val_cost",
                                "best_val_cost",
                                "lr",
                            ]
                        )
                    writer.writerow(
                        [i, train_cost, val_cost, best_val_cost, current_lr]
                    )

            # ---------------------------------------------------
            # Save model if validation cost significantly drops
            # ---------------------------------------------------
            if save_model and model_filename:
                if is_significant_drop_in_cost(val_cost, best_val_cost, drop_threshold):
                    # Save FNCC model (g here)
                    torch.save(pinn_obj.solution.g.state_dict(), model_filename)

            # Update best validation cost
            if best_val_cost is None or val_cost < best_val_cost:
                best_val_cost = val_cost

            best_val_costs.append(best_val_cost)

            # -----------------------
            # Live plotting
            # -----------------------
            if display_costs:
                if HAS_CLEAR_OUTPUT:
                    clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_cost_curves(
                    iterations, train_costs, val_costs, best_val_costs, lrs, ax
                )
                fig.tight_layout()
                plt.show()

            # -----------------------
            # Summary
            # -----------------------
            elapsed_str, elapsed_sec, _ = format_elapsed_time(tm.time, start_time)
            iteration_rate = (i + 1) / elapsed_sec

            print(
                f"[Iteration {i:8d}]  "
                f"LR: {current_lr:8.1e}  |  "
                f"Iter/s: {iteration_rate:5.1f}  |  "
                f"Time: {elapsed_str}"
            )

            print(
                f"   └── Cost [Train / Val / Best Val]:  "
                f"{train_cost:.3e}  /  {val_cost:.3e}  /  {best_val_cost:.3e}"
            )

    # -----------------------
    # End of training
    # -----------------------
    elapsed_str, elapsed_sec, _ = format_elapsed_time(tm.time, start_time)
    iteration_rate = len(train_loader) / elapsed_sec
    n_total_iterations = len(train_loader)

    print("\nEnd of training.\n")
    print(f"Total training time:    {elapsed_str:>10}")
    print(f"Average iteration rate: {iteration_rate:7.1f}/s")
    print(f"Total iterations:       {n_total_iterations:>10,}")

    if display_costs and plot_filename:
        fig.savefig(plot_filename)
        plt.close(fig)

    print("\nSaved files:")

    # Model weights
    if save_model:
        if model_filename is None:
            print(
                "Note: No model filename was provided; model weights were not saved.\n"
            )
        else:
            print(f"  - Model weights:  {model_filename}")

    # Training log
    if log_filename:
        print(f"  - Training log:   {log_filename}")

    # Cost plot
    if plot_filename:
        print(f"  - Cost plot:      {plot_filename}")


# -------------------------------------------------------------------------
# Evaluation Utilities
# -------------------------------------------------------------------------


def compute_avg_loss(objective, loader):
    """
    Compute the scalar cost from a single (phi, init_conds) batch.

    Parameters
    ----------
    objective : callable
        A function that maps (phi, init_conds) tensors to a scalar tensor
        representing the cost. Expected input shapes:
            - phi: Tensor of shape (N, 1)
            - init_conds: Tensor of shape (N, 2)
        where N is the batch size.

    loader : DataLoader
        A custom DataLoader expected to yield exactly one batch. This
        is typically enforced by setting the dataset size equal to the
        batch size.

    Returns
    -------
    float
        The scalar cost value, detached from the computation graph and
        moved to the CPU for logging or analysis.
    """
    # assert len(loader) == 1, "Loader must yield exactly one batch"

    for phi, init_conds in loader:
        # Detach from computation tree and send to CPU (if on a GPU)
        avg_loss = float(objective(phi, init_conds).detach().cpu())

    return avg_loss


def is_significant_drop_in_cost(val_cost, best_cost, drop_threshold=0.005):
    """
    Decide whether the current validation cost represents a significant drop.

    Parameters
    ----------
    val_cost : float
        Current validation cost (e.g., over the validation batch).

    best_cost : float or None
        Lowest validation cost observed so far. If None, assumes first evaluation.

    drop_threshold : float, optional
        Minimum relative improvement to count as significant. Default 0.5%.

    Returns
    -------
    bool
        True if val_cost improved by at least drop_threshold over best_cost,
        or if best_cost is None.
    """
    if best_cost is None:
        return True
    return val_cost < (1 - drop_threshold) * best_cost


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------


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


# Not used
def plot_grad_flow_violin(named_parameters, ax):
    """
    Plot gradient distribution per layer as a violin plot.
    Helps detect vanishing or exploding gradients.
    """

    grad_data = []

    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grads = param.grad.detach().cpu().flatten().numpy()
            grad_data.append((name, grads))

    flat_data = [(name, g) for name, grads in grad_data for g in grads]
    df = pd.DataFrame(flat_data, columns=["layer", "gradient"])

    sns.violinplot(
        data=df, x="layer", y="gradient", inner="quartile", density_norm="width", ax=ax
    )

    # Set axis labels and title
    ax.set_title("Gradient Flow per Layer")
    ax.set_xlabel("Layer")
    ax.tick_params(axis="x", rotation=45)

    # Plot scale on the right y-axis
    ax.set_ylabel("")  # Clear left y-axis
    ax.set_yticks([])

    ax_right = ax.twinx()
    ax_right.set_ylabel("Gradients", color="black")
    ax_right.tick_params(axis="y", labelcolor="black")
    ax_right.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax.grid(True, linestyle=":", linewidth=0.5, color="grey")
