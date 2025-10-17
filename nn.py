import torch
import torch.nn as nn


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
