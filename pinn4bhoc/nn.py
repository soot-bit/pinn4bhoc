# ----------------------------------------------------------------------------
# Description: code associated with NN models
# Created: Mar 2025
# Updated: Mon Oct 20, 2025: move compute_avg_loss to nn.py from pinn_copy.py
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import csv
import yaml
from datetime import datetime
from pinn4bhoc.utils.data import ensure_dir_exists
from pinn4bhoc.utils.monitoring import plot_cost_curves
# ----------------------------------------------------------------------------
try:
    from IPython.display import clear_output

    HAS_CLEAR_OUTPUT = True
except ImportError:
    HAS_CLEAR_OUTPUT = False
# ----------------------------------------------------------------------------
def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
class Sin(nn.Module):
    """Sine activation function for neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
# ----------------------------------------------------------------------------
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

    def save(self, dictfile):
        # Save model parameters
        torch.save(self.state_dict(), dictfile)

    def load(self, dictfile):
        # Load model parameters and set to eval mode
        self.load_state_dict(
            torch.load(dictfile, weights_only=True, map_location=torch.device("cpu"))
        )
        self.eval()
# ----------------------------------------------------------------------------
class Solution(nn.Module):
    """
    Solution ansatz using Theory of Connections (ToC) for 2nd-order ODEs:
        u(φ) = u₀ + g(φ) - g(0) + φ * (v₀ - dg/dφ|_{φ=0})

    Corresponding code variables:
        u = u0 + gphi - g0 + phi * (v0 - gdot0)

    Enforces exact initial conditions:
        u(0) = u0
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
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
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

    start_time = time.time()

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
            elapsed_str, elapsed_sec, _ = format_elapsed_time(time.time, start_time)
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
    elapsed_str, elapsed_sec, _ = format_elapsed_time(time.time, start_time)
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
# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
class Config:
    '''
        Manage simple ML application configuration

          name:      name stub for all files, including the yaml file
          batchsize: 
          niter:     number of iterations
          base_lr:   base learning rate
          network:   network structure (n_hidden, n_width)
            :
          etc.
    '''
    def __init__(self, name, verbose=0):
        import time
        '''
        name:   name stub for all files, including the yaml file, or 
                the name of a yaml file. A json file is identified 
                by the extension .yaml
                
                    1. if name is a name stub, create a new yaml object.
                
                    2. if name is a yaml filename, create the yaml object
                       from the file.
        '''
        self.time = time.ctime()

        # create run folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        basedir = f"runs/{timestamp}"
        os.makedirs("runs", exist_ok=True)
        os.makedirs(basedir, exist_ok=True)
        
        # check if a yaml file has been specified
        if name.endswith('.yaml') or name.endswith('.yml'):
            self.cfg_filename = name # cache filename
            self.load(name)
        else:
            # this not a yaml file specification, assume it is a name stub
            # and build a Python dictionary that specifies the structure of
            # 
            self.cfg = {}
            cfg = self.cfg
            
            cfg['name'] = name
    
            # construct output file names    
            fcg = {}

            fcg['losses']     = f'{basedir}/{name}_losses.csv'
            fcg['params']     = f'{basedir}/{name}_params.pth'
            fcg['init_params']= f'{basedir}/{name}_init_params.pth'
            fcg['plots']      = f'{basedir}/{name}_plots.png'
            
            cfg['file'] = fcg
    
            # create a default name for yaml configuration file
            # this name will be used if a filename is not
            # specified in the save method
            self.cfg_filename = f'{basedir}/{name}_config.yaml'
    
        if verbose:
            print(self.__str__())
            
    def load(self, filename):
        # make sure file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f'{filename}')
        
        # read yaml file and cache as Python dictionary
        with open(filename, mode="r") as file:
            self.cfg = yaml.safe_load(file)

    def save(self, filename=None):
        # if no filename specified use default filename
        if filename == None:
            filename = self.cfg_filename

        # require .yaml extension
        if not (filename.endswith('.yaml') or filename.endswith('.yml')):
            raise NameError('the output file must have extension .yaml')
            
        # save to yaml file
        open(filename, 'w').write(self.__str__())
        
    def __call__(self, key, value=None):
        '''
        Return the value of the specified key.

        Notes
        -----
        1. If the key is in the dictionary and value is specified then 
        update the value of the key and return the value, otherwise 
        return the existing value of the key.

        2. If the key is not in the dictionary add it to the dictionary with
        the specified value and return the value. If no value is given raise 
        a KeyError exception.
        '''
        # this method can be used to fill out the rest
        # of the Python dictionary
        keys = key.split('/')
        
        # if key exists and value !=None update the value
        # else return its value
        cfg = self.cfg
        
        for ii, lkey in enumerate(keys):
            depth = ii + 1
            
            if lkey in cfg:
                # key is in dictionary
                
                val = cfg[lkey]
                if depth < len(keys):
                    # recursion
                    cfg = val
                else:
                    if type(value) == type(None):
                        # key exists and no value has been specified
                        # so return existing value
                        value = val
                    else:
                        # key exists and a value has been specified
                        # so update key and return new value
                        cfg[key] = value # update value
                    break
            else:
                # key is not in dictionary object, so add it
                
                if value == None:
                    # no value specified, so we can't add this key
                    raise KeyError(f'key "{lkey}" not found')
                    
                elif depth < len(keys):
                    cfg[lkey] = {}
                    cfg = cfg[lkey]
                else:
                    try:
                        cfg[lkey] = value
                    except:
                        pkey = keys[ii-1]
                        print(
                            f'''
    Warning: key '{key}' not created because '{pkey}' is 
    of type {str(type(pkey))}
                        ''')
        return value

    def __str__(self):
        # return a pretty printed string of the yaml object (help from ChatGPT)
        return str(yaml.dump(
            self.cfg,                 
            sort_keys=False,           # keep key order
            default_flow_style=False,  # use block style 
            indent=1,                  # indentation level
            allow_unicode=True))
    
