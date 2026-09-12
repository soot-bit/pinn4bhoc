# ----------------------------------------------------------------------------
# Description: code associated with data set handling.
# ----------------------------------------------------------------------------
from scipy.stats import qmc
import torch
import torch.utils.data as dt
import numpy as np
# ----------------------------------------------------------------------------
def ensure_dir_exists(filepath):
    """
    Ensure the directory for the given filepath exists.
    If not, create it.
    """
    import os

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
# ----------------------------------------------------------------------------
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
        self.sample = self.sample.astype(np.float32)

        if verbose:
            print(f"  SobolSample")
            print(f"  {2**num_points_exp} Sobol points created.")

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]
# ----------------------------------------------------------------------------
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
        self.sample = self.sample.astype(np.float32)

        if verbose:
            print(f"  UniformSample")
            print(f"  {num_points} uniformly sampled points created.")

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]
# ----------------------------------------------------------------------------
class Dataset(dt.Dataset):
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
            tdata = torch.tensor(data[start:end], dtype=dtype)
        else:
            # Create a random sample from items in the specified range (start, end)
            assert isinstance(random_sample_size, int)
            length = end - start
            assert length > 0
            indices = torch.randint(start, end - 1, size=(random_sample_size,))
            tdata = torch.tensor(data[indices], dtype=dtype)

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

# ---------------------------------------------------------------------------
# Custom DataLoader that is much faster than the default usage of the PyTorch
# DataLoader.
# ---------------------------------------------------------------------------
class DataLoader:
    '''
    A data loader that is much faster than the default PyTorch DataLoader.

    Notes:

       If num_iterations is specified, it is assumed that this is the
       desired maximum number of iterations, maxiter, per for-loop.
       The flag shuffle is automatically set to True and an internal
       count, defined by shuffle_step = floor(len(dataset) / batch_size)
       is computed. The indices for accessing items from the dataset
       are shuffled every time the following condition is True

           itnum % shuffle_step == 0,

       where itnum is an internal counter that keeps track of the iteration
       number. If num_iterations is not specified (the default), then
       the maximum number of iterations, maxiter = shuffle_step.

       This data loader, unlike the PyTorch data loader does not provide the
       option to return the last batch if the latter is shorter than batch_size.

       This class uses the Python generator pattern
    '''
    def __init__(self, dataset,
                 batch_size=None,
                 num_iterations=None,
                 verbose=1,
                 debug=0,
                 shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.debug   = debug
        self.shuffle = shuffle

        self.size = len(dataset)

        # need batch_size
        if self.batch_size is None:
            raise ValueError("you must specify a batch_size!")

        # if shuffle, then shuffle the dataset every shuffle_step iterations
        self.shuffle_step = max(1, self.size // self.batch_size)

        if self.verbose:
            print('DataLoader')

        if self.num_iterations is not None:
            if self.verbose:
                print('  Number of iterations has been specified')

            # the user has specified the number of iterations
            assert(type(self.num_iterations)==type(0))
            assert(self.num_iterations > 0)

            self.maxiter = self.num_iterations

            # IMPORTANT: shuffle indices every self.shuffle_step iterations
            self.shuffle = True  
            
        elif self.size > self.batch_size:
            self.maxiter = self.shuffle_step
            
        else:
            # Note: this could be = 2 for a 2-tuple of tensors!
            self.shuffle_step = 1
            self.maxiter = self.shuffle_step

        if self.verbose:
            print(f'  maxiter:      {self.maxiter:10d}')
            print(f'  batch_size:   {self.batch_size:10d}')
            print(f'  shuffle_step: {self.shuffle_step:10d}')
            print()

        assert(self.maxiter > 0)

        # initialize iteration number
        self.itnum = 0

        # initial indices for dataset (useful for debugging)
        self.indices = torch.arange(self.size)

    # ---------------------------------------------------------------------
    # This method implements the Python generator pattern.
    # The for loop
    #  for batch in loader:
    #          : :
    # is logically equivalent to:
    #
    #  iterator = iter(loader) # call __iter__(self) once
    #  while True
    #     try:
    #        batch = next(iterator) # which resumes execution at yield call
    #     except StopIteration:
    #        break
    # ---------------------------------------------------------------------
    def __iter__(self):

        self.itnum = 0
        while self.itnum < self.maxiter:

            if self.shuffle:
                # create a new tensor indexing dataset via a random
                # sequence of indices
                jtnum = self.itnum % self.shuffle_step
                if self.itnum > 0 and jtnum == 0:
                    self.indices = torch.randperm(self.size)
                    if self.debug > 0:
                        print(f'DataLoader shuffled indices @ index {self.itnum}')

                start   = jtnum * self.batch_size
                end     = start + self.batch_size
                indices = self.indices[start:end]
                batch   = self.dataset[indices]
            else:
                # create a new tensor directly indexing dataset
                start   = self.itnum * self.batch_size
                end     = start + self.batch_size
                batch   = self.dataset[start:end]

            # increment iteration number
            self.itnum += 1

            # pause function and return a value
            yield batch

    def __len__(self):
        return self.maxiter

