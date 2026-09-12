"""
Minimum PINN validation loss as a function of the training window dPhi.

Two ways to get the numbers behind the plot:

  - from training logs: each run writes a losses csv (see `train_pinn`), and
    the minimum of its `val_cost` column is that run's minimum validation loss.
    Several runs (seeds) may share the same dPhi.

  - from saved parameters: if the csv is not available, the saved model of a
    run is re-evaluated on freshly drawn uniform validation samples. Since
    models are saved on significant drops in validation cost, this is close to
    the minimum reached during training; the spread then reflects the
    validation draw, not the training seed.
"""

import csv
import glob
import os

import numpy as np
import yaml
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Reading runs
# ----------------------------------------------------------------------------
def find_run_dirs(root="runs", pattern="dphi*"):
    """
    Returns the sorted list of run directories under `root` matching `pattern`.
    """
    return sorted(d for d in glob.glob(os.path.join(root, pattern)) if os.path.isdir(d))
# ----------------------------------------------------------------------------
def load_run_config(run_dir):
    """
    Loads the single *_config.yaml found in `run_dir`.

    Returns:
        config (dict): contents of the yaml file, with the key 'run_dir' added.
    """
    matches = glob.glob(os.path.join(run_dir, "*_config.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"/!\\ expected exactly one *_config.yaml in {run_dir}, found {len(matches)}"
        )

    with open(matches[0], "r") as f:
        config = yaml.safe_load(f)

    config["run_dir"] = run_dir
    return config
# ----------------------------------------------------------------------------
def resolve_run_file(config, key, root="."):
    """
    Resolves the path of the file registered under config['file'][key].

    Paths in the config are relative to the project root, but a run directory
    may have been moved; the basename is looked up inside 'run_dir' as well.

    Returns:
        path (str) or None if the file cannot be found.
    """
    entry = config.get("file", {}).get(key)
    if entry is None:
        return None

    candidates = [
        os.path.join(root, entry),
        os.path.join(config["run_dir"], os.path.basename(entry)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    return None
# ----------------------------------------------------------------------------
def min_val_loss_from_csv(csv_path, column="val_cost"):
    """
    Returns the minimum of `column` over a losses csv written by `train_pinn`.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"/!\\ no rows in {csv_path}")

    if column not in rows[0]:
        raise KeyError(f"/!\\ no column '{column}' in {csv_path}")

    return min(float(row[column]) for row in rows if row[column] not in ("", None))
# ----------------------------------------------------------------------------
def eval_val_losses(config, params_path, repeats=5, val_size=None, seed=0, verbose=0):
    """
    Re-evaluates a saved model on freshly drawn uniform validation samples.

    The samples are drawn in the same domain as during training, that is
    between the bounds stored in the run config, and the validation cost is
    the mean squared ODE residual over one batch of `val_size` points.

    Parameters:
        config (dict): run config, see `load_run_config`.
        params_path (str): saved state dict of the FCNN.
        repeats (int): number of independent validation samples.
        val_size (int): points per sample (default: config['val_size']).
        seed (int): base seed; sample k uses seed + k.

    Returns:
        losses (list of float): one validation cost per sample.
    """
    # torch is only needed on this path, keep the import local
    import torch

    from ..nn import FCNN, Objective, Solution, compute_avg_loss
    from . import data as dat

    val_size = val_size or config["val_size"]

    solution = Solution(FCNN())
    solution.load(params_path)
    objective = Objective(solution)
    objective.eval()

    losses = []
    for k in range(repeats):
        np.random.seed(seed + k)
        sample = dat.UniformSample(
            config["lower_bounds"],
            config["upper_bounds"],
            num_points=val_size,
            verbose=0,
        )
        dataset = dat.Dataset(sample, start=0, end=val_size, verbose=0)
        loader = dat.DataLoader(dataset, batch_size=val_size, verbose=0)
        losses.append(compute_avg_loss(objective, loader))

    if verbose:
        print(f"  {os.path.basename(params_path)}: {np.mean(losses):.3e}")

    return losses
# ----------------------------------------------------------------------------
def collect_loss_vs_dphi(
    run_dirs=None,
    root="runs",
    pattern="dphi*",
    source="auto",
    repeats=5,
    val_size=None,
    seed=0,
    dphi_range=None,
    verbose=1,
):
    """
    Collects minimum validation losses of a set of runs, grouped by dPhi.

    Parameters:
        run_dirs (list of str): run directories (default: found under `root`).
        source (str): 'csv' uses the losses csv only, 'eval' re-evaluates the
            saved parameters only, 'auto' (default) prefers the csv and falls
            back to the saved parameters.
        repeats, val_size, seed: passed to `eval_val_losses`.
        dphi_range (tuple): optional (min, max) filter on dPhi, inclusive.

    Returns:
        raw (dict): {dPhi: [minimum validation loss, ...]}, one entry per run
            for the csv source, `repeats` entries per run otherwise.
    """
    run_dirs = run_dirs if run_dirs is not None else find_run_dirs(root, pattern)
    if not run_dirs:
        raise FileNotFoundError(f"/!\\ no run directories matching {root}/{pattern}")

    raw = {}
    for run_dir in run_dirs:
        config = load_run_config(run_dir)
        dPhi = float(config["dPhi"])

        if dphi_range is not None and not (dphi_range[0] <= dPhi <= dphi_range[1]):
            continue

        losses_path = resolve_run_file(config, "losses")
        params_path = resolve_run_file(config, "params")

        if source in ("csv", "auto") and losses_path:
            losses = [min_val_loss_from_csv(losses_path)]
            used = "csv"
        elif source in ("eval", "auto") and params_path:
            losses = eval_val_losses(config, params_path, repeats, val_size, seed)
            used = f"eval x{repeats}"
        else:
            if verbose:
                print(f"  /!\\ skipping {run_dir}: no {source} data")
            continue

        raw.setdefault(dPhi, []).extend(losses)

        if verbose:
            print(
                f"  dPhi = {dPhi:<5g} {config['name']:<10s} "
                f"{np.mean(losses):.3e}  ({used})"
            )

    if not raw:
        raise ValueError("/!\\ no runs contributed any losses")

    return dict(sorted(raw.items()))
# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def summarise(raw, band="sem"):
    """
    Summarises {dPhi: [losses]} in log space, where the losses are spread over
    orders of magnitude.

    Parameters:
        band (str): 'sem' (default), 'std' or 'minmax'.

    Returns:
        dphis, centre, lower, upper (ndarray): the geometric mean of the
            losses at each dPhi, and the edges of the requested band.
    """
    dphis = np.array(sorted(raw))
    logs = [np.log(np.asarray(raw[d], dtype=float)) for d in dphis]

    centre = np.array([lg.mean() for lg in logs])

    if band == "minmax":
        lower = np.array([lg.min() for lg in logs])
        upper = np.array([lg.max() for lg in logs])
    else:
        spread = np.array([lg.std(ddof=1) if len(lg) > 1 else 0.0 for lg in logs])
        if band == "sem":
            spread = spread / np.sqrt([max(len(lg), 1) for lg in logs])
        elif band != "std":
            raise ValueError(f"/!\\ unknown band '{band}'")
        lower, upper = centre - spread, centre + spread

    return dphis, np.exp(centre), np.exp(lower), np.exp(upper)
# ----------------------------------------------------------------------------
def plot_loss_vs_dphi(
    raw,
    name="loss_vs_dphi",
    save=False,
    outdir="figures",
    formats=("pdf",),
    band="sem",
    show_points=True,
    ax=None,
    color="tab:blue",
):
    """
    Plots the minimum PINN validation loss against dPhi.

    The line is the geometric mean of the runs at each dPhi, the shaded area
    the requested band (see `summarise`) and the grey dots the individual runs.

    Parameters:
        raw (dict): {dPhi: [losses]}, see `collect_loss_vs_dphi`.
        name (str): stem of the saved figure.
        save (bool): write the figure to `outdir`.
        formats (tuple of str): file formats to save.

    Returns:
        ax (matplotlib Axes)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    dphis, centre, lower, upper = summarise(raw, band=band)

    if show_points:
        for dPhi in dphis:
            losses = raw[dPhi]
            ax.plot(
                np.full(len(losses), dPhi),
                losses,
                linestyle="none",
                marker="o",
                markersize=4,
                color="grey",
                alpha=0.6,
                zorder=2,
            )

    ax.fill_between(dphis, lower, upper, color=color, alpha=0.2, zorder=1)
    ax.plot(dphis, centre, color=color, linewidth=2, zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta\phi$")
    ax.set_ylabel("minimum PINN validation loss")
    ax.tick_params(which="both", direction="in", top=True, right=True)

    fig = ax.get_figure()
    fig.tight_layout()

    if save:
        os.makedirs(outdir, exist_ok=True)
        for fmt in formats:
            path = os.path.join(outdir, f"{name}.{fmt}")
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved: {path}")

    return ax
# ----------------------------------------------------------------------------
def main(argv=None):
    """
    Command line entry point:

        python -m pinnslicer.utils.loss_vs_dphi --save
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", default="runs", help="directory holding the runs")
    parser.add_argument("--pattern", default="dphi*", help="run directory pattern")
    parser.add_argument("--source", default="auto", choices=("auto", "csv", "eval"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dphi-max", type=float, default=None)
    parser.add_argument("--band", default="sem", choices=("sem", "std", "minmax"))
    parser.add_argument("--name", default="loss_vs_dphi")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    dphi_range = None if args.dphi_max is None else (0.0, args.dphi_max)

    raw = collect_loss_vs_dphi(
        root=args.root,
        pattern=args.pattern,
        source=args.source,
        repeats=args.repeats,
        val_size=args.val_size,
        seed=args.seed,
        dphi_range=dphi_range,
    )

    plot_loss_vs_dphi(
        raw,
        name=args.name,
        save=args.save,
        outdir=args.outdir,
        band=args.band,
    )

    if not args.save:
        plt.show()

    return raw
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
