"""
Adversarial Robin: toy demonstrator.

Companion to the essay "The Adversarial Robin: In pursuit of an AI scientist"
by J. Babin.
code written by Claude Code

Conceit:
- A scalar "tacit" feature theta_true is the thing we want to recover.
- Each replicate's posterior over theta is multimodal: one true broad mode
  at theta_true, plus one or two narrow spurious modes at random locations.
  Spurious modes are *taller* in peak density (mimicking a sharp false
  attractor) but supported by much smaller mass than the true mode.
- "Worms" = a number of short Metropolis-Hastings chains with random starts
  and mixed temperatures, exploring the posterior. Cold chains lock into
  whichever local mode they started nearest. The ensemble of chains is the
  "compost".
- Without the robin: aggregate cold-chain samples, KDE the result, pick the
  highest-density peak. When a chain happens to land in a tall-narrow
  spurious mode, that peak can dominate the aggregate KDE even though only
  one chain contributed -- this is the failure case the essay diagnoses.
- With the robin (chain jackknife): repeatedly delete a random subset of
  chains and recompute the candidate-mode supports. A mode supported by
  many chains keeps at least one supporter through most deletions; a mode
  supported by a single chain collapses (zero supporters) whenever that
  chain is in the deleted subset. Score each candidate mode by the fraction
  of jackknife draws in which at least one supporting chain survives.

Across many replicates we plot the resulting theta estimates with and
without the robin. The robin cloud should concentrate near theta_true while
the no-robin cloud also lights up at the spurious mode locations.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


# ---------------------------------------------------------------- config
THETA_TRUE = 0.30
THETA_LO, THETA_HI = 0.05, 0.95

# True mode (broad, lots of mass) and spurious mode (narrow, tall peak)
TRUE_WIDTH_RANGE = (0.07, 0.10)
TRUE_WEIGHT_RANGE = (0.58, 0.75)
SPUR_WIDTH_RANGE = (0.012, 0.022)
SPUR_LOC_RANGE = (0.55, 0.90)
N_SPUR_CHOICES = (1, 2)  # each replicate has 1 or 2 spurious modes

# Chains: 10 cold + 4 hot. Only cold chains contribute to the aggregate KDE;
# the hot chains exist to echo the essay's "different temperatures" language
# (and would feed a parallel-tempering swap scheme in a fuller version).
N_COLD = 10
N_HOT = 4
COLD_TEMP = 1.0
HOT_TEMPS = (2.5, 2.5, 6.0, 6.0)

N_STEPS = 1800
BURN = 800            # keep last 1000 samples per chain
THIN = 5              # thin to 200 samples per chain -> faster KDE
PROP_SD = 0.030

JACK_DROP = 4         # robin removes this many cold chains per draw
JACK_DRAWS = 30       # number of jackknife draws per replicate

N_REPLICATES = 80
SEED = 11


# ------------------------------------------------------- log-posterior
def make_log_post(rng: np.random.Generator):
    """One replicate's log-posterior: broad true mode + 1-2 narrow spurious."""
    true_w = rng.uniform(*TRUE_WIDTH_RANGE)
    true_wt = rng.uniform(*TRUE_WEIGHT_RANGE)

    n_spur = int(rng.integers(N_SPUR_CHOICES[0], N_SPUR_CHOICES[1] + 1))
    spur_locs = []
    attempts = 0
    while len(spur_locs) < n_spur and attempts < 2000:
        s = float(rng.uniform(*SPUR_LOC_RANGE))
        if all(abs(s - l) > 0.08 for l in spur_locs):
            spur_locs.append(s)
        attempts += 1
    spur_locs = np.array(spur_locs)
    n_spur = len(spur_locs)
    spur_widths = rng.uniform(*SPUR_WIDTH_RANGE, size=n_spur)
    spur_weights = np.full(n_spur, (1.0 - true_wt) / n_spur)

    def log_post(theta):
        a = true_wt * np.exp(-0.5 * ((theta - THETA_TRUE) / true_w) ** 2) / true_w
        for loc, w, wt in zip(spur_locs, spur_widths, spur_weights):
            a = a + wt * np.exp(-0.5 * ((theta - loc) / w) ** 2) / w
        return np.log(a + 1e-300)

    return log_post, spur_locs


# --------------------------------------------------------------- worms
def run_all_chains(log_post, starts, temperatures, rng,
                   n_steps=N_STEPS, prop_sd=PROP_SD):
    """Vectorized Metropolis across many chains at once. log_post must be
    array-vectorized (returns array if given an array of thetas)."""
    starts = np.asarray(starts, dtype=float)
    temps = np.asarray(temperatures, dtype=float)
    n_chains = starts.size

    thetas = starts.copy()
    lls = log_post(thetas) / temps
    samples = np.empty((n_steps, n_chains))

    for i in range(n_steps):
        props = thetas + prop_sd * rng.standard_normal(n_chains)
        in_bounds = (props >= THETA_LO) & (props <= THETA_HI)
        prop_lls = np.where(in_bounds, log_post(props) / temps, -np.inf)
        accept = np.log(rng.random(n_chains) + 1e-300) < (prop_lls - lls)
        thetas = np.where(accept, props, thetas)
        lls = np.where(accept, prop_lls, lls)
        samples[i] = thetas
    return samples


def explore(log_post, rng):
    cold_starts = rng.uniform(THETA_LO, THETA_HI, size=N_COLD)
    hot_starts = rng.uniform(THETA_LO, THETA_HI, size=N_HOT)
    starts = np.concatenate([cold_starts, hot_starts])
    temps = np.concatenate([
        np.full(N_COLD, COLD_TEMP),
        np.array(HOT_TEMPS),
    ])
    all_samples = run_all_chains(log_post, starts, temps, rng)
    cold_chains = [
        all_samples[BURN::THIN, c]
        for c in range(N_COLD)
    ]
    return cold_chains


# --------------------------------------------------------------- modes
GRID = np.linspace(THETA_LO, THETA_HI, 600)

def kde_modes(samples: np.ndarray):
    if samples.size < 50:
        return np.array([]), np.array([])
    if np.std(samples) < 1e-4:
        loc = float(np.mean(samples))
        return np.array([loc]), np.array([1.0])
    try:
        kde = gaussian_kde(samples, bw_method=0.04)
        dens = kde(GRID)
    except np.linalg.LinAlgError:
        return np.array([float(np.mean(samples))]), np.array([1.0])
    peaks, _ = find_peaks(dens, height=dens.max() * 0.08)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(dens))])
    return GRID[peaks], dens[peaks]


def top_mode(samples: np.ndarray) -> float:
    locs, hts = kde_modes(samples)
    if locs.size == 0:
        return np.nan
    return float(locs[int(np.argmax(hts))])


# --------------------------------------------------------------- robin
def robin_filter(cold_chains, rng, tol=0.05):
    """
    The robin plucks worms (chains) at random. A mode whose presence depends
    on a single chain collapses when that chain is plucked; a mode supported
    by several chains survives.

    Each chain "votes" for whichever candidate mode lies nearest to its
    sample median. We then run JACK_DRAWS jackknife draws, dropping
    JACK_DROP chains each time, and ask: for each candidate, how often does
    at least one of its supporting chains survive the deletion? A mode
    supported by k chains survives a draw with probability
    1 - C(n-k, JACK_DROP) / C(n, JACK_DROP).
    """
    full_samples = np.concatenate(cold_chains)
    cand_locs, cand_hts = kde_modes(full_samples)
    if cand_locs.size == 0:
        return np.nan

    chain_votes = np.empty(len(cold_chains), dtype=int)
    for c, chain in enumerate(cold_chains):
        loc = float(np.median(chain))
        dists = np.abs(cand_locs - loc)
        nearest = int(np.argmin(dists))
        chain_votes[c] = nearest if dists[nearest] <= tol else -1

    n = len(cold_chains)
    keep_n = max(2, n - JACK_DROP)
    counts = np.zeros((JACK_DRAWS, cand_locs.size))

    for j in range(JACK_DRAWS):
        idx = rng.choice(n, size=keep_n, replace=False)
        kept_votes = chain_votes[idx]
        for k in range(cand_locs.size):
            counts[j, k] = float((kept_votes == k).sum())

    survives = (counts > 0).astype(float)
    survival_rate = survives.mean(axis=0)
    min_surviving_chains = counts.min(axis=0)

    score = (survival_rate
             + 0.01 * min_surviving_chains
             + 1e-4 * (cand_hts / cand_hts.max()))
    return float(cand_locs[int(np.argmax(score))])


# --------------------------------------------------------------- driver
def run():
    rng = np.random.default_rng(SEED)
    no_robin = np.empty(N_REPLICATES)
    with_robin = np.empty(N_REPLICATES)
    spur_records = []

    for r in range(N_REPLICATES):
        log_post, spur_locs = make_log_post(rng)
        cold_chains = explore(log_post, rng)
        full_samples = np.concatenate(cold_chains)

        no_robin[r] = top_mode(full_samples)
        with_robin[r] = robin_filter(cold_chains, rng)
        spur_records.append(spur_locs)

        print(f"  replicate {r+1:3d}/{N_REPLICATES}  "
              f"no_robin={no_robin[r]:.3f}  with_robin={with_robin[r]:.3f}  "
              f"spur={np.round(spur_locs, 2).tolist()}", flush=True)

    err_no = np.abs(no_robin - THETA_TRUE)
    err_yes = np.abs(with_robin - THETA_TRUE)
    fail_no = (err_no > 0.10).mean() * 100
    fail_yes = (err_yes > 0.10).mean() * 100

    print()
    print(f"Summary across {N_REPLICATES} replicates:")
    print(f"  no-robin  median |err|: {np.median(err_no):.3f}   "
          f"%(|err|>0.10): {fail_no:.1f}%")
    print(f"  with-robin median |err|: {np.median(err_yes):.3f}   "
          f"%(|err|>0.10): {fail_yes:.1f}%")

    plot(no_robin, with_robin, spur_records, fail_no, fail_yes)


def plot(no_robin, with_robin, spur_records, fail_no, fail_yes):
    rng = np.random.default_rng(0)
    jit_n = rng.uniform(-0.35, 0.35, size=no_robin.size)
    jit_r = rng.uniform(-0.35, 0.35, size=with_robin.size)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)

    panels = [
        (axes[0], no_robin, jit_n,
         f"Without robin  ({fail_no:.0f}% off-target)"),
        (axes[1], with_robin, jit_r,
         f"With robin  ({fail_yes:.0f}% off-target)"),
    ]
    for ax, ests, jit, title in panels:
        ax.axhline(THETA_TRUE, color="crimson", linestyle="--", linewidth=1.4,
                   label=fr"$\theta_{{\rm true}}={THETA_TRUE}$")
        ax.scatter(jit, ests, s=42, alpha=0.65, edgecolor="black",
                   linewidth=0.4, color="#3b6ea5", label="estimate")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_xlim(-1, 1)
        ax.set_ylim(THETA_LO, THETA_HI)
        ax.grid(alpha=0.25)

    spur_x = []
    spur_y = []
    for r, locs in enumerate(spur_records):
        for loc in locs:
            spur_x.append(0.78)
            spur_y.append(loc)
    axes[0].scatter(spur_x, spur_y, s=10, marker="_", alpha=0.35,
                    color="gray", label="spurious mode locations")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel(r"estimated $\theta$", fontsize=11)
    fig.suptitle(
        "Adversarial Robin: scalar latent recovery from multimodal posterior\n"
        f"{N_REPLICATES} replicates  -  {N_COLD} cold chains  -  "
        f"robin drops {JACK_DROP} chains x {JACK_DRAWS} draws",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(
        0.5, 0.012,
        "Toy demonstrator for 'The Adversarial Robin'  -  J. Babin  -  "
        "code: Claude (Anthropic)",
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )
    out = "robin_demo.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    run()
