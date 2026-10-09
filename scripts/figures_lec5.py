"""Figures of lecture 5: the two graphical models, redrawn in the style of
lecture 4, and two plots comparing the continuous-time Lorenz system with its
discrete-time version, without noise (an ODE) and with noise (an SDE).

The drawing functions and constants come from figures_lec4.py, and the canvas
has the width of the lecture 4 figures, so that both lectures show their
graphical models at the same scale with a single .width-NN class.

Usage: uv run python scripts/figures_lec5.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from figures_lec4 import (FS, GREY, MARGIN, OBSERVED, R, STEP, UNIT, arrow, box,  # noqa: E402
                          figure, measure, node, plate, rescale)

WIDTH = 246.5568 / 72 / UNIT    # canvas width of the lecture 4 figures, in drawing units


def static():
    """A static latent variable model: theta points at z_i and at x_i."""
    fig, ax = figure()
    node(ax, (0, STEP), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    observations = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    theta = (observations[2] + .34 + R, STEP)
    node(ax, theta, r"$\theta$")
    arrow(ax, theta, (0, STEP))
    arrow(ax, theta, (0, 0))
    return fig, ax, "figures/lec5/lvm.svg"


def chain():
    """A state-space model: the states form a Markov chain, each emits an observation."""
    fig, ax = figure()
    gap = 1.5                                   # tighter than COL, to fit the canvas
    xs = [-gap, 0, gap]
    for x, t in zip(xs, ["t-1", "t", "t+1"]):
        for y, v in ((STEP, "z"), (0, "x")):
            ax.add_patch(Circle((x, y), R, facecolor=OBSERVED if v == "x" else "white",
                                edgecolor=GREY, lw=1.15, zorder=2))
            # time indices are wider than i: a smaller label keeps them inside the node
            ax.text(x, y, r"$\mathbf{%s}_{%s}$" % (v, t), ha="center", va="center",
                    zorder=3, fontsize=FS - 3)
        arrow(ax, (x, STEP), (x, 0))
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a, STEP), (b, STEP))
    for side in (-1, 1):
        dots = (side * (gap + 1.0), STEP)
        ax.text(*dots, r"$\cdots$", ha="center", va="center", fontsize=FS)
        if side < 0:
            arrow(ax, dots, (xs[0], STEP), r_start=.32)
        else:
            arrow(ax, (xs[-1], STEP), dots, r_end=.32)
    return fig, ax, "figures/lec5/sm.svg"


BLUE = "#0173b2"
LIGHT = "#b8c0c6"

# Lorenz system: a chaotic flow, whose trajectories fill a butterfly-shaped attractor.
S, RHO, B = 10.0, 28.0, 8 / 3
T, DT, FINE, SIGMA = 8.0, .01, 1e-4, 3.0      # horizon, discrete step, simulation step, noise


def drift(z):
    return np.array([S * (z[1] - z[0]), z[0] * (RHO - z[2]) - z[1], z[0] * z[1] - B * z[2]])


def lorenz(noise):
    """The Lorenz system and its discrete-time model with step DT, in 3d and z1 over time.

    The continuous path is computed by a high-order solver at tight tolerance without
    noise (its error stays negligible over T, even under chaos), and by Euler-Maruyama
    on a grid 100 times finer than DT with noise, driven by the same Brownian path."""
    ode = dict(method="DOP853", rtol=1e-12, atol=1e-12)
    z0 = solve_ivp(lambda t, z: drift(z), (0, 5), [1.0, 1.0, 1.0], **ode).y[:, -1]  # burn-in
    rng = np.random.default_rng(0)
    n = int(round(T / FINE))
    dw = rng.normal(0, np.sqrt(FINE), (n, 3)) * (SIGMA if noise else 0)
    if noise:
        z = np.empty((n + 1, 3))
        z[0] = z0
        for k in range(n):                    # Euler-Maruyama on a very fine grid
            z[k + 1] = z[k] + drift(z[k]) * FINE + dw[k]
    else:
        z = solve_ivp(lambda t, z: drift(z), (0, T), z0, t_eval=np.arange(n + 1) * FINE,
                      **ode).y.T
    every = int(round(DT / FINE))
    w = np.add.reduceat(dw, np.arange(0, n, every), axis=0)   # the same noise, per step
    zd = np.empty((len(w) + 1, 3))
    zd[0] = z0
    for k in range(len(w)):
        zd[k + 1] = zd[k] + drift(zd[k]) * DT + w[k]
    t, td = np.arange(n + 1) * FINE, np.arange(len(zd)) * DT

    fig = plt.figure(figsize=(7.6, 3.0), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0], projection="3d")
    ax.plot(*z.T, color=GREY, lw=.45, label="continuous time")
    ax.plot(*zd.T, color=BLUE, lw=.45, alpha=.85, label=r"discrete time, $\Delta t = %g$" % DT)
    ax.view_init(18, -58)
    ax.set_box_aspect(None, zoom=1.25)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color(LIGHT)
    ax.set_xlabel("$z_1$", labelpad=-12), ax.set_ylabel("$z_2$", labelpad=-12)
    ax.set_zlabel("$z_3$", labelpad=-12)

    ax2 = fig.add_subplot(grid[1])
    ax2.plot(t, z[:, 0], color=GREY, lw=.8)
    ax2.plot(td, zd[:, 0], color=BLUE, lw=.8)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("$z_1$")
    ax2.set_xlim(0, T)
    for side in ("right", "top"):
        ax2.spines[side].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center",
               ncols=2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .9))
    path = "figures/lec5/%s.svg" % ("sde-discretization" if noise else "ode-discretization")
    fig.savefig(path, facecolor="white")
    gap = np.abs(z[::every, 0] - zd[:, 0])
    print("wrote", path, "trajectories part at t = %.2f" % td[np.argmax(gap > 5)])


ORANGE = "#de8f05"
STATES = ["resting", "foraging", "traveling"]
STATE_COLORS = ["#b8c0c6", "#029e73", "#0173b2"]


def tidy(ax, left=True):
    for side in ("right", "top") if left else ("right", "top", "left"):
        ax.spines[side].set_visible(False)


def timeline():
    """Which observations each inference problem conditions on, and which state it targets."""
    T, t, k = 10, 6, 2
    rows = [("prediction", "$p(\\mathbf{z}_{t+k} \\mid \\mathbf{x}_{1:t})$", t, t + k),
            ("filtering", "$p(\\mathbf{z}_t \\mid \\mathbf{x}_{1:t})$", t, t),
            ("smoothing", "$p(\\mathbf{z}_t \\mid \\mathbf{x}_{1:T})$", T, t)]
    fig, ax = plt.subplots(figsize=(5.8, 2.2), dpi=200)
    for r, (name, formula, used, target) in enumerate(rows):
        y = len(rows) - 1 - r
        for s_ in range(1, T + 1):
            ax.plot(s_, y, "o", ms=7, color=BLUE if s_ <= used else "white",
                    markeredgecolor=BLUE if s_ <= used else LIGHT, zorder=2)
        ax.plot(target, y, "o", ms=15, mfc="none", mec=ORANGE, mew=1.6, zorder=3)
        ax.text(0.2, y, name, ha="right", va="center", fontsize=10)
        ax.text(T + .8, y, formula, ha="left", va="center", fontsize=11)
    ax.set_xticks([1, t, t + k, T], ["$1$", "$t$", "$t+k$", "$T$"])
    ax.set_yticks([])
    ax.set_xlim(-2.3, T + 4.6)
    ax.set_ylim(-.6, len(rows) - .4)
    ax.set_xlabel("time")
    tidy(ax, left=False)
    fig.tight_layout()
    fig.savefig("figures/lec5/inference-problems.svg", facecolor="white")
    print("wrote figures/lec5/inference-problems.svg")


def predict_update():
    """One step of the Bayes filter in one dimension, with illustrative numbers."""
    m, P = 1.0, .02                       # filtering distribution at t - 1
    A, Q = .9, .03                        # transition
    x, R = .6, .04                        # observation and its noise variance
    m_pred, P_pred = A * m, A ** 2 * P + Q
    K = P_pred / (P_pred + R)
    m_new, P_new = m_pred + K * (x - m_pred), (1 - K) * P_pred
    z = np.linspace(-.2, 1.8, 600)
    pdf = lambda mean, var: np.exp(-.5 * (z - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
    fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
    ax.plot(z, pdf(m, P), color=LIGHT, lw=1.6, label="filtering at $t-1$")
    ax.plot(z, pdf(m_pred, P_pred), color=GREY, lw=1.4, ls=(0, (4, 2)), label="prediction")
    ax.plot(z, pdf(x, R), color=ORANGE, lw=1.4, label="likelihood of $x_t$")
    ax.plot(z, pdf(m_new, P_new), color=BLUE, lw=2, label="filtering at $t$")
    ax.axvline(x, color=ORANGE, lw=.8, ls=":")
    ax.set_xlabel("$z$")
    ax.set_yticks([])
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/lec5/predict-update.svg", facecolor="white")
    print("wrote figures/lec5/predict-update.svg: gain %.2f" % K)


def hmm():
    """A wolf switching between three behaviors, seen through its noisy speed, and the
    filtering and smoothing distributions of the forward-backward algorithm."""
    rng = np.random.default_rng(4)
    T = 200
    A = np.array([[.95, .04, .01], [.03, .93, .04], [.02, .06, .92]])
    mean, sd = np.array([.1, .7, 1.6]), np.array([.15, .3, .4])
    prior = np.array([1, 0, 0])
    z = np.empty(T, dtype=int)
    z[0] = 0
    for t in range(1, T):
        z[t] = rng.choice(3, p=A[z[t - 1]])
    x = rng.normal(mean[z], sd[z])
    like = np.exp(-.5 * ((x[:, None] - mean) / sd) ** 2) / sd
    filt = np.empty((T, 3))
    filt[0] = prior * like[0] / (prior * like[0]).sum()
    for t in range(1, T):                 # forward pass: the Bayes filter with sums
        f = like[t] * (A.T @ filt[t - 1])
        filt[t] = f / f.sum()
    smooth = np.empty((T, 3))
    smooth[-1] = filt[-1]
    for t in range(T - 2, -1, -1):        # backward pass: the Bayes smoother with sums
        pred = A.T @ filt[t]
        smooth[t] = filt[t] * (A @ (smooth[t + 1] / pred))
    acc_f, acc_s = (filt.argmax(1) == z).mean(), (smooth.argmax(1) == z).mean()

    fig, axes = plt.subplots(4, 1, figsize=(6.4, 4.4), dpi=200, sharex=True,
                             gridspec_kw=dict(height_ratios=[.35, 1, .8, .8]))
    tt = np.arange(T)
    for j in range(3):
        axes[0].fill_between(tt, 0, 1, where=z == j, step="mid", color=STATE_COLORS[j],
                             lw=0, label=STATES[j])
    axes[0].set_yticks([])
    axes[0].set_ylabel("true", rotation=0, ha="right", va="center", fontsize=10)
    axes[0].legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(.5, 1),
                   fontsize=9)
    axes[1].plot(tt, x, ".", color=GREY, ms=3)
    axes[1].set_ylabel("speed $x_t$", fontsize=10)
    for ax, probs, name in ((axes[2], filt, "filtering"), (axes[3], smooth, "smoothing")):
        ax.stackplot(tt, probs.T, colors=STATE_COLORS, lw=0, step="mid")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_ylabel(name, fontsize=10)
    axes[3].set_xlabel("$t$")
    axes[3].set_xlim(0, T - 1)
    for ax in axes:
        tidy(ax, left=ax is not axes[0])
    fig.tight_layout()
    fig.savefig("figures/lec5/hmm-wolf.svg", facecolor="white")
    print("wrote figures/lec5/hmm-wolf.svg: most probable state right %.0f%% (filtering), "
          "%.0f%% (smoothing)" % (100 * acc_f, 100 * acc_s))


def wolf_data():
    """The simulated GPS observations of nb05, reproduced with the same seed and calls."""
    np.random.seed(42)
    dt, Delta, T = .01, .25, 20.0
    n_steps, every = int(T / dt), int(Delta / dt)
    mu, kappa, sigma = np.zeros(2), np.array([.25, .25]), np.array([.1, .1])
    R = np.eye(2) * .2 ** 2
    z = np.zeros((n_steps + 1, 2))
    z[0] = [2.0, 1.0]
    for i in range(n_steps):
        dB = np.random.normal(size=2) * np.sqrt(dt)
        z[i + 1] = z[i] - kappa * (z[i] - mu) * dt + sigma * dB
    obs = z[np.arange(0, n_steps + 1, every)]
    x = obs + np.random.normal(size=obs.shape) @ np.sqrt(R)
    return x, Delta, mu, sigma[0], R, kappa[0]


def kalman_loglik(x, kappa, Delta, mu, sigma, R):
    """log p(x_1:T | kappa), one pass of the Kalman filter with the exact transition."""
    a = np.exp(-kappa * Delta)
    A, b = a * np.eye(2), (1 - a) * mu
    Q = sigma ** 2 / (2 * kappa) * (1 - a ** 2) * np.eye(2)
    m, P, ll = mu.copy(), np.eye(2), 0.0
    for xt in x:
        m, P = A @ m + b, A @ P @ A.T + Q
        S = P + R
        r = xt - m
        ll += -.5 * (r @ np.linalg.solve(S, r) + np.log(np.linalg.det(2 * np.pi * S)))
        K = P @ np.linalg.inv(S)
        m, P = m + K @ r, (np.eye(2) - K) @ P
    return ll


def likelihood_kappa():
    """The log-likelihood of the attraction strength kappa for the wolf data."""
    x, Delta, mu, sigma, R, kappa_true = wolf_data()
    kappas = np.linspace(.02, 1.0, 300)
    ll = np.array([kalman_loglik(x, k, Delta, mu, sigma, R) for k in kappas])
    best = kappas[ll.argmax()]
    fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
    ax.plot(kappas, ll, color=BLUE, lw=1.8)
    ax.axvline(kappa_true, color=GREY, lw=1, ls=(0, (4, 3)))
    ax.text(kappa_true + .015, ll.min() + .1 * (ll.max() - ll.min()), "true $\\kappa$",
            fontsize=10)
    ax.plot(best, ll.max(), "o", color=BLUE, ms=5)
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("$\\log p(\\mathbf{x}_{1:T} \\mid \\kappa)$")
    ax.set_xlim(kappas[0], kappas[-1])
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/lec5/likelihood-kappa.svg", facecolor="white")
    print("wrote figures/lec5/likelihood-kappa.svg: maximum at kappa = %.3f (true %.2f)"
          % (best, kappa_true))


def assimilation(members=40, obs_every=.25, obs_sd=2.0, horizon=12.0):
    """Data assimilation on the Lorenz system: only z1 is observed, every obs_every time
    units with noise; an ensemble Kalman filter tracks the unobserved z3, while a forecast
    from the same uncertain start, without observations, loses it."""
    rng = np.random.default_rng(7)
    ode = dict(method="DOP853", rtol=1e-10, atol=1e-10)
    z0 = solve_ivp(lambda t, z: drift(z), (0, 5), [1.0, 1.0, 1.0], **ode).y[:, -1]
    t_obs = np.arange(obs_every, horizon + 1e-9, obs_every)
    grid = np.linspace(0, horizon, 3001)
    truth = solve_ivp(lambda t, z: drift(z), (0, horizon), z0, t_eval=grid, **ode).y.T
    truth_obs = solve_ivp(lambda t, z: drift(z), (0, horizon), z0, t_eval=t_obs, **ode).y.T
    y = truth_obs[:, 0] + rng.normal(0, obs_sd, len(t_obs))

    def step(z, dt=.01, n=None):          # RK4, the forecast model of every member
        for _ in range(n):
            k1 = drift(z); k2 = drift(z + dt / 2 * k1)
            k3 = drift(z + dt / 2 * k2); k4 = drift(z + dt * k3)
            z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    n_sub = int(round(obs_every / .01))
    start = z0 + rng.normal(0, 2.0, (members, 3))       # an uncertain initial state
    ens, free = start.copy(), start.copy()
    t_plot, mean, sd = [0.0], [ens.mean(0)], [ens.std(0)]
    free_mean, free_sd = [free.mean(0)], [free.std(0)]
    record = 5                             # record the forecasts every 0.05 time units
    for k, t in enumerate(t_obs):
        for r in range(n_sub // record):
            ens = np.array([step(e, n=record) for e in ens])
            free = np.array([step(e, n=record) for e in free])
            if r < n_sub // record - 1:
                t_plot.append(t - obs_every + (r + 1) * record * .01)
                mean.append(ens.mean(0)), sd.append(ens.std(0))
                free_mean.append(free.mean(0)), free_sd.append(free.std(0))
        ens = ens + rng.normal(0, .1, ens.shape)
        P = np.cov(ens.T)                  # update with the observation of z1 alone
        gain = P[:, 0] / (P[0, 0] + obs_sd ** 2)
        innovations = y[k] + rng.normal(0, obs_sd, members) - ens[:, 0]
        ens = ens + innovations[:, None] * gain[None, :]
        t_plot.append(t), mean.append(ens.mean(0)), sd.append(ens.std(0))
        free_mean.append(free.mean(0)), free_sd.append(free.std(0))
    t_plot, mean, sd, free_mean, free_sd = map(np.array, (t_plot, mean, sd, free_mean, free_sd))

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.6), dpi=200, sharex=True)
    for ax, j, name in ((axes[0], 0, "$z_1$, observed"), (axes[1], 2, "$z_3$, not observed")):
        ax.plot(grid, truth[:, j], color=GREY, lw=1.1, label="truth")
        ax.fill_between(t_plot, free_mean[:, j] - 2 * free_sd[:, j],
                        free_mean[:, j] + 2 * free_sd[:, j], color=ORANGE, alpha=.15, lw=0)
        ax.plot(t_plot, free_mean[:, j], color=ORANGE, lw=1, ls=(0, (4, 2)),
                label="forecast without observations")
        ax.fill_between(t_plot, mean[:, j] - 2 * sd[:, j], mean[:, j] + 2 * sd[:, j],
                        color=BLUE, alpha=.2, lw=0)
        ax.plot(t_plot, mean[:, j], color=BLUE, lw=1.3, label="with data assimilation")
        ax.set_ylabel(name, fontsize=10)
        tidy(ax)
    axes[0].plot(t_obs, y, "o", color=GREY, mfc="white", ms=3, mew=.8, zorder=4,
                 label="observations")
    axes[1].set_xlabel("$t$")
    axes[1].set_xlim(0, horizon)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=4, loc="upper center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig("figures/lec5/assimilation-lorenz.svg", facecolor="white")
    at_obs = np.isin(np.round(t_plot, 6), np.round(t_obs, 6))
    h = len(t_obs) // 2
    rmse = lambda a: np.sqrt(np.mean((a[at_obs][h:, 2] - truth_obs[h:, 2]) ** 2))
    print("wrote figures/lec5/assimilation-lorenz.svg: z3 error, second half: "
          "assimilation %.1f, free forecast %.1f" % (rmse(mean), rmse(free_mean)))


if __name__ == "__main__":
    lorenz(noise=False)
    lorenz(noise=True)
    timeline()
    predict_update()
    hmm()
    likelihood_kappa()
    assimilation()
    for fig, ax, path in [static(), chain()]:
        x0, y0, x1, y1 = measure(fig, ax)
        width = max(WIDTH, x1 - x0 + 2 * MARGIN)
        centre = (x0 + x1) / 2
        rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
        fig.savefig(path, facecolor="white")
        print("wrote %-24s %.2f x %.2f in" % (path, *fig.get_size_inches()))
