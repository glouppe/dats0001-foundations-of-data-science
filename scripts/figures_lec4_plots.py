"""Regenerate the lecture 4 plots made from the penguin data.

The graphical models are drawn by scripts/figures_lec4.py; this file holds the
figures that come from `data/`, with the same models as `nb04a` and `nb04b` and
at the resolution the slides need.

Usage: uv run python scripts/figures_lec4_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

GREY = "#354046"
BLUE = "#0173b2"
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
COLOR = {"Adelie": "#0173b2", "Chinstrap": "#de8f05", "Gentoo": "#029e73"}
COLUMNS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

plt.rcParams.update({"font.size": 11, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": "#b8c0c6"})


def load():
    """The four numeric columns and the species, as in nb04a and nb04b."""
    df = pd.read_csv("data/penguins.csv")[COLUMNS + ["species"]].dropna()
    return df[COLUMNS].to_numpy(dtype=np.float32), df["species"].to_numpy()


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def body_mass_histogram(data):
    """The normal model fitted to body mass, as in nb04a."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    mass = data[:, 3]
    ax.hist(mass, bins=20, density=True, alpha=.45, color=GREY, label=r"$p_r(x)$")
    mean, std = mass.mean(), mass.std()
    x = np.linspace(mean - 3 * std, mean + 3 * std, 200)
    ax.plot(x, norm(loc=mean, scale=std).pdf(x), color=BLUE, lw=2,
            label=r"$\mathcal{N}(x \mid \mu, \sigma^2)$")
    ax.set_xlabel("Body mass (g)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, "figures/lec4/body_mass_histogram.png")


def prior_predictive_samples(seed=0):
    """Body masses simulated from the prior, as in nb04a."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(5000, 2000, size=10000)
    sigma2 = rng.uniform(0, 100, size=10000)
    x = rng.normal(mu, np.sqrt(sigma2))

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    ax.hist(x, bins=20, density=True, alpha=.45, color=GREY, label=r"$p(x)$")
    ax.set_xlabel("Body mass (g)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, "figures/lec4/prior_predictive_samples.png")


def fit_ppca(X, m):
    """Maximum likelihood probabilistic PCA, as in nb04b."""
    mu = np.mean(X, axis=0)
    S = np.cov(X - mu, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    sigma2 = np.mean(eigenvalues[m:])
    B = eigenvectors[:, :m] @ np.sqrt(np.diag(eigenvalues[:m] - sigma2))
    return B, mu, sigma2


def project_ppca(X, B, mu, sigma2):
    """The posterior means of the latent variables, as in nb04b."""
    Sigma_inv = np.linalg.inv(B @ B.T + sigma2 * np.eye(B.shape[0]))
    return (B.T @ Sigma_inv @ (X - mu).T).T


def ppca_projections(data, species):
    """Each penguin in the two-dimensional latent space."""
    B, mu, sigma2 = fit_ppca(data, m=2)
    Z = project_ppca(data, B, mu, sigma2)

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200)
    for name in SPECIES:
        mask = species == name
        ax.scatter(Z[mask, 0], Z[mask, 1], s=22, alpha=.75, label=name, color=COLOR[name])
    ax.set_xlabel(r"$z_{i1}$")
    ax.set_ylabel(r"$z_{i2}$")
    ax.legend(title="Species")
    save(fig, "figures/lec4/ppca_projections.png")


def bill_clustering(data, species, seed=0):
    """A three-component Gaussian mixture fitted to the bill measurements.

    Each component is drawn in the colour of the species it mostly contains, as
    the mixture recovers the three species without ever being shown them.
    """
    X = data[:, :2]
    model = GaussianMixture(n_components=3, random_state=seed).fit(X)
    labels = model.predict(X)
    dominant = [pd.Series(species[labels == k]).mode()[0] for k in range(3)]
    order = sorted(range(3), key=lambda k: SPECIES.index(dominant[k]))

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    for rank, k in enumerate(order):
        color = COLOR[dominant[k]]
        ax.scatter(*X[labels == k].T, s=22, alpha=.6, color=color,
                   label="$k = %d$" % (rank + 1))
        values, vectors = np.linalg.eigh(model.covariances_[k])
        angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
        for scale in (1, 2):
            ax.add_patch(Ellipse(model.means_[k], *(2 * scale * np.sqrt(values[::-1])),
                                 angle=angle, facecolor="none", edgecolor=color, lw=1.2))
        ax.plot(*model.means_[k], "o", ms=7, color=color, markeredgecolor="white")
    ax.set_xlabel("Bill length (mm)")
    ax.set_ylabel("Bill depth (mm)")
    ax.legend(title="Component")
    save(fig, "figures/lec4/bill-clustering.png")


if __name__ == "__main__":
    data, species = load()
    body_mass_histogram(data)
    prior_predictive_samples()
    ppca_projections(data, species)
    bill_clustering(data, species)
