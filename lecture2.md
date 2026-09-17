class: middle, center, title-slide

# Foundations of Data Science

Lecture 2: Data and exploratory analysis

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

---

class: middle

# Data

---

class: middle

.center.width-10[![](figures/lec2/height.png)]

## What is data?

Data are .bold[recorded observations] about the world.

They can take many forms, including numbers, text, images, and more. 

---

class: middle

## A running example

.center.width-60[![](figures/lec2/lter-penguins.png)]

We will follow one dataset through this lecture: the .bold[Palmer Archipelago penguins], 344 birds of three species measured on three islands between 2007 and 2009.

Someone stood in the cold, caught a penguin, measured its bill with a caliper and put it on a scale. What follows is about what such records are, and what they are not.

.footnote[Credits: [Horst et al.](https://allisonhorst.github.io/palmerpenguins/), 2020; artwork by @allison\_horst.]

---

class: middle

Mathematically, data can be viewed as a function $f$ that maps real-world entities $\omega$ to measurable values $x$,
$$f : \Omega \to \mathcal{X},$$
where
- $\Omega$ is the sample space of possible states $\omega$ of the world,
- $\mathcal{X}$ is the measurement space of possible observations $x$.

---

class: middle

Examples&#58;
- Penguin body mass: $\omega \in \\{ \text{penguins} \\} \to x \in \mathbb{R}^+$ (g)
- Stock price: $\omega \in \\{ \text{market states} \\} \to x \in \mathbb{R}^+$ (USD)
- Pixel colour: $\omega \in \\{ \text{scenes} \\} \to x \in \\{0, \ldots, 255\\}^3$ (RGB values)

---

class: middle

If the sample space $\Omega$ carries a probability distribution $p(\omega)$, then $f$ turns a random state of the world into a random observation $x = f(\omega)$. Its distribution is the one $f$ induces from $p$, called the .bold[data distribution] $p\_r(x)$, where $r$ stands for "real".

When $\mathcal{X}$ is continuous, it can be written as
$$p\_r(x) = \int\_{\omega \in \Omega} p(\omega) \delta(x - f(\omega)) d\omega,$$
where $\delta$ is the Dirac delta function. When $\mathcal{X}$ is discrete, $p\_r$ assigns probabilities rather than a density.

---

class: middle

The .bold[measurement process] is part of the data generation mechanism. We make it explicit by adding the measurement conditions $\xi \in \Xi$ (instrument settings, environmental conditions, observer effects) to the map,
$$f : \Omega \times \Xi \to \mathcal{X}.$$

Measurements can introduce quantization (continuous to discrete), noise (random perturbations), and bias (systematic deviations). If $\Omega \times \Xi$ carries a joint distribution $p(\omega, \xi)$, then
$$p\_r(x) = \iint\_{\omega \in \Omega, \xi \in \Xi} p(\omega, \xi) \delta(x - f(\omega, \xi)) d\omega d\xi,$$
which captures the variability of both the phenomenon and its measurement.

---

class: middle

.center.width-10[![](figures/lec2/penguin.png)]

Example: how a penguin record is made.

- Bill length and depth: dial calipers, to 0.1 mm.
- Flipper length: ruler, to 1 mm.
- Body mass: Pesola spring scale and a weigh bag, to 25 g.
- Sex: not measured at all, but inferred in the lab from a blood sample.

Nests are found at the one-egg stage, both adults are caught, measured, sampled and released. All of this is $\xi$.

.success[The instruments are visible in the data: every body mass is a multiple of 25 g, every bill length has one decimal, every flipper length is a whole millimetre.]

.footnote[Credits: [Gorman et al.](https://doi.org/10.1371/journal.pone.0090081), 2014.]

---

class: middle

## Data types

Atomic data are the indivisible units of information collected through measurements. They are often categorized based on their nature and the operations that can be performed on it.

- Numerical (continuous, discrete)
- Categorical (nominal, ordinal)

---

class: middle

Numerical data
- Continuous: $x \in \mathbb{R}$ (e.g., temperature), $x \in \mathbb{R}^+$ (e.g., height, weight)
- Discrete: $x \in \mathbb{N}$ (e.g., counts), $x \in \mathbb{Z}$ (e.g., differences)

---

class: middle

Categorical data
- Nominal: $x \in \mathcal{C} = \\{c\_1, c\_2, ..., c\_n \\}$ (e.g., colors, types, text characters) without intrinsic order
- Ordinal: $x \in \mathcal{C}$ with ordering relations $c\_1 \prec c\_2 \prec ... \prec c\_n$ (e.g., ratings, grades)

---

class: middle 

.center.width-10[![](figures/lec2/penguin.png)]

Example: one penguin record
```
species: Adelie          # Categorical, nominal
island: Torgersen        # Categorical, nominal
bill length: 39.1 mm     # Numerical, continuous
body mass: 3750 g        # Numerical, continuous
sex: male                # Categorical, nominal
year: 2007               # Numerical, discrete
```
No variable here is ordinal; a rating from 1 to 5, or a grade, would be.

---

class: middle

## Data structures

A measurement $x$ can be a single atomic value or a composite structure made of multiple atomic values. Common aggregates or data structures include:
- Tabular data
- Arrays and tensors
- Sequences
- Networks and graphs

---

class: middle

.grid[
.kol-1-2.center[Tabular data<br>.width-100[![](figures/lec2/penguins-tabular.png)]]
.kol-1-2.center[Arrays and tensors<br>.width-60[![](figures/lec2/penguin-photo.jpg)]]
]

<br>

.grid[
.kol-1-2.center[Sequences

`['P', 'e', 'n', 'g', 'u', 'i', 'n']`]
.kol-1-2.center[Networks and graphs<br>.width-70[![](figures/lec2/graph.png)]]
]

.footnote[Credits: Hannes Grobe, [Adelie penguin](https://commons.wikimedia.org/wiki/File:Pygoscelis_adeliae_hg.jpg) (CC BY-SA 2.5), cropped.]

---

class: middle

A data frame $\mathbf{X}$ represents a .bold[tabular collection] of $n$ records (rows) over $d$ variables/atomic measurements (columns),
$$\mathbf{X} = \begin{pmatrix}
x\_{11} & x\_{12} & \cdots & x\_{1d} \\\\
x\_{21} & x\_{22} & \cdots & x\_{2d} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
x\_{n1} & x\_{n2} & \cdots & x\_{nd}
\end{pmatrix}.$$
Each entry $x\_{ij}$ corresponds to the value of variable $j$ for record $i$. 

Variables are often heterogeneous (mixing numerical and categorical types). When all variables are numerical, the data frame can be viewed as a matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$.

For the penguins, $n = 344$ records over $d = 8$ variables, four of them numerical.

---

class: middle

Collections of homogeneous measurements can be represented as .bold[arrays] or .bold[tensors] $\mathbf{X} \in \mathbb{R}^{d\_1 \times d\_2 \times \cdots \times d\_k}$, where the position of each atomic value in the array is usually associated to a spatial or temporal location.
- Images: 3d arrays $\mathbf{X} \in \\{0, \ldots, 255\\}^{h \times w \times c}$ (height, width, channels). The photo on the previous slide is such an array.
- Videos: 4d arrays $\mathbf{X} \in [0, 255]^{t \times h \times w \times c}$ (time, height, width, channels).

---

class: middle

Data can also be structured as ordered .bold[sequences] $S = (x\_1, x\_2, ..., x\_T)$ indexed by time or position. Each element $x\_t$ can be atomic or composite.
- Time series: $S = (x\_1, x\_2, ..., x\_T)$ where $x\_t$ is a measurement at time $t$ (e.g., stock prices).
- Text: $S = (w\_1, w\_2, ..., w\_T)$ where $w\_t$ is the $t$-th word in a document.

---

class: middle

Finally, data can be organized as .bold[networks] or .bold[graphs] $G = (V, E)$, where entities are represented as nodes $V$ and relationships as edges $E$. Each may also have associated attributes, $x\_v$ for nodes and $x\_{uv}$ for edges.
- Molecular structures, where nodes represent atoms and edges represent bonds.
- Social networks, where nodes represent individuals and edges represent interactions or relationships.

---

class: middle

.center.width-10[![](figures/lec2/laptop.png)]

Follow the tutorials in `nb02a-tables.ipynb`, `nb02b-jax.ipynb`, and `nb02c-data-wrangling.ipynb` to practice working with arrays and data frames in Python.

---

class: middle

## Data quality 

Real-world data are often imperfect and may suffer from various .bold[quality issues] that can impact analysis and modeling. Common data quality issues include:
- Missing values
- Measurement errors
- Outliers

---

class: middle

.bold[Missing values] are common in real-world datasets and can arise from various factors such as non-response in surveys, sensor malfunctions, or data corruption. 

Let $\mathbf{X}\_\text{full}$ be the complete data and $\mathbf{M} \in \\{0, 1\\}^{n \times d}$ the missingness pattern, with $m\_{ij} = 1$ when entry $ij$ is observed and $m\_{ij} = 0$ when it is missing$^1$.

What we actually hold is the pair $(\mathbf{X}\_\text{obs}, \mathbf{M})$, where $\mathbf{X}\_\text{obs} = \\{ x\_{ij} : m\_{ij} = 1 \\}$ are the observed entries and $\mathbf{X}\_\text{mis} = \\{ x\_{ij} : m\_{ij} = 0 \\}$ the missing ones. A missing entry is not a zero, and no arithmetic can recover it.

.footnote[1: Rubin's convention is the opposite, 1 for missing.]

---

class: middle

.center.width-100[![](figures/lec2/penguins-missing.png)]

.center[The 11 penguin records with missing entries&#58; two birds were never measured,<br> nine have no recorded sex.]

???

Whether these holes matter depends on the mechanism: a bird that escaped before being weighed is plausibly missing completely at random, but if the largest and strongest escaped most often, the same holes would be missing not at random.

---

class: middle

The pattern $\mathbf{M}$ is itself produced by the measurement process, and is modeled as such:
- Missing completely at random (MCAR): the pattern is independent of the data, $p(\mathbf{M} \mid \mathbf{X}\_\text{full}) = p(\mathbf{M})$.
- Missing at random (MAR): the pattern may depend on what is observed, but not on what is missing, $p(\mathbf{M} \mid \mathbf{X}\_\text{full}) = p(\mathbf{M} \mid \mathbf{X}\_\text{obs})$.
- Missing not at random (MNAR): the pattern depends on the missing entries themselves.

---

class: middle

Why the distinction matters&#58; what we can compute is
$$p(\mathbf{X}\_\text{obs}, \mathbf{M}) = \int p(\mathbf{X}\_\text{full}) p(\mathbf{M} \mid \mathbf{X}\_\text{full}) d\mathbf{X}\_\text{mis}.$$

Under MCAR or MAR, the second factor does not depend on the missing entries and, provided it shares no parameters with the first, it can be ignored: modelling the observed data is enough. Under MNAR it cannot be ignored, and the mechanism must be modeled jointly with the data.

---

class: middle

Example&#58;

Survey of $n=1000$ respondents, 30% do not answer the income question.
- MCAR: Randomly selected respondents skip the question.
- MAR: Younger respondents are less likely to answer.
- MNAR: High earners refuse to answer.

.alert[.bold[Imputing or discarding missing values] without making explicit the assumptions about the missingness mechanism can lead to biased results.]

---

class: middle

The measurement process can also introduce acquisition errors and produce observations that deviate significantly from regular measurements. These .bold[outliers] can arise from instrument malfunctions, data entry errors, or rare events.

---

class: middle

.center.width-50[![](figures/lec2/gw-glitch-powerline6.png)]

.footnote[Credits: Omega scan of a power-line glitch at LIGO Livingston (O4), [Gravity Spy](https://gravityspy.org); data from [GWOSC](https://gwosc.org) (CC BY 4.0).]

Example&#58;

Glitches in gravitational wave detectors are outliers that can mimic true signals and complicate detection efforts. They can arise from environmental disturbances, instrumental artifacts, or other non-astrophysical sources.

???

Here powerline glitches are visible at 60 Hz, due to electromagnetic interference from electrical power systems.

---

class: middle

Treating outliers requires a model of the measurement process that either describes measurements under normal conditions or explicitly accounts for anomalies. 

Two models make this explicit&#58; a contamination mixture
$$p(x) = (1 - \varepsilon) p\_\text{model}(x) + \varepsilon p\_\text{bad}(x),$$
which gives bad measurements their own distribution, or a heavy-tailed data model, which allows rare large deviations without special-casing them.

.alert[.bold[Outliers should not be removed blindly] unless explicitly justified by the measurement model or domain knowledge.]

---

class: middle

.center.width-10[![](figures/lec2/high-temperature.png)]

Temperature readings in Liège:

```
20.1, 19.8, 20.3, 1000.0, 20.2, 19.9
```
The 1000.0 value is an outlier likely due to a sensor error.

```
19.2, 19.8, 20.3, 37.8, 20.2, 19.9
```
The 37.8 value is a rare but plausible measurement on a hot day.

---

class: middle

## Where this leaves us

Everything in this part is one picture of how the data came to be:
- the map $f$ from states of the world to measurements,
- the conditions $\xi$ under which measurements are taken,
- the pattern $\mathbf{M}$ of what ends up recorded,
- the data distribution $p\_r(x)$ that all of this induces.

---

class: middle

# Exploratory data analysis 

---

class: middle

.bold[Exploratory data analysis] (EDA) is the systematic examination of data to understand its structure, patterns and anomalies before formal modeling. It typically involves visual and quantitative techniques to summarize key characteristics of the data.

The goal is to generate hypotheses and inform modeling decisions, not to confirm preconceived notions.

---

class: middle

## What we are looking at

All we hold is $n$ records. Their .bold[empirical distribution] puts equal mass on each of them,
$$\hat{p}\_n(x) = \frac{1}{n} \sum\_{i=1}^n \delta(x - x\_i).$$

Every plot and every statistic that follows is a functional of $\hat{p}\_n$: a histogram is a marginal, a scatter plot a joint, body mass by species a conditional, correlation and mutual information measure dependence, and PCA is a projection.

.alert[$\hat{p}\_n$ is not $p\_r$. It is what $n$ records show of it, and that gap is what the rest of the course is about.]

---

class: middle

.center.width-10[![](figures/lec2/penguin.png)]

Back to the penguins, with the whole table in hand: 344 records, four numerical measurements, three species and three islands.

Switch to `nb02d-eda.ipynb` to follow along.

---

class: middle

## Univariate analysis

Let us consider a variable $j$ from a data frame $\mathbf{X} \in \mathbb{R}^{n \times d}$, represented as the vector $\mathbf{x}\_j = (x\_{1j}, x\_{2j}, ..., x\_{nj})^T$.

Univariate analysis focuses on understanding the distribution and characteristics of this single variable.

For numerical variables, common techniques include:
- Looking at the raw data: print values, scroll through them.
- Plotting the data: histograms reveal the distribution shape.
- Summarizing with statistics: mean, median, mode, variance, skewness, kurtosis.

For categorical variables:
- Counting occurrences of each category.

---

class: middle

.center[
.width-80[![](figures/lec2/body_mass_histogram.png)]
]

.center[Numerical: Histogram of body mass for all penguins.<br>
.italic[One broad mode with a long right tail: perhaps not a single population.]]

---

class: middle

.center[
.width-80[![](figures/lec2/species_counts.png)]
]

.center[Categorical: Bar plot of species counts for all penguins.<br>
.italic[Unbalanced groups: 152 Adelie, 124 Gentoo, 68 Chinstrap.]]

---

class: middle

## Bivariate analysis

Bivariate analysis examines the relationship between two variables $j$ and $k$ from a data frame $\mathbf{X} \in \mathbb{R}^{n \times d}$, represented as the vectors $\mathbf{x}\_j$ and $\mathbf{x}\_k$.

Depending on the types of variables, different techniques are used:
- Pair plots for two numerical variables (scatter or 2d histogram).
- Multiple histograms for categorical vs numerical variables.
- Contingency tables for two categorical variables.
- Correlation coefficients (e.g., Pearson, Spearman) for numerical variables.

---

class: middle

.center.width-80[![](figures/lec2/body_mass_vs_flipper_length.png)]
.center[Numerical vs. numerical: Scatter plot of body mass vs flipper length.<br>
.italic[Mass grows with flipper length, close to linearly.]]

---

class: middle

.center.width-75[![](figures/lec2/pairplot.png)]
.center[Pair plots of all numerical variables.<br>
.italic[Most panels show two or three clouds rather than one.]]

---

class: middle

.center.width-80[![](figures/lec2/body_mass_by_species.png)]
.center[Categorical vs. numerical: Histograms of body mass by species.<br>
.italic[The right tail was Gentoo, at 5076 g on average against about 3700 g for the others.]]

---

class: middle

.center[![](figures/lec2/contingency.png)]
.center[Categorical vs. categorical: Contingency table of species and island.<br>
.italic[Gentoo live only on Biscoe, Chinstrap only on Dream: species and island are dependent.]]

---

class: middle

.bold[Correlation coefficients] can quantify the dependency between two numerical variables. They are useful but come with assumptions and limitations.

- Pearson correlation measures linear relationships. Its empirical version reads
$$\hat{\rho}\_{jk} = \frac{\sum\_{i=1}^n (x\_{ij} - \bar{x}\_j)(x\_{ik} - \bar{x}\_k)}{\sqrt{\sum\_{i=1}^n (x\_{ij} - \bar{x}\_j)^2} \sqrt{\sum\_{i=1}^n (x\_{ik} - \bar{x}\_k)^2}},$$
an estimate of $\rho\_{jk} = \text{cov}(x\_j, x\_k) / (\sigma\_j \sigma\_k)$. It ignores non-linear dependencies.
- Spearman correlation is the Pearson correlation of the rank-transformed variables. It captures monotonic relationships.
- Correlation does not imply causation and can be affected by outliers.

???

Intuition behind Pearson: the ratio of the covariance to the product of standard deviations measures how much two variables co-vary relative to their individual variability. It captures linear relationships because covariance is a linear measure.

---

class: middle

The .bold[mutual information] between two variables $x\_j$ and $x\_k$ measures the reduction in uncertainty about one given knowledge of the other. For discrete variables,
$$I(x\_j; x\_k) = \sum\_{x\_j} \sum\_{x\_k} p(x\_j, x\_k) \log \frac{p(x\_j, x\_k)}{p(x\_j)p(x\_k)},$$
with integrals in place of the sums for continuous ones.

Mutual information captures any statistical relationship, not just linear or monotonic ones. It is defined on the distributions, however, which must themselves be estimated from the $n$ records, and that is hard.

---

class: middle

## Multivariate analysis

Multivariate analysis explores relationships among three or more variables in a data frame $\mathbf{X} \in \mathbb{R}^{n \times d}$.

Common techniques include:
- The same as bivariate analysis, but conditioning on a third variable (e.g., pair plots colored by species).
- Dimensionality reduction methods (e.g., PCA, t-SNE) to visualize high-dimensional data.
- Clustering algorithms (e.g., k-means, hierarchical clustering) to identify groups of similar records.

---

class: middle

.center.width-80[![](figures/lec2/pairplot_by_species.png)]
.center[Pair plots of all numerical variables, colored by species.<br>
.italic[Colour by species and the clouds line up.]]

---

class: middle

.center.width-70[![](figures/lec2/pca_penguins.png)]
.center[PCA projection of all numerical variables, colored by species.<br>
.italic[Two components are enough to separate the species.]]

---

class: middle

## What the penguins leave us

Four hypotheses, none of them a result:
- body mass is not one population, but a .bold[mixture] of groups,
- flipper length and body mass move together, as in a .bold[regression],
- species and island are dependent, which is .bold[group structure],
- four measurements carry much the same information, suggesting a .bold[low-dimensional] description.

Each is built as a model later: mixtures and latent variables in Lecture 4, regression in Lecture 6.

---

class: middle

EDA does not stop once a model is built. In the critique step, the same plots come back on what the model gets wrong.

- Residuals reveal patterns the model does not capture.
- Prediction errors point to subsets of the data that are problematic.
- Unexpected patterns suggest new variables or a revised model.

Next lecture: how to draw all of these plots well.

---

class: end-slide, center
count: false

The end.
