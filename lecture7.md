class: middle, center, title-slide

# Foundations of Data Science

Lecture 7: Model criticism and validation

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

---

class: middle, black-slide

.center[
<iframe width="640" height="400" src="https://www.youtube.com/embed/LIxvQMhttq4?cc_load_policy=1&hl=en&version=3" frameborder="0" allowfullscreen></iframe>
]

.quote[If it (a model) disagrees with experiment, it is wrong. In that simple statement is the key to science. .author[Richard Feynman]]

---

class: middle

# Model checking 

---

class: middle

The critique of a fitted model asks two questions.

- Can the computation be trusted? Diagnostics and calibration check that the sampler draws from the posterior of the model.
- Does the model explain the data? Only the data can tell. A sampler can be exact for a model that is wrong, and pass every check of the computation.

---

class: middle

## Newcomb's experiment (1882)

Simon Newcomb measured the speed of light with a rotating mirror. He recorded $N = 66$ measurements of the time light takes to travel a known distance.

.center.width-70[![](./figures/lec7/newcomb-data.png)]
.center[Newcomb's 66 measurements, in nanoseconds above 24800 ns.]

---

class: middle

.italic[Example.] We model the measurements $\mathbf{x}\_\text{obs} = (x\_1, \ldots, x\_N)$ as independent Gaussian draws,

.center.width-50[![](./figures/lec7/newcomb-model.svg)]

$$\begin{aligned}
\mu &\sim \mathcal{U}(-1000, 1000) \\\\
\sigma &\sim \mathcal{U}(0.1, 1000) \\\\
x\_i &\sim \mathcal{N}(\mu, \sigma^2) \quad \text{for } i = 1, \ldots, N.
\end{aligned}$$

---

class: middle

MCMC gives samples $(\mu^{(m)}, \sigma^{(m)})$, $m = 1, \ldots, M$, from the posterior $p(\mu, \sigma \mid \mathbf{x}\_\text{obs})$.

.center.width-45[![](./figures/lec7/newcomb-posterior.png)]

.center[Posterior samples, with the 2.5%, 50% and 97.5% quantiles.]

---

class: middle

The posterior mean of $\mu$ estimates the passage time, in nanoseconds above 24800 ns,
$$\hat{\mu} = \frac{1}{M} \sum\_{m=1}^M \mu^{(m)} = 26.199182694915095.$$

Two uncertainties decide which digits mean something:
- the .bold[Monte Carlo standard error] $\text{sd}(\mu^{(m)}) / \sqrt{\text{ESS}}$ of the average, where the effective sample size ESS counts the independent samples with the same precision; here about 0.01 ns, with $\text{ESS} \approx 9000$;
- the posterior standard deviation of $\mu$, about 1.4 ns, the uncertainty about $\mu$ itself.

The second dominates. Report $\mu = 26.2 \pm 1.4$ ns.

---

class: middle

## Credible intervals

A credible interval of level $1 - \alpha$ is an interval $[a, b]$ such that
$$P(a \leq \mu \leq b \mid \mathbf{x}\_\text{obs}) = 1 - \alpha.$$
The highest posterior density (HPD) interval is the narrowest one. The equal-tailed interval leaves $\alpha / 2$ of the posterior on each side.

---

class: middle

From the samples, the 95% equal-tailed interval runs from the 2.5% to the 97.5% quantile of the $\mu^{(m)}$,
$$[\mu\_{2.5\%}, \mu\_{97.5\%}] = [23.6, 28.9].$$

The probability is over $\mu$, with the data fixed. A confidence interval instead holds the parameter fixed and describes the variability of the interval over repeated data sets.

---

class: middle

## Posterior predictive checks

Before any data are observed, a prior predictive check simulates data from the prior predictive distribution $p(\mathbf{x}) = \int p(\mathbf{x} \mid \theta) p(\theta) d\theta$, and compares them with what is known. After the data, the same simulation with the posterior in place of the prior checks the fitted model.

The .bold[posterior predictive distribution] of replicated data $\mathbf{x}^\text{rep}$ is
$$p(\mathbf{x}^\text{rep} \mid \mathbf{x}\_\text{obs}) = \int p(\mathbf{x}^\text{rep} \mid \theta) p(\theta \mid \mathbf{x}\_\text{obs}) d\theta.$$

If the model fits, replicated data resemble the observed data. A .bold[posterior predictive check] compares the two.

---

class: middle

For Newcomb's data, each replicated data set is drawn in two steps:
- draw $(\mu^{(m)}, \sigma^{(m)}) \sim p(\mu, \sigma \mid \mathbf{x}\_\text{obs})$;
- draw $x\_i^{\text{rep}(m)} \sim \mathcal{N}(\mu^{(m)}, (\sigma^{(m)})^2)$ for $i = 1, \ldots, N$.

---

class: middle

.center.width-70[![](./figures/lec7/newcomb-replicates.png)]
.center[The observed data among 19 data sets replicated from the posterior predictive distribution.]

---

class: middle

To check one aspect of the fit, compare a .bold[test statistic] $T(\mathbf{x}\_\text{obs})$ with its distribution $p(T(\mathbf{x}^\text{rep}) \mid \mathbf{x}\_\text{obs})$ over replicated data sets.

.center.width-70[![](./figures/lec7/newcomb-replicate-stats.png)]

The mean and the variance of the observed data are typical of the replicated data: the model captures location and spread.

---

class: middle

.center.width-70[![](./figures/lec7/newcomb-replicate-mins.png)]

The observed minimum is lower than the minimum of every replicated data set. The Gaussian model cannot produce the lowest measurements.

---

class: middle

The .bold[Bayesian p-value] of a test statistic $T$ is
$$P(T(\mathbf{x}^\text{rep}) \geq T(\mathbf{x}\_\text{obs}) \mid \mathbf{x}\_\text{obs}) = \int P(T(\mathbf{x}^\text{rep}) \geq T(\mathbf{x}\_\text{obs}) \mid \theta) p(\theta \mid \mathbf{x}\_\text{obs}) d\theta.$$
A value close to 0 or 1 signals a misfit for $T$.

A frequentist p-value fixes $\theta$, at a hypothesized value $\theta^\*$ or at the estimate $\hat{\theta}$, and computes $P(T(\mathbf{x}^\text{rep}) \geq T(\mathbf{x}\_\text{obs}) \mid \theta)$ over data sets drawn from $p(\mathbf{x} \mid \theta)$. A test at level 5% rejects $\theta$ when this probability falls below 0.05. The Bayesian p-value averages it over the posterior instead.

---

class: middle

For Newcomb's data, the Bayesian p-values are
- 0.49 for the mean;
- 0.52 for the variance;
- 1.00 for the minimum.

The model fails on the lower tail of the data.

---

class: middle

## Residual analysis

Suppose each measurement is a prediction plus additive noise,
$$x\_i = f\_i(\theta) + \sigma\epsilon\_i,$$
where $f\_i(\theta)$ is the value the model predicts for measurement $i$ (it may depend on an input, as in a regression), $\sigma$ is a scale and $\epsilon\_i \sim p(\epsilon)$ is noise. The standardized .bold[residuals] are
$$r\_i = \frac{x\_i - f\_i(\theta)}{\sigma}, \quad i = 1, \ldots, N,$$
with $\theta \sim p(\theta \mid \mathbf{x}\_\text{obs})$.

In a state-space model, the residuals are the standardized innovations of the Kalman filter: each observation minus its prediction from the previous ones, divided by the predicted standard deviation.

---

class: middle

If the model fits, the residuals of the observed data are distributed as the residuals of replicated data. If the posterior is concentrated, both are close to the noise distribution $p(\epsilon)$.

A .bold[quantile-quantile (Q-Q) plot] compares two distributions by drawing the quantiles of one against those of the other. Points on the diagonal mean the two distributions agree.

---

class: middle

For the Gaussian model, the standardized residuals are
$$r\_i = \frac{x\_i - \mu}{\sigma}, \quad i = 1, \ldots, N,$$
with $(\mu, \sigma) \sim p(\mu, \sigma \mid \mathbf{x}\_\text{obs})$.

.center.width-70[![](./figures/lec7/newcomb-residuals.png)]

.center[Residuals of observed and replicated data (left), and their Q-Q plot (right). <br>The observed residuals have a heavier lower tail and a narrower core.]

---

class: middle

.center.width-10[![](./figures/lec7/repair.png)]

Model checking compares the observed data with data simulated from the fitted model, by plots and by test statistics. It reveals .bold[misspecifications], and points to what the next model must fix.

---

class: middle

# Model comparison

---

class: middle

.italic[Example.] The two lowest measurements could be deleted as outliers, but nothing in the data says they are errors. A heavy-tailed model keeps them: a Student's $t$ gives large deviations a larger probability than a Gaussian.

.center.width-50[![](./figures/lec7/newcomb-model2.svg)]

$$\begin{aligned}
\mu &\sim \mathcal{U}(-1000, 1000) \\\\
\sigma &\sim \mathcal{U}(0.1, 1000) \\\\
\nu &\sim \text{Gamma}(2, 0.1) \\\\
\epsilon\_i &\sim t\_{\nu} \\\\
x\_i &= \mu + \sigma \epsilon\_i \quad \text{for } i = 1, \ldots, N.
\end{aligned}$$

---

class: middle

MCMC gives samples from the posterior $p(\mu, \sigma, \nu \mid \mathbf{x}\_\text{obs})$.

.center.width-55[![](./figures/lec7/newcomb-model2-posterior.png)]

.center[Posterior samples of the $t$ model, with the 2.5%, 50% and 97.5% quantiles.]

---

class: middle

## Bayes factors

Consider two models $\mathcal{M}\_1$ and $\mathcal{M}\_2$, with prior probabilities $p(\mathcal{M}\_1)$ and $p(\mathcal{M}\_2)$. Bayes' rule gives the ratio of their posterior probabilities,
$$\frac{p(\mathcal{M}\_1 \mid \mathbf{x}\_\text{obs})}{p(\mathcal{M}\_2 \mid \mathbf{x}\_\text{obs})} = \frac{p(\mathbf{x}\_\text{obs} \mid \mathcal{M}\_1)}{p(\mathbf{x}\_\text{obs} \mid \mathcal{M}\_2)} \frac{p(\mathcal{M}\_1)}{p(\mathcal{M}\_2)}.$$
The first factor on the right is the .bold[Bayes factor] $\text{BF}\_{1,2}$.

---

class: middle

The Bayes factor measures how much more probable the data are under one model than under the other. $\text{BF}\_{1,2} > 1$ favours $\mathcal{M}\_1$.

A common scale reads a Bayes factor
- from 1 to 3 as weak evidence;
- from 3 to 10 as moderate evidence;
- above 10 as strong evidence.

For Newcomb's data, $\log \text{BF}\_{t, \text{Gaussian}} \approx 30$: strong evidence for the $t$ model.

The Bayes factor depends on the priors. A prior on $\mu$ ten times wider leaves it unchanged, since both models share it. A prior on $\nu$ ten times wider, $\text{Gamma}(2, 0.01)$, lowers $\log \text{BF}\_{t, \text{Gaussian}}$ to about 25.

---

class: middle

A Bayes factor is a ratio of .bold[marginal likelihoods],
$$p(\mathbf{x}\_\text{obs} \mid \mathcal{M}) = \int p(\mathbf{x}\_\text{obs} \mid \theta, \mathcal{M}) p(\theta \mid \mathcal{M}) d\theta,$$
the probability of the data averaged over the prior. It is the normalizer of Bayes' rule,
$$p(\theta \mid \mathbf{x}\_\text{obs}, \mathcal{M}) = \frac{p(\mathbf{x}\_\text{obs} \mid \theta, \mathcal{M}) p(\theta \mid \mathcal{M})}{p(\mathbf{x}\_\text{obs} \mid \mathcal{M})}.$$

The integral rarely has a closed form, and MCMC never computes it: it samples the posterior from the unnormalized density alone.

---

class: middle

The .bold[Laplace approximation] replaces the posterior by a Gaussian. Let $\hat{\theta} = \arg\max\_{\theta} p(\theta \mid \mathbf{x}\_\text{obs}, \mathcal{M})$ be the mode of the posterior, the maximum a posteriori (MAP) estimate. At the mode the gradient vanishes, and a second-order Taylor expansion of the log posterior is a parabola,
$$\log p(\theta \mid \mathbf{x}\_\text{obs}, \mathcal{M}) \approx \log p(\hat{\theta} \mid \mathbf{x}\_\text{obs}, \mathcal{M}) - \frac{1}{2} (\theta - \hat{\theta})^T \boldsymbol{\Sigma}^{-1} (\theta - \hat{\theta}),$$
where $\boldsymbol{\Sigma}^{-1}$ is the Hessian of $-\log p(\theta \mid \mathbf{x}\_\text{obs}, \mathcal{M})$ at $\hat{\theta}$. Exponentiating,
$$p(\theta \mid \mathbf{x}\_\text{obs}, \mathcal{M}) \approx \mathcal{N}(\theta \mid \hat{\theta}, \boldsymbol{\Sigma}).$$

---

class: middle

Bayes' rule, evaluated at $\hat{\theta}$, and the approximation give
$$\begin{aligned}
p(\mathbf{x}\_\text{obs} \mid \mathcal{M}) &= \frac{p(\mathbf{x}\_\text{obs} \mid \hat{\theta}, \mathcal{M}) p(\hat{\theta} \mid \mathcal{M})}{p(\hat{\theta} \mid \mathbf{x}\_\text{obs}, \mathcal{M})} \\\\
&\approx p(\mathbf{x}\_\text{obs} \mid \hat{\theta}, \mathcal{M}) p(\hat{\theta} \mid \mathcal{M}) (2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2},
\end{aligned}$$
where $d$ is the number of parameters and $(2\pi)^{-d/2} |\boldsymbol{\Sigma}|^{-1/2}$ is the density of the Gaussian at its mode.

---

class: middle

.center.width-10[![](./figures/lec7/razor.png)]

The marginal likelihood averages the likelihood over the prior. A flexible model spreads its prior over many data sets, and gives each a small probability. Comparing marginal likelihoods therefore implements a Bayesian .bold[Occam's razor]: a simpler model is preferred unless the data require a more flexible one.

---

class: middle

.italic[Example.] Polynomial regression of degree 3 and 7, on 20 noisy points.

.center.width-70[![](./figures/lec7/polynomial-fit.png)]

The marginal likelihood favours degree 3 ($\log p(\mathbf{x}\_\text{obs} \mid \mathcal{M}) \approx -67$) over degree 7 ($\approx -86$), although degree 7 contains degree 3 as a special case.

---

class: middle

## Cross-validation

.bold[Cross-validation] compares models by how well they predict data they were not fitted on.

In .bold[$k$-fold cross-validation], the data are split into $k$ folds of $N / k$ observations. Each fold serves once as the validation set, and the other $k - 1$ folds as the training set. The scores are averaged over the $k$ splits.

.center.width-70[![](./figures/lec7/kfold.png)]

---

class: middle

The score of a Bayesian model is the .bold[expected log predictive density] (ELPD),
$$\mathbb{E}\_{p\_r(\mathbf{x}')}[\log p(\mathbf{x}' \mid \mathbf{x}\_\text{obs})],$$
where $p\_r$ is the data distribution and $p(\mathbf{x}' \mid \mathbf{x}\_\text{obs})$ is the posterior predictive distribution, fitted on the training set. Cross-validation estimates the expectation with the validation set.

For Newcomb's data, 5-fold cross-validation, summed over the 66 held-out measurements, gives $-225$ for the $t$ model and $-287$ for the Gaussian model.

---

class: middle

The ELPD measures how close the predictions are to the data distribution. The Kullback-Leibler divergence from $p\_r$ to the posterior predictive distribution is
$$\text{KL}(p\_r \,\\|\, p(\cdot \mid \mathbf{x}\_\text{obs})) = \mathbb{E}\_{p\_r(\mathbf{x}')}[\log p\_r(\mathbf{x}')] - \mathbb{E}\_{p\_r(\mathbf{x}')}[\log p(\mathbf{x}' \mid \mathbf{x}\_\text{obs})].$$
The first term does not depend on the model, so a higher ELPD means a smaller divergence. Neither model needs to be true: the ELPD ranks them by how close they come to $p\_r$.

If the posterior is replaced by a point estimate $\hat{\theta}$, the ELPD reduces to the log-likelihood $\log p(\mathbf{x}' \mid \hat{\theta})$ of the held-out data.

---

class: end-slide, center
count: false

The end.
