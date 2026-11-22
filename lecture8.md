class: middle, center, title-slide

# Foundations of Data Science

Lecture 8: Expectation-maximization

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

---

class: middle

.center.width-70[![](figures/lec8/penguin-bills.png)]

.question[How many groups are there, and which penguin belongs to which?]

---

class: middle

## Gaussian mixture model

We model the bills $\mathbf{x}\_i \in \mathbb{R}^2$ as a mixture of $K = 3$ Gaussian components, with latent variables $z\_i \in \\{1, \ldots, K\\}$ labelling the component of each penguin,
$$\begin{aligned}
p(z\_i = k \mid \theta) &= \pi\_k, \\\\
p(\mathbf{x}\_i \mid z\_i = k, \theta) &= \mathcal{N}(\mathbf{x}\_i \mid \boldsymbol{\mu}\_k, \boldsymbol{\Sigma}\_k),
\end{aligned}$$
where the parameters $\theta = (\boldsymbol{\pi}, \boldsymbol{\mu}\_{1:K}, \boldsymbol{\Sigma}\_{1:K})$ are the weights, means and covariances of the components.

.center.width-50[![](figures/lec8/gmm-model.svg)]

---

class: middle

## Maximum likelihood

Summing out the latent variables gives the log-likelihood
$$\begin{aligned}
\log p(\mathbf{x}\_{1:N} \mid \theta) &= \sum\_{i=1}^N \log \sum\_{k=1}^K p(\mathbf{x}\_i, z\_i = k \mid \theta) \\\\
&= \sum\_{i=1}^N \log \sum\_{k=1}^K \pi\_k \mathcal{N}(\mathbf{x}\_i \mid \boldsymbol{\mu}\_k, \boldsymbol{\Sigma}\_k).
\end{aligned}$$
Its maximum has no closed-form solution.

.alert[More generally, latent variable models have sums or integrals inside the logarithm, which are often intractable even to evaluate.]

---

class: middle

## If the components were known...

With the $z\_i$ known, the complete-data log-likelihood splits into one term per component,
$$\log p(\mathbf{x}\_{1:N}, z\_{1:N} \mid \theta) = \sum\_{i=1}^N \log \pi\_{z\_i} + \sum\_{i=1}^N \log \mathcal{N}(\mathbf{x}\_i \mid \boldsymbol{\mu}\_{z\_i}, \boldsymbol{\Sigma}\_{z\_i}).$$
Setting its gradients to zero gives the mean, the covariance and the fraction of the $N\_k$ penguins in each component,
$$\begin{aligned}
\boldsymbol{\mu}\_k &= \frac{1}{N\_k} \sum\_{i : z\_i = k} \mathbf{x}\_i, \\\\
\boldsymbol{\Sigma}\_k &= \frac{1}{N\_k} \sum\_{i : z\_i = k} (\mathbf{x}\_i - \boldsymbol{\mu}\_k)(\mathbf{x}\_i - \boldsymbol{\mu}\_k)^T, \\\\
\pi\_k &= \frac{N\_k}{N}.
\end{aligned}$$

---

class: middle

.center.width-10[![](figures/lec8/light-bulb.png)]

.center[What if we alternate between inferring the latent variables<br> and fitting the parameters?]

---

class: middle

# Expectation-maximization

---

class: middle

.center.width-50[![](figures/lec8/lvm.svg)]

Consider a latent variable model $p(\mathbf{x}, \mathbf{z} \mid \theta)$. We look for the maximum likelihood estimate
$$\hat{\theta} = \arg\max\_\theta \log p(\mathbf{x} \mid \theta), \quad \text{with } p(\mathbf{x} \mid \theta) = \int p(\mathbf{x}, \mathbf{z} \mid \theta) d\mathbf{z}.$$

---

class: middle

## Jensen's inequality

Since the logarithm is concave, .bold[Jensen's inequality] gives
$$\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$$
for any positive random variable $Y$, with equality when $Y$ is constant.

.center.width-70[![](figures/lec8/jensen.png)]

.center[$Y$ takes the values $y\_1$ and $y\_2$ with probability 1/2 each. <br>The mean of the logarithms lies on the chord, below the logarithm of the mean.]

---

class: middle

## Kullback-Leibler divergence

Recall the Kullback-Leibler divergence between two distributions $q$ and $p$,
$$\text{KL}(q \\| p) = \mathbb{E}\_{q(\mathbf{z})}\left[\log \frac{q(\mathbf{z})}{p(\mathbf{z})}\right].$$
It is nonnegative, and zero only when $q = p$, since by Jensen's inequality
$$-\text{KL}(q \\| p) = \mathbb{E}\_{q}\left[\log \frac{p(\mathbf{z})}{q(\mathbf{z})}\right] \leq \log \mathbb{E}\_{q}\left[\frac{p(\mathbf{z})}{q(\mathbf{z})}\right] = \log 1 = 0.$$

.center.width-70[![](figures/lec8/kl.png)]

---

class: middle

## Evidence lower bound

The likelihood is an expectation over the prior of the latent variables,
$$\log p(\mathbf{x} \mid \theta) = \log \int p(\mathbf{x} \mid \mathbf{z}, \theta) p(\mathbf{z} \mid \theta) d\mathbf{z} = \log \mathbb{E}\_{p(\mathbf{z} \mid \theta)}\left[p(\mathbf{x} \mid \mathbf{z}, \theta)\right].$$
It could be estimated by averaging $p(\mathbf{x} \mid \mathbf{z}, \theta)$ over samples of the prior. But a $\mathbf{z}$ drawn from the prior is unlikely to explain $\mathbf{x}$, and $p(\mathbf{x} \mid \mathbf{z}, \theta)$ is then close to zero.

Instead, we draw $\mathbf{z}$ from a distribution $q(\mathbf{z})$ that favours the values explaining $\mathbf{x}$, and reweight,
$$\log p(\mathbf{x} \mid \theta) = \log \mathbb{E}\_{q(\mathbf{z})}\left[\frac{p(\mathbf{x} \mid \mathbf{z}, \theta) p(\mathbf{z} \mid \theta)}{q(\mathbf{z})}\right].$$

---

class: middle

By Jensen's inequality,
$$\begin{aligned}
\log p(\mathbf{x} \mid \theta) &= \log \mathbb{E}\_{q(\mathbf{z})}\left[\frac{p(\mathbf{x}, \mathbf{z} \mid \theta)}{q(\mathbf{z})}\right] \\\\
&\geq \mathbb{E}\_{q(\mathbf{z})}\left[\log \frac{p(\mathbf{x}, \mathbf{z} \mid \theta)}{q(\mathbf{z})}\right] = \mathcal{L}(q, \theta),
\end{aligned}$$
where $\mathcal{L}(q, \theta)$ is the .bold[evidence lower bound] (ELBO).

Writing $p(\mathbf{x}, \mathbf{z} \mid \theta) = p(\mathbf{z} \mid \mathbf{x}, \theta) p(\mathbf{x} \mid \theta)$ gives
$$\log p(\mathbf{x} \mid \theta) = \mathcal{L}(q, \theta) + \text{KL}(q(\mathbf{z}) \\| p(\mathbf{z} \mid \mathbf{x}, \theta)).$$
The gap is zero when $q$ is the posterior $p(\mathbf{z} \mid \mathbf{x}, \theta)$.

---

class: middle

.center.width-70[![](figures/lec8/em-elbo-gap.png)]

.center[For $q$ other than the posterior at $\theta^{(t)}$, the ELBO lies below the log-likelihood.]

---

class: middle

## Expectation-maximization algorithm

EM maximizes the ELBO over $q$ and $\theta$ in turn. Starting from $\theta^{(0)}$, it repeats
- .bold[E-step]: $q^{(t+1)} = \arg\max\_q \mathcal{L}(q, \theta^{(t)})$,
- .bold[M-step]: $\theta^{(t+1)} = \arg\max\_\theta \mathcal{L}(q^{(t+1)}, \theta)$.

---

class: middle

The E-step sets $q$ to the posterior,
$$q^{(t+1)}(\mathbf{z}) = p(\mathbf{z} \mid \mathbf{x}, \theta^{(t)}).$$
.italic[Proof.] We have
$$\mathcal{L}(q, \theta^{(t)}) = \log p(\mathbf{x} \mid \theta^{(t)}) - \text{KL}(q(\mathbf{z}) \\| p(\mathbf{z} \mid \mathbf{x}, \theta^{(t)})),$$
where only the KL divergence depends on $q$. It is smallest, zero, when $q$ is the posterior. □

---

class: middle

.center.width-70[![](figures/lec8/em-elbo-nogap.png)]

.center[With $q$ set to the posterior at $\theta^{(t)}$, the gap at $\theta^{(t)}$ vanishes.]

---

class: middle

The M-step maximizes the expected complete-data log-likelihood,
$$\theta^{(t+1)} = \arg\max\_\theta \mathbb{E}\_{q^{(t+1)}(\mathbf{z})}\left[\log p(\mathbf{x}, \mathbf{z} \mid \theta)\right].$$
.italic[Proof.] We have
$$\begin{aligned}
\mathcal{L}(q^{(t+1)}, \theta) &= \mathbb{E}\_{q^{(t+1)}(\mathbf{z})}\left[\log \frac{p(\mathbf{x}, \mathbf{z} \mid \theta)}{q^{(t+1)}(\mathbf{z})}\right] \\\\
&= \mathbb{E}\_{q^{(t+1)}(\mathbf{z})}\left[\log p(\mathbf{x}, \mathbf{z} \mid \theta) - \log q^{(t+1)}(\mathbf{z})\right] \\\\
&= \mathbb{E}\_{q^{(t+1)}(\mathbf{z})}\left[\log p(\mathbf{x}, \mathbf{z} \mid \theta)\right] - \mathbb{E}\_{q^{(t+1)}(\mathbf{z})}\left[\log q^{(t+1)}(\mathbf{z})\right],
\end{aligned}$$
where the second term does not depend on $\theta$. □

---

class: middle

.center.width-70[![](figures/lec8/em-elbo-mstep.png)]

.center[The M-step moves $\theta^{(t)}$ to the maximum of the ELBO, $\theta^{(t+1)}$.]

---

class: middle

.center.width-70[![](figures/lec8/em-elbo-next.png)]

.center[The next E-step closes the gap at $\theta^{(t+1)}$, where the log-likelihood is higher.]

---

class: middle

.italic[Proposition.] EM never decreases the log-likelihood,
$$\log p(\mathbf{x} \mid \theta^{(t)}) \leq \log p(\mathbf{x} \mid \theta^{(t+1)}) \quad \text{for all } t \geq 0.$$

.italic[Proof.]
$$\begin{aligned}
\log p(\mathbf{x} \mid \theta^{(t)}) &= \mathcal{L}(q^{(t+1)}, \theta^{(t)}) && \text{(the E-step closes the gap)} \\\\
&\leq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) && \text{(the M-step maximizes over } \theta\text{)} \\\\
&\leq \log p(\mathbf{x} \mid \theta^{(t+1)}) && \text{(the ELBO is a lower bound)}. \quad □
\end{aligned}$$

---

class: middle

## EM for the Gaussian mixture

The posterior factorizes over the penguins. By Bayes' rule, the E-step computes the .bold[responsibility] of component $k$ for penguin $i$,
$$\begin{aligned}
r\_{ik} = q^{(t+1)}(z\_i = k) &= p(z\_i = k \mid \mathbf{x}\_i, \theta^{(t)}) \\\\
&= \frac{p(\mathbf{x}\_i \mid z\_i = k, \theta^{(t)}) p(z\_i = k \mid \theta^{(t)})}{\sum\_{j=1}^K p(\mathbf{x}\_i \mid z\_i = j, \theta^{(t)}) p(z\_i = j \mid \theta^{(t)})} \\\\
&= \frac{\pi\_k \mathcal{N}(\mathbf{x}\_i \mid \boldsymbol{\mu}\_k, \boldsymbol{\Sigma}\_k)}{\sum\_{j=1}^K \pi\_j \mathcal{N}(\mathbf{x}\_i \mid \boldsymbol{\mu}\_j, \boldsymbol{\Sigma}\_j)},
\end{aligned}$$
with the parameters $\theta^{(t)}$.

---

class: middle

The M-step maximizes the expected complete-data log-likelihood,
$$\begin{aligned}
\mathbb{E}\_{q^{(t+1)}}\left[\log p(\mathbf{x}\_{1:N}, z\_{1:N} \mid \theta)\right] &= \sum\_{i=1}^N \mathbb{E}\_{q^{(t+1)}(z\_i)}\left[\log p(\mathbf{x}\_i, z\_i \mid \theta)\right] \\\\
&= \sum\_{i=1}^N \sum\_{k=1}^K r\_{ik} \log p(\mathbf{x}\_i, z\_i = k \mid \theta) \\\\
&= \sum\_{i=1}^N \sum\_{k=1}^K r\_{ik} \left[\log \pi\_k + \log \mathcal{N}(\mathbf{x}\_i \mid \boldsymbol{\mu}\_k, \boldsymbol{\Sigma}\_k)\right].
\end{aligned}$$

---

class: middle

Setting the gradients to zero gives the M-step updates. They are the estimates for known components, with each penguin weighted by $r\_{ik}$ and $N\_k = \sum\_{i} r\_{ik}$,
$$\begin{aligned}
\boldsymbol{\mu}\_k^{(t+1)} &= \frac{1}{N\_k} \sum\_{i=1}^N r\_{ik} \mathbf{x}\_i, \\\\
\boldsymbol{\Sigma}\_k^{(t+1)} &= \frac{1}{N\_k} \sum\_{i=1}^N r\_{ik} (\mathbf{x}\_i - \boldsymbol{\mu}\_k^{(t+1)})(\mathbf{x}\_i - \boldsymbol{\mu}\_k^{(t+1)})^T, \\\\
\pi\_k^{(t+1)} &= \frac{N\_k}{N}.
\end{aligned}$$

---

class: middle

.center.width-70[![](figures/lec8/em-bills.gif)]

.center[One frame per iteration, each penguin coloured by its responsibilities. <br>The three components match the species of 96% of the penguins.]

---

class: middle

.center.width-90[![](figures/lec8/em-local.png)]

.center[Two runs of EM from different initial values. The second stops at a local maximum. <br>In practice, EM is run from several initial values, and the best run is kept.]

---

class: middle

## Empirical Bayes

Many data sets hold many similar units, such as restaurants, patients or schools. Each unit $i$ has its own parameter $\theta\_i$, drawn from a prior $p(\theta\_i \mid \alpha)$ shared by all. So far, the hyperparameters $\alpha$ were fixed in advance. Here, the data can estimate them.

.center.width-50[![](figures/lec8/eb-model.svg)]

.italic[Example.] Each restaurant $i$ has a mean rating $\theta\_i \sim \mathcal{N}(\mu, \tau^2)$, observed through $n\_i$ noisy reviews $x\_{ij} \sim \mathcal{N}(\theta\_i, \sigma^2)$. One restaurant says little about $\mu$ and $\tau$, but thousands together pin them down.

---

class: middle

.bold[Empirical Bayes] estimates the hyperparameters by maximizing the marginal likelihood,
$$\hat{\alpha} = \arg\max\_\alpha \log p(\mathbf{x} \mid \alpha) = \arg\max\_\alpha \sum\_{i=1}^N \log \int p(\mathbf{x}\_i \mid \theta\_i) p(\theta\_i \mid \alpha) d\theta\_i,$$
and then infers each $\theta\_i$ with the fitted prior $p(\theta\_i \mid \hat{\alpha})$.

A fully Bayesian treatment would put a prior on $\alpha$ and integrate it out. With many units, the posterior of $\alpha$ is concentrated, and replacing it by $\hat{\alpha}$ changes little.

.alert[The data are used twice, to set the prior and to update it. With few units, empirical Bayes understates the uncertainty.]

---

class: middle

The marginal likelihood has the form of the mixture's, with $\theta\_i$ in the role of $z\_i$ and $\alpha$ in the role of $\theta$. EM applies unchanged.
- E-step: $q^{(t+1)}(\theta\_i) = p(\theta\_i \mid \mathbf{x}\_i, \alpha^{(t)})$ for each unit.
- M-step: $\alpha^{(t+1)} = \arg\max\_\alpha \sum\_{i=1}^N \mathbb{E}\_{q^{(t+1)}(\theta\_i)}\left[\log p(\theta\_i \mid \alpha)\right]$.

The M-step drops $\log p(\mathbf{x}\_i \mid \theta\_i)$ from the expected complete-data log-likelihood, since it does not depend on $\alpha$.

---

class: middle

.italic[Example.] For the restaurants, with $\bar{x}\_i$ the average of the reviews, the E-step gives Gaussian posteriors of variance and mean
$$v\_i = \left(\frac{1}{\tau^2} + \frac{n\_i}{\sigma^2}\right)^{-1}, \qquad m\_i = v\_i \left(\frac{\mu}{\tau^2} + \frac{n\_i \bar{x}\_i}{\sigma^2}\right),$$
The posterior mean $m\_i$ is a weighted average of $\bar{x}\_i$ and $\mu$, with weights $n\_i / \sigma^2$ and $1 / \tau^2$. The M-step sets
$$\mu = \frac{1}{N} \sum\_{i=1}^N m\_i, \qquad \tau^2 = \frac{1}{N} \sum\_{i=1}^N \left[(m\_i - \mu)^2 + v\_i\right].$$

---

class: middle

.center.width-70[![](figures/lec8/eb-restaurants.png)]

.center[EM on 150 simulated restaurants. <br>With few reviews, the posterior mean is pulled towards $\hat{\mu}$. <br>With many, it stays close to the average of the reviews.]

---

class: middle

## Learning diffusion priors by EM

EM can learn a prior given by a diffusion model, from noisy and incomplete observations only.

.center.width-90[![](figures/lec8/diffusion-prior-cifar10.png)]

.footnote[Credits: [Rozet et al](https://arxiv.org/abs/2405.13712), 2024.]

---

class: middle

.center.width-90[![](figures/lec8/diffusion-prior-accelerated-mri.png)]

.footnote[Credits: [Rozet et al](https://arxiv.org/abs/2405.13712), 2024.]

---

class: middle

## Beyond EM

The E-step needs the exact posterior $p(\mathbf{z} \mid \mathbf{x}, \theta)$. When it is intractable, the ELBO can still be maximized over a family of simpler distributions $q$. This is .bold[variational inference].

---

class: end-slide, center
count: false

The end.
