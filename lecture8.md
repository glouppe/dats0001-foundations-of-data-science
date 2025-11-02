class: middle, center, title-slide

# Foundations of Data Science

Lecture 8: Variational inference

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](g.louppe@uliege.be)

---

why is EM not enough -- it requires a tractable posterior

---

class: middle

# Variational inference

---

approximate the posterior with a tractable distribution
rederive the elbo from the KL(q(z) || p(z|x,theta))
mean-field variational inference 
automatic differentiation variational inference (ADVI)
diagnostics

---

class: middle

# Simulation-based inference

---

intractable densities
amortized inference 
neural posterior estimation (others are interesting but not directly VI approaches)
diagnostics

---

case study on exoplanet characterization

---

case study on cytometry data and cell population identification

---

case study on data assimilation

---

class: end-slide, center
count: false

The end.
