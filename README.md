# DATS0001 Foundations of Data Science

Materials for DATS0001 Foundations of Data Science, ULiège, Fall 2026.

- Instructor: Gilles Louppe
- Teaching assistant: Matthias Pirlet
- When: Monday 1:30 PM
- Classroom: B28/1.21
- Discord: [https://discord.gg/qpvh4ueEFQ](https://discord.gg/qpvh4ueEFQ)

## Agenda

| Date | Topic |
| --- | --- |
| September 14 | [Course syllabus](https://glouppe.github.io/dats0001-foundations-of-data-science/?p=course-syllabus.md) [[PDF](https://glouppe.github.io/dats0001-foundations-of-data-science/pdf/course-syllabus.pdf)]<br> Lecture 1: [Build, compute, critique, repeat](https://glouppe.github.io/dats0001-foundations-of-data-science/?p=lecture1.md) [[PDF](https://glouppe.github.io/dats0001-foundations-of-data-science/pdf/lec1.pdf)]<br>`nb01`: Build, compute, critique, repeat [[notebook](./nb01-box-loop.ipynb)]<br>Reading: Gelman, Vehtari, McElreath et al, [Bayesian Workflow](https://avehtari.github.io/Bayesian-Workflow/), 2026 [Chapter 2]<br>Reading: Blei, [Build, Compute, Critique, Repeat](http://www.cs.columbia.edu/~blei/fogm/2020F/readings/Blei2014.pdf), 2014 [Section 1]<br>Reading: Box, [Science and Statistics](https://www.jstor.org/stable/2286841), 1976 |
| September 21 | Lecture 2: Data and exploratory analysis<br>`nb02a`: Arrays with NumPy<br>`nb02b`: Arrays with JAX<br>`nb03b`: Data wrangling with Pandas<br>`nb02d`: Exploratory data analysis |
| September 28 | Lecture 3: Visualization<br>`nb03`: Plots with Matplotlib<br>Reading: Rougier et al, [Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003833&type=printable), 2014 |
| October 5 | Lecture 4: Latent variable models<br>`nb04b`: Probabilistic PCA<br>Reading: Blei, [Build, Compute, Critique, Repeat](http://www.cs.columbia.edu/~blei/fogm/2020F/readings/Blei2014.pdf), 2014 [Sections 1-3] |
| October 12 | Lecture 5: State-space models<br>`nb05`: State-space models |
| October 19 | Lecture 6: Markov chain Monte Carlo<br>`nb06a`: Markov chains <br>`nb06b`: MCMC<br>Reading: Gelman et al, [Bayesian Data Analysis, 3rd](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf), 2021 [Chapter 11] |
| October 26 | _No lecture (Toussaint break)_ |
| November 2 | Lecture 7: Model criticism and validation<br>`nb07a`: Model checking<br>`nb07b`: Model comparison<br>`nb07c`: Bayesian Occam's razor  |
| November 9 | _No lecture_ |
| November 16 | Lecture 8: Expectation-maximization<br>`nb08`: Expectation-maximization<br>Reading: Dempster et al, [Maximum Likelihood from Incomplete Data via EM](https://www.jstor.org/stable/2984875), 1977 |
| November 23 | Lecture 9: Variational inference<br>`nb09a`: Coordinate ascent variational inference<br>`nb09b`: Automatic differentiation variational inference<br>Reading: Kucukelbir et al, [Automatic differentiation variational inference](https://arxiv.org/abs/1603.00788), 2016 |
| November 30 | Lecture 10: Simulation-based inference |
| December 7 | Lecture 11: Wrap-up case study<br>`nb11`: Estimating air pollution from satellite data |
| December 14 | _No lecture_ |

## Homework

(TBD.)

## Setup

The notebooks run in a Python environment managed by [uv](https://docs.astral.sh/uv/). Install uv:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then clone the repository and create the environment:

```bash
git clone https://github.com/glouppe/dats0001-foundations-of-data-science.git
cd dats0001-foundations-of-data-science
uv sync
```

`uv sync` downloads Python 3.13 and installs the exact package versions pinned in `uv.lock` into a local `.venv/` directory. Launch Jupyter with

```bash
uv run jupyter lab
```

or, in VS Code, select `.venv` as the notebook kernel. To get new materials during the semester, run `git pull` followed by `uv sync`.

## Slides

The slides are Markdown files rendered in the browser. To view them locally, serve the repository with Python's built-in web server from its root directory:

```bash
uv run python -m http.server
```

Then open [http://localhost:8000/?p=lecture1.md](http://localhost:8000/?p=lecture1.md), replacing `lecture1.md` with the lecture you want. Opening `index.html` directly from disk does not work, since browsers block it from loading the Markdown file.
