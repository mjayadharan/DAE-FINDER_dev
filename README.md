<div align="center">

# 🔎 DaeFinder

**Discover the hidden equations inside your data.**

DaeFinder is a Scientific Machine Learning toolkit that recovers
**Differential Algebraic Equations (DAEs)** from noisy measurements using a
sparse-optimization framework — the **SODAs** algorithm.

[![PyPI version](https://img.shields.io/pypi/v/DaeFinder.svg)](https://pypi.org/project/DaeFinder/)
[![Python](https://img.shields.io/badge/python-3.9_--_3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2503.05993-b31b1b.svg)](https://arxiv.org/abs/2503.05993)

</div>

---

## ✨ Why DaeFinder?

Real systems — chemical reaction networks, power grids, mechanical systems —
are governed by a mix of **differential** equations (how things change in time)
and **algebraic** constraints (relationships that always hold). DaeFinder
automatically untangles both directly from time-series data:

- 🧩 **Decouples algebraic & dynamic relations** — finds the constraints *and* the ODEs.
- 🪶 **Model-agnostic by design** — plug in any estimator that implements `fit()` and `score()` (linear models, regularized regressors, or your own custom optimizer).
- 🔇 **Robust to noise** — built-in smoothing and derivative estimation (splines, Savitzky–Golay).
- 🧮 **Rich feature engineering** — polynomial libraries, sparse feature coupling, and SVD analysis.
- ⚡ **Parallel discovery** — fit many candidate relations concurrently.
- ✅ **Tested & current** — runs on the latest scientific-Python stack (see [Compatibility](#-compatibility)).

---

## 📝 Citation

If DaeFinder supports your research, please cite the SODAs paper:

> M. Jayadharan, C. Catlett, A. N. Montanari, and N. M. Mangan,
> *"SODAs: Sparse Optimization for the Discovery of Differential and Algebraic
> Equations."* Proc. A 1 May 2026; 482 (2337): 20250201. https://doi.org/10.1098/rspa.2025.0201 

---


## 📦 Installation

DaeFinder requires **Python 3.9+**.

```bash
pip install DaeFinder
```

<details>
<summary>Install the latest development version from source</summary>

```bash
git clone https://github.com/mjayadharan/DAE-FINDER_dev.git
cd DAE-FINDER_dev
pip install -e .
```
</details>

**Dependencies** (installed automatically): `numpy`, `scipy`, `pandas`,
`sympy`, `scikit-learn`, `matplotlib`, `joblib`.

---

## 🚀 Quick start

```python
import pandas as pd
from daeFinder import PolyFeatureMatrix, AlgModelFinder

# Your measurements: one column per state variable, one row per time point.
data = pd.DataFrame({"S": ..., "E": ..., "ES": ..., "P": ...})

# 1) Build a candidate library of nonlinear (polynomial) features.
library = PolyFeatureMatrix(degree=2).fit_transform(data)

# 2) Discover sparse algebraic relations among the library terms.
finder = AlgModelFinder(model_id="lasso", alpha=0.01)
finder.fit(library, scale_columns=True)

# ...or fit every candidate in parallel:
finder.fit(library, scale_columns=True, parallelize=True, num_cpu=8)

# 3) Rank and inspect the strongest recovered relations.
print(finder.best_models(num=5))
```

Working from noisy data? Smooth it and estimate derivatives first:

```python
from daeFinder import smooth_data

smoothed = smooth_data(data, smooth_method="spline", noise_perc=2, derr_order=1)
```

See the [`Examples/`](Examples/) folder for full, runnable walkthroughs.

---

## 🧠 How it works

DaeFinder follows the **SODAs** pipeline:

1. **Smooth & differentiate** noisy measurements to obtain clean state variables and their time derivatives.
2. **Construct a candidate library** of nonlinear terms (e.g. polynomial features, coupled features).
3. **Sparsely regress** library terms against one another to surface algebraic constraints, then against derivatives to surface the dynamics.
4. **Refine & simplify** the discovered relations into interpretable symbolic equations.

---

## 🔧 Compatibility

DaeFinder is continuously tested across the modern scientific-Python stack:

| Component | Supported |
|-----------|-----------|
| Python | 3.9 – 3.14 |
| NumPy | 1.x and 2.x |
| pandas | 2.x and 3.x |
| scikit-learn | ≥ 1.2 (incl. 1.9) |

A GitHub Actions matrix runs the full test suite on every supported Python
version. (Recent releases resolved Python 3.13+ `exec()`/PEP 667, NumPy 2,
pandas 3 copy-on-write, and scikit-learn ≥ 1.7 incompatibilities.)

---

## 🧪 Testing

The package ships with a comprehensive regression suite under [`tests/`](tests/).

```bash
pip install -r tests/requirements-test.txt

pytest                     # run everything
./tests/run_tests.sh       # run all tests + save a timestamped report
```

See [`tests/README.md`](tests/README.md) for how to run the suite, read its
reports, and add new tests.

---

## 📚 Examples

Step-by-step notebooks live in [`Examples/`](Examples/), covering:

- A guided **walkthrough** of the discovery pipeline.
- **Chemical reaction networks** (Michaelis–Menten enzyme kinetics).
- **Nonlinear & double pendulums.**
- **Power-grid networks.**

> Some examples need extra data or tools. Data files are included in the
> repository (download the relevant folders). The power-grid example also
> requires [Matpower 6.0](https://matpower.org/download/) for power-flow
> calculations.

---

## 👥 Authors & contributors

Manu Jayadharan · Christina Catlett · Arthur Montanari · Grace Hooper ·
Niall Mangan · Finn Hagerty · Yuxiang Feng

Developed with the Mangan Group at Northwestern University.

## 🤝 Contributing

Contributions are welcome! Whether it's a bug report, a feature, a new example,
or related research collaboration, please open an issue or reach out to the
authors or the Mangan Group.

## 📬 Contact

- **Manu Jayadharan** — [manu.jayadharan@gmail.com](mailto:manu.jayadharan@gmail.com) · [manu.jayadharan@northwestern.edu](mailto:manu.jayadharan@northwestern.edu)
- **Niall Mangan** — [niall.mangan@northwestern.edu](mailto:niall.mangan@northwestern.edu)

---

<div align="center">
Released under the <a href="LICENSE">MIT License</a>.
</div>
