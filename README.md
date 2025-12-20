## DaeFinder

**DaeFinder** is a Python-based Scientific ML package for discovering **Differential Algebraic Equations (DAEs)** from noisy data using a sparse optimization framework. The package is based on the **SODAs algorithm**, described in the accompanying publication: https://arxiv.org/abs/2503.05993.

### Model-agnostic and extensible design

A key design principle of DaeFinder is that it is **model-agnostic**. Any fitting statistical or machine-learning method can be used to fit algebraic and differential equations, as long as it conforms to the package’s abstract interface by implementing the required `fit()` and `score()` methods. This makes it easy to integrate linear models, nonlinear regressors, probabilistic models, or fully custom estimators without modifying the core code.

DaeFinder is also **modular and extensible**, allowing you to:
- Define **custom model classes** for representing algebraic or differential equation structures
- Use **custom optimizers** in place of the default sparse optimization routines
- Mix and match **fitting strategies** and **scoring metrics** depending on the application

### Citation

If you use DaeFinder for development or research, please cite the SODAs paper:

- https://arxiv.org/abs/2503.05993
---

## **Author and Contributors**
- Manu Jayadharan
- Christina Catlett
- Arthur Montanari
- Grace Hooper
- Niall Mangan
- Finn Hagerty
- Yuxiang Feng
---

## **Features**
- Decoupling of Algebraic and Dynamic Equations
- Smoothening noisy data and calculating derivatives.
- Generate polynomial features for regression models.
- Support for sparse feature coupling.
- SVD Analysis.
- Example notebooks for practical demonstrations including chemical reaction networks, power grid networks, etc. 

---

### **Dependencies**

The following Python packages are required to use `DaeFinder`:

- `numpy`
- `scipy`
- `pandas`
- `sympy`
- `scikit-learn`
- `matplotlib`
- `joblib`

## **Installation**

To install the `DaeFinder` package, follow these steps:

1. Ensure you have Python 3.7 or higher installed.
2. Install the package using pip:
   ```bash
   pip install DaeFinder

## Examples

Walkthrough notebooks are available in the `Examples/` folder of the repository. These notebooks include:

- A step-by-step guide to using DaeFinder.
- Application to chemical reaction network, non-linear pendulum, power grid, etc. 

For examples that require additional data (e.g., the power grid example), the data files are included in the GitHub repository. Be sure to download the required datasets from the relevant folders in the repository.

# Specific Example-Dependency

- Some example notebooks require specific dependencies.
- In the power-grid example, Matpower 6.0 (https://matpower.org/download/) is required for the power flow calculation.

## Known Issues

- The parallel function currently has some bugs that need fixing.
- If you encounter issues with the installation or the package itself, please feel free to contact the authors or contributors.

## Contributing

We welcome contributions to improve DaeFinder! If you are interested in contributing to the package or working on related research, please reach out to the author or the Mangan Group.

## Contact

For any questions, issues, or collaboration inquiries, please contact:

- Manu Jayadharan [manu.jayadharan@gmail.com](mailto:manu.jayadharan@gmail.com), [manu.jayadharan@northwestern.edu](mailto:manu.jayadharan@gmail.com)
- Niall Mangan [niall.mangan@northwestern.edu](mailto:niall.mangan@northwestern.edu)
