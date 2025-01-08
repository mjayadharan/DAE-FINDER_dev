from setuptools import setup, find_packages

setup(
    name="DaeFinder",
    version="0.1.0",
    description="A Python package to discover Differential Algebraic Equations from data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mjayadharan/DAE-FINDER_dev",
    author="Manu Jayadharan",
    author_email="manu.jayadharan@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "sympy",
        "scikit-learn",
        "matplotlib",
        "joblib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
