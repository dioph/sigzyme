#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import setuptools

version = re.search(
    '^__version__\\s*=\\s*"(.*)"', open("src/sigzyme/__init__.py").read(), re.M
).group(1)

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "torch",
    "numba",
]

extras_require = {
    "docs": [
        "jupyter",
        "myst-nb<0.11",
        "numpydoc",
        "sphinx_rtd_theme",
    ],
    "test": [
        "black==20.8b1",
        "flake8",
        "isort",
        "numpy",
        "pytest",
        "pytest-cov",
        "scipy>0.14",
        "tox",
    ],
}

setuptools.setup(
    name="sigzyme",
    version=version,
    author="Eduardo Nunes",
    author_email="dioph@pm.me",
    license="MIT",
    description="Efficient batch decomposition of time series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/sigzyme",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
