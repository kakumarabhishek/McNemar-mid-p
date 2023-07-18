# McNemar's mid-p Statistical Test

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the Python implementation of the McNemar's mid-p statistical significance test for comparing the predictions from 2 models.

While `statsmodels` provides the functionality for the McNemar's test, it only implements the exact version of the test and not the mid-p version. The latter is provided by MATLAB in the `testcholdout()` function, and this repository aims to implement the same in Python.