import numpy as np
from scipy.stats import binom
from statsmodels.stats.contingency_tables import mcnemar

from numpy import typing as npt


def mcnemar_test(
    preds1: npt.ArrayLike, preds2: npt.ArrayLike, gts: npt.ArrayLike, test_type: str
):
    """
    Porting the `testcholdout()` function from MATLAB to Python.
    Performs the McNemar's exact [1] and mid-p [2] statistical tests. Some code
    borrowed from [3].

    [1] McNemar, Q. "Note on the Sampling Error of the Difference Between Correlated
        Proportions or Percentages." Psychometrika, Vol. 12, Number 2, 1947, pp.
        153–157.
    [2] Fagerlan, M.W., S. Lydersen, and P. Laake. "The McNemar Test for Binary
        Matched-Pairs Data: Mid-p and Asymptotic Are Better Than Exact Conditional."
        BMC Medical Research Methodology. Vol. 13, 2013, pp. 1–8.
    [3] Schlegel, A. "McNemar's test for paired data with Python".
        https://aaronschlegel.me/mcnemars-test-paired-data-python.html. August 2019.

    Args:
        preds1 (npt.ArrayLike): Set of predictions from 1st model as a 1D NumPy array.
        preds2 (npt.ArrayLike): Set of predictions from 2nd model as a 1D NumPy array.
        gts (npt.ArrayLike): Set of ground truth labels as a 1D NumPy array.
        test_type (str): Type of test; should be in ["exact", "mid-p"].

    Returns:
        float: The calculated p-value.
    """

    # First, check if the predictions and the ground truths are of the same shape.
    try:
        assert preds1.shape == preds2.shape
        assert preds1.shape == gts.shape
    except AssertionError:
        print("Array shape mismatch.")
        exit(1)

    # This function supports the exact statistical test or the mid-p test.
    if test_type not in ["exact", "mid-p"]:
        print("'test_type' must be in ['exact', 'mid-p'].")
        exit(1)

    # Generate a binary array denoting which samples are correctly classified by each
    # model. This is useful because even in the N-class classification task, this step
    # ensures that the contingency matrix is 2x2.
    pred1_correct = 1 * (preds1 == gts)
    pred2_correct = 1 * (preds2 == gts)

    # Creating the contingency matrix.
    a = np.sum((pred1_correct == 1) & (pred2_correct == 1))
    b = np.sum((pred1_correct == 1) & (pred2_correct == 0))
    c = np.sum((pred1_correct == 0) & (pred2_correct == 1))
    d = np.sum((pred1_correct == 0) & (pred2_correct == 0))

    ct: npt.ArrayLike = np.array([[a, b], [c, d]])

    # Calculate the exact p-value.
    i: int = ct[0, 1]
    n: int = ct[1, 0] + ct[0, 1]
    i_n: npt.ArrayLike = np.arange(i + 1, n + 1)

    p_value_exact: float = 2 * (1 - np.sum(binom.pmf(i_n, n, 0.5)))

    if test_type == "exact":
        return p_value_exact
    else:
        mid_p_value: float = p_value_exact - binom.pmf(ct[0, 1], n, 0.5)
        return mid_p_value


def statsmodels_mcnemar(
    preds1: npt.ArrayLike,
    preds2: npt.ArrayLike,
    gts: npt.ArrayLike,
):
    """
    Uses the `statsmodels` library to perform the McNemar's test. Note that this is
    not the mid-p valued test, and is the same as the `exact` p-value test of our
    implementation `mcnemar_test()`.

    Args:
        preds1 (npt.ArrayLike): Set of predictions from 1st model as a 1D NumPy array.
        preds2 (npt.ArrayLike): Set of predictions from 2nd model as a 1D NumPy array.
        gts (npt.ArrayLike): Set of ground truth labels as a 1D NumPy array.

    Returns:
        float: The calculated p-value.
    """

    # First, check if the predictions and the ground truths are of the same shape.
    try:
        assert preds1.shape == preds2.shape
        assert preds1.shape == gts.shape
    except AssertionError:
        print("Array shape mismatch.")
        exit(1)

    # Generate a binary array denoting which samples are correctly classified by each
    # model. This is useful because even in the N-class classification task, this step
    # ensures that the contingency matrix is 2x2.
    pred1_correct = 1 * (preds1 == gts)
    pred2_correct = 1 * (preds2 == gts)

    # Creating the contingency matrix.
    a = np.sum((pred1_correct == 1) & (pred2_correct == 1))
    b = np.sum((pred1_correct == 1) & (pred2_correct == 0))
    c = np.sum((pred1_correct == 0) & (pred2_correct == 1))
    d = np.sum((pred1_correct == 0) & (pred2_correct == 0))

    ct: npt.ArrayLike = np.array([[a, b], [c, d]])

    # Perform the McNemar's test.
    result = mcnemar(ct, exact=True)

    return result.pvalue
