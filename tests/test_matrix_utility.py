import pytest
from pytest import approx
import tests.matrix_utility
from .matrix_utility import isMatrix, checkMatrixDimEquality, checkMatrixEqualityApprox


TEST_MATRIX_VALID = [
    pytest.param([], [],
                 [True, True, True],
                 id = "2_empty_matrix"),
    pytest.param([], [ [1] ],
                 [True, False, False],
                 id = "1_empty__&_single_elem_matrix"),
    pytest.param([[1]], [[1]],
                 [True, True, True],
                 id = "2_single_value_matrix"),
    pytest.param([[1, 2],  [3, 4]],
                 [[1, 2],  [3, 4]],
                 [True, True, True],
                 id = "2_dummy_2*2_equal_matrix"),
    pytest.param([[1, 2],  [3, 4]],
                 [[1, 3],  [2, 4]],
                 [True, True, False],
                 id = "2_dummy_2*2_diff_matrix"),
    ]

@pytest.mark.parametrize("matA, matB, expected_results", TEST_MATRIX_VALID,)
def test_valid_matrix_checking(matA, matB, expected_results):
    resMat = []
    resMat.append(isMatrix(matA))
    resMat.append(checkMatrixDimEquality(matA, matB))
    resMat.append(checkMatrixEqualityApprox(matA, matB))
    assert(resMat == expected_results)
