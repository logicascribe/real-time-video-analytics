import pytest
from pytest import approx

def isMatrix(python_list_of_lists):
    # Maps len() over every sublist and checks if there is exactly 1 unique length
    return 0 == len(python_list_of_lists) or len(set(len(row) for row in python_list_of_lists)) == 1


def checkMatrixDimEquality(matA, matB):
    if (len(matA) != len(matB)):
        return False
    nbRow = len(matA)
    if (0 == nbRow):
        return True
    return isMatrix(matA) and isMatrix(matB) and (len(matA[0]) == len(matB[0]))

def checkMatrixEqualityApprox(matA, matB):
    if not checkMatrixDimEquality(matA, matB):
        return False
    nbRow = len(matA)
    if (0 == nbRow):
        return True
    nbCol = len(matA[0])

    for itRow in range(nbRow):
            if not (matA[itRow] == approx(matB[itRow])):
                return False
    return True
