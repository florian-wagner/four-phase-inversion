import numpy as np
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sgetrf, sgetrs


def trans_dot(A):
    """
    Return dot product A^T * A exploiting matrix symmetry, i.e. only
    calculating the upper triangle of A^T * A. Makes use of the ssyrk blas
    routine.
    """
    ATA = ssyrk(1.0, A, trans=1)
    ATA += ATA.T
    ATA[np.diag_indices_from(ATA)] /= 2
    return ATA


def calc_R(J, C, alpha=0.5):
    """
    Calculates the formal model resolution matrix deterministically following:

    .. math::

        \mathbf{R_m} = (\mathbf{J}^T\mathbf{D}^T\mathbf{D}\mathbf{J} + \alpha
        \mathbf{C}^T\mathbf{C})^{-1}
        \mathbf{J}^T\mathbf{D}^T\mathbf{D}\mathbf{J}

    .. note::

        The current implementation assumes that :math:`\mathbf{D}` is the
        identitiy matrix, i.e. equal data weights.

    Parameters
    ----------
    J : array
        Jacobian matrix.
    C : array
        Constraint matrix.
    alpha : float
        Regularization strength :math:`\alpha`.
    """

    JTJ = trans_dot(J)
    CMinv = trans_dot(C)
    RT = JTJ + alpha * CMinv

    # SGETRF computes an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges.
    lu, piv, _ = sgetrf(np.asfortranarray(RT))

    # Backsubstitution
    R, _ = sgetrs(lu, piv, JTJ)

    return R
