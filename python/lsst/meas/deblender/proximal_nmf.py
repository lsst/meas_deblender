from __future__ import print_function, division
import numpy as np, scipy.sparse, scipy.sparse.linalg
from functools import partial

import logging

logging.basicConfig()
logger = logging.getLogger("lsst.meas.deblender.proximal_nmf")

# identity
def prox_id(X, step):
    return X

# projection onto 0
def prox_zero(X, step):
    return np.zeros_like(X)

# hard thresholding: X if X >= k, otherwise 0
# NOTE: modifies X in place
def prox_hard(X, step, l=0):
    below = X - l*step < 0
    X[below] = 0
    return X

# projection onto non-negative numbers
def prox_plus(X, step):
    return prox_hard(X, step, l=0)

# projection onto numbers above l
# NOTE: modifies X in place
def prox_min(X, step, l=0):
    below = X - l*step < 0
    X[below] = l*step
    return X

# soft thresholding operator
def prox_soft(X, step, l=0):
    return np.sign(X)*prox_plus(np.abs(X) - l*step, step)

# same but with projection onto non-negative
def prox_soft_plus(X, step, l=0):
    return prox_plus(prox_soft(X, step, l=l), step)

# projection onto sum=1 along each axis
def prox_unity(X, step, axis=0):
    return X / np.sum(X, axis=axis, keepdims=True)

# same but with projection onto non-negative
def prox_unity_plus(X, step, axis=0):
    return prox_unity(prox_plus(X, step), step, axis=axis)

def l2sq(x):
    return (x**2).sum()

def l2(x):
    return np.sqrt((x**2).sum())

def convolve_band(P, I):
    if isinstance(P, list) is False:
        return P.dot(I.T).T
    else:
        PI = np.empty(I.shape)
        B = I.shape[0]
        for b in range(B):
            PI[b] = P[b].dot(I[b])
        return PI

def get_peak_model(A, S, Tx, Ty, P=None, shape=None, k=None):
    """Get the model for a single source
    """
    # Allow the user to send full A,S, ... matrices or matrices for a single source
    if k is not None:
        Ak = A[:, k]
        Sk = S[k]
        if Tx is not None or Ty is not None:
            Txk = Tx[k]
            Tyk = Ty[k]
    else:
        Ak, Sk, Txk, Tyk = A, S.copy(), Tx, Ty
    # Check for a flattened or 2D array
    if len(Sk.shape)==2:
        Sk = Sk.flatten()
    B,N = Ak.shape[0], Sk.shape[0]
    model = np.zeros((B,N))

    # NMF without translation
    if Tx is None or Ty is None:
        if Tx is not None or Ty is not None:
            raise ValueError("Expected Tx and Ty to both be None or neither to be None")
        for b in range(B):
            if P is None:
                model[b] = A[b]*Sk
            else:
                model[b] = Ak[b] * P[b].dot(Sk)
    # NMF with translation
    else:
        if P is None:
            Gamma = Tyk.dot(Txk)
        for b in range(B):
            if P is not None:
                Gamma = Tyk.dot(P[b].dot(Txk))
            model[b] = Ak[b] * Gamma.dot(Sk)
    # Reshape the image into a 2D array
    if shape is not None:
        model = model.reshape((B, shape[0], shape[1]))
    return model

def get_model(A, S, Tx, Ty, P=None, shape=None):
    """Build the model for an entire blend
    """
    B,K,N = A.shape[0], A.shape[1], S.shape[1]
    if len(S.shape)==3:
        N = S.shape[1]*S.shape[2]
        S = S.reshape(K,N)
    model = np.zeros((B,N))

    if Tx is None or Ty is None:
        if Tx is not None or Ty is not None:
            raise ValueError("Expected Tx and Ty to both be None or neither to be None")
        model = A.dot(S)
        if P is not None:
            model = convolve_band(P, model)
        if shape is not None:
            model = model.reshape(B, shape[0], shape[1])
    else:
        for pk in range(K):
            for b in range(B):
                if P is None:
                    Gamma = Ty[pk].dot(Tx[pk])
                else:
                    Gamma = Ty[pk].dot(P[b].dot(Tx[pk]))
                model[b] += A[b,pk]*Gamma.dot(S[pk])
    if shape is not None:
        model = model.reshape(B, shape[0], shape[1])
    return model

def delta_data(A, S, Y, Gamma, D, W=1):
    """Gradient of model with respect to A or S
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    B,K,N = A.shape[0], A.shape[1], S.shape[1]
    # We need to calculate the model for each source individually and sum them
    model = np.zeros((B,N))
    for pk in range(K):
        for b in range(B):
            model[b] += A[b,pk]*Gamma[pk][b].dot(S[pk])
        #plt.imshow(A[b,pk]*Gamma[pk][b].dot(S[pk]).reshape(71,104))
        #plt.show()
    
    #plt.imshow(model[2].reshape(71,104))
    #plt.title("model")
    #plt.show()
    
    diff = W*(model-Y)
    
    if D == 'S':
        result = np.zeros((K,N))
        for pk in range(K):
            for b in range(B):
                result[pk] += A[b,pk]*Gamma[pk][b].T.dot(diff[b])
    elif D == 'A':
        result = np.zeros((B,K))
        for pk in range(K):
            for b in range(B):
                result[b][pk] = diff[b].dot(Gamma[pk][b].dot(S[pk]))
    else:
        raise ValueError("Expected either 'A' or 'S' for variable `D`")
    return result


def grad_likelihood_A(A, S, Y, Gamma=None, W=1):
    return delta_data(A, S, Y, D='A', Gamma=Gamma, W=W)

def grad_likelihood_S(S, A, Y, Gamma=None, W=1):
    return delta_data(A, S, Y, D='S', W=W, Gamma=Gamma)

# executes one proximal step of likelihood gradient, folloVed by prox_g
def prox_likelihood_A(A, step, S=None, Y=None, prox_g=None, W=1, Gamma=None):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, W=W, Gamma=Gamma), step)

def prox_likelihood_S(S, step, A=None, Y=None, prox_g=None, W=1, Gamma=None):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, W=W, Gamma=Gamma), step)

def dot_components(C, X, axis=0, transpose=False):
    """Apply a linear constraint C to each peak in X
    """
    K = X.shape[axis]

    if axis == 0:
        if not transpose:
            CX = [C.dot(X[k]) for k in range(K)]
        else:
            CX = [C.T.dot(X[k]) for k in range(K)]
    if axis == 1:
        if not transpose:
            CX = [C.dot(X[:,k]) for k in range(K)]
        else:
            CX = [C.T.dot(X[:,k]) for k in range(K)]
    return np.stack(CX, axis=axis)

# accelerated proximal gradient method
# Combettes 2009, Algorithm 3.6
def APGM(X, prox, step, e_rel=1e-6, max_iter=1000):
    Z = X.copy()
    t = 1.
    for it in range(max_iter):
        X_ = prox(Z, step)
        t_ = 0.5*(1 + np.sqrt(4*t*t + 1))
        gamma = 1 + (t - 1)/t_
        Z = X + gamma*(X_ - X)

        # test for fixed point convergence
        if l2sq(X - X_) <= e_rel**2*l2sq(X):
            X[:] = X_[:]
            break

        t = t_
        X[:] = X_[:]

    return it

def check_NMF_convergence(it, newX, oldX, e_rel, K, min_iter=10):
    """Check the that NMF converges
    
    Uses the check from Langville 2014, Section 5, to check if the NMF
    deblender has converged
    """
    norms = np.zeros((K, 2))
    norms[:,0] = [newX[k].dot(oldX[k]) for k in range(K)]
    norms[:,1] = [l2sq(oldX[k]) for k in range(K)]

    convergent = it > min_iter and np.all([ct >= (1-e_rel**2)*o2 for ct,o2 in norms])
    return convergent, norms

def get_variable_errors(A, AX, Z, U, e_rel):
    """Get the errors in a single multiplier method step
    
    For a given linear operator A, (and its dot product with X to save time),
    calculate the errors in the prime and dual variables, used by the
    Boyd 2011 Section 3 stopping criteria.
    """
    e_pri2 = e_rel**2*np.max([l2sq(AX), l2sq(Z), 1])
    if A is None:
        e_dual2 = e_rel**2*l2sq(U)
    else:
        e_dual2 = e_rel**2*l2sq(dot_components(A, U, transpose=True))
    return e_pri2, e_dual2

def get_linearization(C, X, Z, U):
    """Get the quadratic regularization term for a linear operator
    
    Using the method of augmented Lagrangians requires a quadratic regularization used
    to updated the X matrix in ADMM. This method calculates those terms.
    """
    return dot_components(C, dot_components(C, X) - Z + U, transpose=True)

# Alternating direction method of multipliers
# initial: initial guess of solution
# K: number of iterations
# A: KxN*M: minimizes f(X) + g(AX) for each component k (i.e. rows of X)
# See Boyd+2011, Section 3, with arbitrary A, B=-Id, c=0
def ADMM(X0, prox_f, step_f, prox_g, step_g, A=None, max_iter=1000, e_rel=1e-3):

    if A is None:
        U = np.zeros_like(X0)
        Z = X0.copy()
    else:
        X = X0.copy()
        Z = dot_components(A, X)
        U = np.zeros_like(Z)

    errors = []
    for it in range(max_iter):
        if A is None:
            X = prox_f(Z - U, step_f)
            AX = X
        else:
            X = prox_f(X - (step_f/step_g)[:,None] * get_linearization(A, X, Z, U), step_f)
            AX = dot_components(A, X)
        Z_ = prox_g(AX + U, step_g)
        # this uses relaxation parameter of 1
        U = U + AX - Z_

        # compute prime residual rk and dual residual sk
        R = AX - Z_
        if A is None:
            S = -(Z_ - Z)
        else:
            S = -(step_f/step_g)[:,None] * dot_components(A, Z_ - Z, transpose=True)
        Z = Z_

        # stopping criteria from Boyd+2011, sect. 3.3.1
        # only relative errors
        e_pri2, e_dual2 = get_variable_errors(A, AX, Z, U, e_rel)

        # Store the errors
        errors.append([[e_pri2, e_dual2, l2sq(R), l2sq(S)]])

        if l2sq(R) <= e_pri2 and l2sq(S) <= e_dual2:
            break

    return it, X, Z, U, errors

def update_sdmm_variables(X, Y, Z, prox_f, step_f, proxOps, proxSteps, constraints):
    """Update the prime and dual variables for multiple linear constraints
    
    Both SDMM and GLMM require the same method of updating the prime and dual
    variables for the intensity matrix linear constraints.
    
    """
    linearization = [step_f/proxSteps[i] * get_linearization(c, X, Y[i], Z[i])
                     for i, c in enumerate(constraints)]
    X_ = prox_f(X - np.sum(linearization, axis=0), step=step_f)
    # Iterate over the different constraints
    CX = []
    Y_ = Y.copy()
    for i in range(len(constraints)):
        # Apply the constraint for each peak to the peak intensities
        CXi = dot_components(constraints[i], X_)
        # TODO: the following is wrong!
        #CXi = dot_components(constraints[i], X)
        Y_[i] = proxOps[i](CXi+Z[i], step=proxSteps[i])
        Z[i] = Z[i] + CXi - Y_[i]
        CX.append(CXi)
    return X_ ,Y_, Z, CX

def test_multiple_constraint_convergence(step_f, step_g, X, CX, Z_, Z, U, constraints, e_rel):
    """Calculate if all constraints have converged
    
    Using the stopping criteria from Boyd 2011, Sec 3.3.1, calculate whether the
    variables for each constraint have converged.
    """
    # compute prime residual rk and dual residual sk
    R = [cx-Z_[i] for i, cx in enumerate(CX)]
    S = [-(step_f/step_g[i]) * dot_components(c, Z_[i] - Z[i], transpose=True)
         for i, c in enumerate(constraints)]
    # Calculate the error for each constraint
    errors = np.zeros((len(constraints), 4))
    errors[:,:2] = np.array([get_variable_errors(c, CX[i], Z[i], U[i], e_rel)
                                for i, c in enumerate(constraints)])
    errors[:,2] = [l2sq(r) for r in R]
    errors[:,3] = [l2sq(s) for s in S]

    # Check the constraints for convergence
    convergence = [e[2]<=e[0] and e[3]<=e[1] for e in errors]
    return np.all(convergence), errors

def SDMM(X0, prox_f, step_f, prox_g, step_g, constraints, max_iter=1000, e_rel=1e-3):
    """Implement Simultaneous-Direction Method of Multipliers
    
    This implements the SDMM algorithm derived from Algorithm 7.9 from Combettes and Pesquet (2009),
    Section 4.4.2 in Parikh and Boyd (2013), and Eq. 2.2 in Andreani et al. (2007).
    
    In Combettes and Pesquet (2009) they use a matrix inverse to solve the problem.
    In our case that is the inverse of a sparse matrix, which is no longer sparse and too
    costly to implement.
    The `scipy.sparse.linalg` module does have a method to solve a sparse matrix equation,
    using Algorithm 7.9 directly still does not yield the correct result,
    as the treatment of penalties due to constraints are on equal footing with our likelihood
    proximal operator and require a significant change in the way we calculate step sizes to converge.
    
    Instead we calculate the constraint vectors (as in SDMM) but extend the update of the ``X`` matrix
    using a modified version of the ADMM X update function (from Parikh and Boyd, 2009),
    using an augmented Lagrangian for multiple linear constraints as given in Andreani et al. (2007).
    
    In the language of Combettes and Pesquet (2009), constraints = list of Li,
    proxOps = list of ``prox_{gamma g i}``.
    """
    # Initialization
    X = X0.copy()
    N,M = X0.shape
    Z = np.zeros((len(constraints), N, M))
    U = np.zeros_like(Z)
    for c, C in enumerate(constraints):
        Z[c] = dot_components(C,X)

    # Update the constrained matrix
    all_errors = []
    for n in range(max_iter):
        # Update the variables
        X_, Z_, U, CX = update_sdmm_variables(X, Z, U, prox_f, step_f, prox_g, step_g, constraints)
        # ADMM Convergence Criteria, adapted from Boyd 2011, Sec 3.3.1
        result = test_multiple_constraint_convergence(step_f, step_g, X, CX, Z_, Z,
                                                      U, constraints, e_rel)

        convergence, errors = result
        all_errors.append(errors)

        X = X_
        Z = Z_
        if convergence:
            break
    return n, X, Z, U, all_errors

def GLMM(shape, data, X10, X20, peaks, W, P,
        prox_f1, prox_f2, prox_g1, prox_g2,
        constraints1, constraints2, lM1, lM2, max_iter=1000, e_rel=1e-3, beta=1, min_iter=20):
    """ Solve for both the SED and Intensity Matrices at the same time
    """
    # Initialize SED matrix
    X1 = X10.copy()
    N1, M1 = X1.shape
    # TODO: Allow for constraints
    #Y1 = np.zeros((len(constraints1), N1, M1))
    #Z1 = np.zeros_like(Y1)
    Z1 = X10.copy()
    U1 = np.zeros_like(Z1)

    # Initialize Intensity matrix
    X2 = X20.copy()
    N2, M2 = X2.shape
    Z2 = np.zeros((len(constraints2), N2, M2))
    U2 = np.zeros_like(Z2)

    # Initialize Other Parameters
    K = X2.shape[0]
    if W is not None:
        W_max = W.max()
    else:
        W = W_max = 1
    
    # Initialize the translation operators
    Tx = []
    Ty = []
    cx, cy = int(shape[1]/2), int(shape[0]/2)
    for pk, (px, py) in enumerate(peaks):
        dx = cx - px
        dy = cy - py
        tx, ty, _ = getTranslationOp(dx, dy, shape, threshold=1e-8)
        Tx.append(tx)
        Ty.append(ty)
    
    # TODO: This is only temporary until we fit for dx, dy
    G = []
    for pk in range(K):
        if P is None:
            gamma = [Ty[pk].dot(Tx[pk])]*N1
        else:
            gamma = []
            for b in range(N1):
                g = Ty[pk].dot(P[b].dot(Tx[pk]))
                gamma.append(g)
        G.append(gamma)
    
    # Evaluate the solution
    logger.info("Beginning Loop")

    all_norms = []
    all_errors = []
    # Used for Langville 2014 Frobenius norm
    # Not currently implemented (see comment in loop)
    #trData = np.trace(data.T.dot(data))
    for it in range(max_iter):
        # Step updates might need to be fixed, this is just a guess using the step updates from ALS
        step_f1 = beta**it / lipschitz_const(X2) / W_max

        # Update SED matrix
        prox_like_f1 = partial(prox_likelihood_A, S=X2, Y=data, prox_g=prox_f1, W=W, Gamma=G)
        # TODO: Implement the more general version using `update_sdmm_variables`
        X1 = prox_like_f1(Z1-U1, step_f1)
        Z1 = prox_f1(X1+U1, step_f1)
        U1 = U1 + X1 - Z1

        # Update Intensity Matrix
        step_f2 = beta**it / lipschitz_const(X1) / W_max
        step_g2 = step_f2 * lM2
        prox_like_f2 = partial(prox_likelihood_S, A=X1, Y=data, prox_g=prox_f2, W=W, Gamma=G)
        X2_, Z2_, U2, CX = update_sdmm_variables(X2, Z2, U2, prox_like_f2, step_f2, prox_g2, step_g2,
                                                 constraints2)

        ## Convergence crit from Langville 2014, section 5 ?
        NMF_converge, norms = check_NMF_convergence(it, X2_, X2, e_rel, K, min_iter)
        all_norms.append(norms)

        # ADMM Convergence Criteria, adapted from Boyd 2011, Sec 3.3.1
        result = test_multiple_constraint_convergence(step_f2, step_g2, X2_, CX, Z2_, Z2,
                                                      U2, constraints2, e_rel)
        ADMM_converge, errors = result
        all_errors.append(errors)

        # Langville 2014 Section 5 uses the Frobenius norm, which is expensive to calculate
        # I include it here in case we need it for testing later, but it is turned off
        if False:
            if P is not None:
                model = np.zeros(data.shape)
                for b in range(data.shape[0]):
                    model[b] = P[b].dot(np.dot(X2.T, X1[b]))
            else:
                model = X1.dot(X2)
            frobenius = trData - 2*np.trace(model.T.dot(data))+np.trace(model.T.dot(model))
            frobeniusNorm.append(frobenius)

        # Check for both NMF and Primal and Dual variable convergence
        if NMF_converge and ADMM_converge:
            X2 = X2_
            break

        X2[:] = X2_[:]
        Z2[:] = Z2_[:]
    # Store the errors for convergence analysis
    all_errors = [all_norms, all_errors]

    # For testing purposes, allow us to probe the reason for non-convergence
    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("{0} iterations".format(it))
    return X1, X2, Tx, Ty, all_errors

def lipschitz_const(M):
    return np.real(np.linalg.eigvals(np.dot(M, M.T)).max())

def getSymmetryOp(shape):
    """Create a linear operator to symmetrize an image
    
    Given the ``shape`` of an image, create a linear operator that
    acts on the flattened image to return its symmetric version.
    """
    size = shape[0]*shape[1]
    idx = np.arange(shape[0]*shape[1])
    sidx = idx[::-1]
    symmetryOp = scipy.sparse.identity(size)
    symmetryOp -= scipy.sparse.coo_matrix((np.ones(size),(idx, sidx)), shape=(size,size))
    return symmetryOp

def getOffsets(width, coords=None):
    """Get the offset and slices for a sparse band diagonal array

    For an operator that interacts with its neighbors we want a band diagonal matrix,
    where each row describes the 8 pixels that are neighbors for the reference pixel
    (the diagonal). Regardless of the operator, these 8 bands are always the same,
    so we make a utility function that returns the offsets (passed to scipy.sparse.diags).

    See `diagonalizeArray` for more on the slices and format of the array used to create
    NxN operators that act on a data vector.
    """
    # Use the neighboring pixels by default
    if coords is None:
        coords = [(-1,-1), (-1,0), (-1, 1), (0,-1), (0,1), (1, -1), (1,0), (1,1)]
    offsets = [width*y+x for y,x in coords]
    slices = [slice(None, s) if s<0 else slice(s, None) for s in offsets]
    slicesInv = [slice(-s, None) if s<0 else slice(None, -s) for s in offsets]
    return offsets, slices, slicesInv

def diagonalizeArray(arr, shape=None, dtype=np.float64):
    """Convert an array to a matrix that compares each pixel to its neighbors

    Given an array with length N, create an 8xN array, where each row will be a
    diagonal in a diagonalized array. Each column in this matrix is a row in the larger
    NxN matrix used for an operator, except that this 2D array only contains the values
    used to create the bands in the band diagonal matrix.

    Because the off-diagonal bands have less than N elements, ``getOffsets`` is used to
    create a mask that will set the elements of the array that are outside of the matrix to zero.

    ``arr`` is the vector to diagonalize, for example the distance from each pixel to the peak,
    or the angle of the vector to the peak.

    ``shape`` is the shape of the original image.
    """
    if shape is None:
        height, width = arr.shape
        data = arr.flatten()
    elif len(arr.shape)==1:
        height, width = shape
        data = np.copy(arr)
    else:
        raise ValueError("Expected either a 2D array or a 1D array and a shape")
    size = width * height

    # We hard code 8 rows, since each row corresponds to a neighbor
    # of each pixel.
    diagonals = np.zeros((8, size), dtype=dtype)
    mask = np.ones((8, size), dtype=bool)
    offsets, slices, slicesInv = getOffsets(width)
    for n, s in enumerate(slices):
        diagonals[n][slicesInv[n]] = data[s]
        mask[n][slicesInv[n]] = 0

    # Create a mask to hide false neighbors for pixels on the edge
    # (for example, a pixel on the left edge should not be connected to the
    # pixel to its immediate left in the flattened vector, since that pixel
    # is actual the far right pixel on the row above it).
    mask[0][np.arange(1,height)*width] = 1
    mask[2][np.arange(height)*width-1] = 1
    mask[3][np.arange(1,height)*width] = 1
    mask[4][np.arange(1,height)*width-1] = 1
    mask[5][np.arange(height)*width] = 1
    mask[7][np.arange(1,height-1)*width-1] = 1

    return diagonals, mask

def diagonalsToSparse(diagonals, shape, dtype=np.float64):
    """Convert a diagonalized array into a sparse diagonal matrix

    ``diagonalizeArray`` creates an 8xN array representing the bands that describe the
    interactions of a pixel with its neighbors. This function takes that 8xN array and converts
    it into a sparse diagonal matrix.

    See `diagonalizeArray` for the details of the 8xN array.
    """
    height, width = shape
    offsets, slices, slicesInv = getOffsets(width)
    diags = [diag[slicesInv[n]] for n, diag in enumerate(diagonals)]

    # This block hides false neighbors for the edge pixels (see comments in diagonalizeArray code)
    # For now we assume that the mask in diagonalizeArray has already been applied, making these
    # lines redundant and unecessary, but if that changes in the future we can uncomment them
    #diags[0][np.arange(1,height-1)*width-1] = 0
    #diags[2][np.arange(height)*width] = 0
    #diags[3][np.arange(1,height)*width-1] = 0
    #diags[4][np.arange(1,height)*width-1] = 0
    #diags[5][np.arange(height)*width] = 0
    #diags[7][np.arange(1,height-1)*width-1] = 0

    diagonalArr = scipy.sparse.diags(diags, offsets, dtype=dtype)
    return diagonalArr

def getRadialMonotonicOp(shape, useNearest=True, minGradient=1):
    """Create an operator to constrain radial monotonicity

    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """
    # Center on the center pixel
    px = int(shape[1]/2)
    py = int(shape[0]/2)
    # Calculate the distance between each pixel and the peak
    size = shape[0]*shape[1]
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    X,Y = np.meshgrid(x,y)
    X = X - px
    Y = Y - py
    distance = np.sqrt(X**2+Y**2)

    # Find each pixels neighbors further from the peak and mark them as invalid
    # (to be removed later)
    distArr, mask = diagonalizeArray(distance, dtype=np.float64)
    relativeDist = (distance.flatten()[:,None]-distArr.T).T
    invalidPix = relativeDist<=0

    # Calculate the angle between each pixel and the x axis, relative to the peak position
    # (also avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually)
    inf = X==0
    tX = X.copy()
    tX[inf] = 1
    angles = np.arctan2(-Y,-tX)
    angles[inf&(Y!=0)] = 0.5*np.pi*np.sign(angles[inf&(Y!=0)])

    # Calcualte the angle between each pixel and it's neighbors
    xArr, m = diagonalizeArray(X)
    yArr, m = diagonalizeArray(Y)
    dx = (xArr.T-X.flatten()[:, None]).T
    dy = (yArr.T-Y.flatten()[:, None]).T
    # Avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually
    inf = dx==0
    dx[inf] = 1
    relativeAngles = np.arctan2(dy,dx)
    relativeAngles[inf&(dy!=0)] = 0.5*np.pi*np.sign(relativeAngles[inf&(dy!=0)])

    # Find the difference between each pixels angle with the peak
    # and the relative angles to its neighbors, and take the
    # cos to find its neighbors weight
    dAngles = (angles.flatten()[:, None]-relativeAngles.T).T
    cosWeight = np.cos(dAngles)
    # Mask edge pixels, array elements outside the operator (for offdiagonal bands with < N elements),
    # and neighbors further from the peak than the reference pixel
    cosWeight[invalidPix] = 0
    cosWeight[mask] = 0

    if useNearest:
        # Only use a single pixel most in line with peak
        cosNorm = np.zeros_like(cosWeight)
        columnIndices =  np.arange(cosWeight.shape[1])
        maxIndices = np.argmax(cosWeight, axis=0)
        indices = maxIndices*cosNorm.shape[1]+columnIndices
        indices = np.unravel_index(indices, cosNorm.shape)
        cosNorm[indices] = minGradient
        # Remove the reference for the peak pixel
        cosNorm[:,px+py*shape[1]] = 0
    else:
        # Normalize the cos weights for each pixel
        normalize = np.sum(cosWeight, axis=0)
        normalize[normalize==0] = 1
        cosNorm = (cosWeight.T/normalize[:,None]).T
        cosNorm[mask] = 0
    cosArr = diagonalsToSparse(cosNorm, shape)

    # The identity with the peak pixel removed represents the reference pixels
    diagonal = np.ones(size)
    diagonal[px+py*shape[1]] = -1
    monotonic = cosArr-scipy.sparse.diags(diagonal)

    return monotonic.tocoo()

def getPSFOp(psfImg, imgShape, threshold=1e-2):
    """Create an operator to convolve intensities with the PSF

    Given a psf image ``psfImg`` and the shape of the blended image ``imgShape``,
    make a banded matrix out of all the pixels in ``psfImg`` above ``threshold``
    that acts as the PSF operator.

    TODO: Optimize this algorithm to
    """
    height, width = imgShape
    size = width * height

    # Hide pixels in the psf below the threshold
    psf = np.copy(psfImg)
    psf[psf<threshold] = 0
    logger.info("Total psf pixels: {0}".format(np.sum(psf>0)))

    # Calculate the coordinates of the pixels in the psf image above the threshold
    indices = np.where(psf>0)
    indices = np.dstack(indices)[0]
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    coords = indices-np.array([cy,cx])

    # Create the PSF Operator
    offsets, slices, slicesInv = getOffsets(width, coords)
    psfDiags = [psf[y,x] for y,x in indices]
    psfOp = scipy.sparse.diags(psfDiags, offsets, shape=(size, size), dtype=np.float64)
    psfOp = psfOp.tolil()

    # Remove entries for pixels on the left or right edges
    cxRange = np.unique([cx for cy,cx in coords])
    for h in range(height):
        for y,x in coords:
            # Left edge
            if x<0 and width*(h+y)+x>=0 and h+y<=height:
                psfOp[width*h, width*(h+y)+x] = 0

                # Pixels closer to the left edge
                # than the radius of the psf
                for x_ in cxRange[cxRange<0]:
                    if (x<x_ and
                        width*h-x_>=0 and
                        width*(h+y)+x-x_>=0 and
                        h+y<=height
                    ):
                        psfOp[width*h-x_, width*(h+y)+x-x_] = 0

            # Right edge
            if x>0 and width*(h+1)-1>=0 and width*(h+y+1)+x-1>=0 and h+y<=height and width*(h+1+y)+x-1<size:
                psfOp[width*(h+1)-1, width*(h+y+1)+x-1] = 0

                for x_ in cxRange[cxRange>0]:
                    # Near right edge
                    if (x>x_ and
                        width*(h+1)-x_-1>=0 and
                        width*(h+y+1)+x-x_-1>=0 and
                        h+y<=height and
                        width*(h+1+y)+x-x_-1<size
                    ):
                        psfOp[width*(h+1)-x_-1, width*(h+y+1)+x-x_-1] = 0

    # Return the transpose, which correctly convolves the data with the PSF
    return psfOp.T.tocoo()

def getTranslationOp(deltaX, deltaY, shape, threshold=1e-8):
    """ Operator to translate an image by deltaX, deltaY pixels
    
    deltaX and deltaY can both be real numbers, which uses a linear interpolation to
    shift the peak by a fractional pixel amount.
    """
    Dx, Dy = int(deltaX), int(deltaY)
    dx = np.abs(deltaX-Dx)
    dy = np.abs(deltaY-Dy)

    height, width = shape
    size = width * height

    # If dx or dy are less than the shift threshold, set them to zero
    if dx < threshold:
        Dx = int(np.ceil(deltaX))
        dx = 0
    if dy < threshold:
        Dy = int(np.ceil(deltaY))
        dy = 0

    # Build the x and y translation matrices
    signX = int(np.sign(deltaX))
    if signX==0:
        signX = 1
    bx = scipy.sparse.diags([(1-dx), dx], [Dx, Dx+signX],
                            shape=(width, width), dtype=np.float64)
    signY = int(np.sign(deltaY))
    if signY==0:
        signY = 1
    tx = scipy.sparse.block_diag([bx]*height)
    ty = scipy.sparse.diags([(1-dy), dy], [Dy*width, (Dy+signY)*width], shape=(size, size), dtype=np.float64)
    # Create the single translation operator (used for A and S likelihoods)
    transOp = ty.dot(tx.T)

    return tx, ty, transOp

def nmf(Y, A0, S0, prox_A, prox_S, prox_S2=None, M2=None, lM2=None,
        max_iter=1000, W=None, P=None, e_rel=1e-3, algorithm='ADMM',
        outer_max_iter=50, min_iter=10):

    K = S0.shape[0]
    A = A0.copy()
    S = S0.copy()
    S_ = S0.copy() # needed for convergence test
    beta = 1. # 0.75    # TODO: unclear how to chose 0 < beta <= 1

    if W is not None:
        W_max = W.max()
    else:
        W = W_max = 1

    all_errors = []
    all_norms = []
    for it in range(outer_max_iter):
        # A: simple gradient method; need to rebind S each time
        prox_like_A = partial(prox_likelihood_A, S=S, Y=Y, prox_g=prox_A, W=W, P=P)
        step_A = beta**it / lipschitz_const(S) / W_max
        it_A = APGM(A, prox_like_A, step_A, max_iter=max_iter)

        # S: either gradient or ADMM, depending on additional constraints
        prox_like_S = partial(prox_likelihood_S, A=A, Y=Y, prox_g=prox_S, W=W, P=P)
        step_S = beta**it / lipschitz_const(A) / W_max
        if prox_S2 is None or algorithm == "APGM":
            it_S = APGM(S_, prox_like_S, step_S, max_iter=max_iter)
            errors = []
        elif algorithm == "ADMM":
            # steps set to upper limit per component
            step_S2 = step_S * lM2
            it_S, S_, _, _, errors = ADMM(S_, prox_like_S, step_S, prox_S2, step_S2, A=M2,
                                          max_iter=max_iter, e_rel=e_rel)
        elif algorithm == "SDMM":
            # TODO: Check that we are properly setting the step size.
            # Currently I am just using the same form as ADMM, with a slightly modified
            # lM2 in nmf_deblender
            step_S2 = step_S * lM2
            it_S, S_, _, _, errors = SDMM(S_, prox_like_S, step_S, prox_S2, step_S2,
                                          constraints=M2, max_iter=max_iter, e_rel=e_rel)
        else:
            raise Exception("Unexpected 'algorithm' to be 'APGM', 'ADMM', or 'SDMM'")

        logger.info("{0} {1} {2} {3} {4} {5}".format(it, step_A, it_A, step_S, it_S,
                                                     [(S[i,:] > 0).sum()for i in range(S.shape[0])]))

        if it_A == 0 and it_S == 0:
            break

        ## Convergence crit from Langville 2014, section 5 ?
        NMF_converge, norms = check_NMF_convergence(it, S_, S, e_rel, K, min_iter)
        all_errors += errors
        all_norms.append(norms)

        # Store norms and errors

        if NMF_converge:
            break

        S[:,:] = S_[:,:]
    S[:,:] = S_[:,:]
    return A,S, [all_norms, all_errors]

def init_A(B, K, peaks=None, I=None):
    # init A from SED of the peak pixels
    if peaks is None:
        A = np.random.rand(B,K)
    else:
        assert I is not None
        assert len(peaks) == K
        A = np.zeros((B,K))
        for k in range(K):
            px,py = peaks[k]
            A[:,k] = I[:,int(py),int(px)]
    A = prox_unity_plus(A, 0)
    return A

def init_S(N, M, K, peaks=None, data=None):
    cx, cy = int(M/2), int(N/2)
    S = np.zeros((K, N*M))
    if data is None or peaks is None:
        S[:,cy*M+cx] = 1
    else:
        tiny = 1e-10
        for pk, (px,py) in enumerate(peaks):
            S[pk, cy*M+cx] = np.abs(data[:,int(py),int(px)].mean()) + tiny
    return S

def adapt_PSF(P, B, shape, threshold=1e-2):
    # Simpler for likelihood gradients if P = const across B
    if isinstance(P, list) is False: # single matrix
        return getPSFOp(P, shape, threshold=threshold)

    P_ = []
    for b in range(B):
        P_.append(getPSFOp(P[b], shape, threshold=threshold))
    return P_

def get_constraint_op(constraint, shape, useNearest=True):
    """Get appropriate constraint operator
    """
    N,M = shape
    if constraint == " ":
        return scipy.sparse.identity(N*M)
    elif constraint == "M":
        return getRadialMonotonicOp((N,M), useNearest=useNearest)
    elif constraint == "S":
        return getSymmetryOp((N,M))
    raise ValueError("'constraint' should be in [' ', 'M', 'S'] but received '{0}'".format(constraint))

def nmf_deblender(I, K=1, max_iter=1000, peaks=None, constraints=None, W=None, P=None, sky=None,
                  l0_thresh=None, l1_thresh=None, gradient_thresh=0, e_rel=1e-3, psf_thresh=1e-2,
                  monotonicUseNearest=False, algorithm="GLMM", outer_max_iter=50):

    # vectorize image cubes
    B,N,M = I.shape
    if sky is None:
        Y = I.reshape(B,N*M)
    else:
        Y = (I-sky).reshape(B,N*M)
    if W is None:
        W_ = W
    else:
        W_ = W.reshape(B,N*M)
    if P is None:
        P_ = P
    else:
        P_ = adapt_PSF(P, B, (N,M), threshold=psf_thresh)
    
    logger.info("Shape: {0}".format((N,M)))

    # init matrices
    A = init_A(B, K, I=I, peaks=peaks)
    S = init_S(N, M, K, data=I, peaks=peaks)

    # define constraints for A and S via proximal operators
    # A: ||A_k||_2 = 1 with A_ik >= 0 for all columns k
    prox_A = prox_unity_plus

    # S: non-negativity or L0/L1 sparsity plus ...
    if l0_thresh is None and l1_thresh is None:
        prox_S = prox_plus
    else:
        # L0 has preference
        if l0_thresh is not None:
            if l1_thresh is not None:
                logger.warn("Warning: l1_thresh ignored in favor of l0_thresh")
            prox_S = partial(prox_hard, l=l0_thresh)
        else:
            prox_S = partial(prox_soft_plus, l=l1_thresh)

    # Load linear constraint operators
    if constraints is not None:
        linear_constraints = {
            " ": prox_id,    # do nothing
            "M": partial(prox_min, l=gradient_thresh), # positive gradients
            "S": prox_zero,   # zero deviation of mirrored pixels
        }
        # Proximal Operator for each constraint
        constraint_prox = [linear_constraints[c] for c in constraints]
        # Linear Operator for each constraint
        constraint_ops = [get_constraint_op(c, (N,M), useNearest=monotonicUseNearest) for c in constraints]
        # Weight of the linear operator (to test for convergence)
        constraint_norm = np.array([scipy.sparse.linalg.norm(C) for C in constraint_ops])
        #constraint_norm = np.sqrt(constraint_norm)
        logger.info("Norm2: {0}".format(constraint_norm))
    else:
        constraint_prox = None
        constraint_ops = None
        constraint_norm = None

    # run the NMF with those constraints
    if algorithm=="ADMM" or algorithm=="SDMM":
        raise NotImplemented("ADMM and SDMM have not yet been updated to the centered S matrices")
        A,S, errors = nmf(Y, A, S, prox_A, prox_S, prox_S2=prox_S2, M2=M2, lM2=lM2, max_iter=max_iter,
                  W=W_, P=P_, e_rel=e_rel, algorithm=algorithm, outer_max_iter=outer_max_iter)
        f = None
    elif algorithm=="GLMM":
        # TODO: Improve this, the following is for testing purposes only
        A, S, Tx, Ty, errors = GLMM(shape=(N, M), data=Y, X10=A, X20=S, peaks=peaks, W=W_, P=P_,
                                    prox_f1=prox_A, prox_f2=prox_S, prox_g1=None, prox_g2=constraint_prox,
                                    constraints1=None, constraints2=constraint_ops, lM1=1,
                                    lM2=constraint_norm, max_iter=max_iter, e_rel=e_rel, beta=1.0)

    # create the model and reshape to have shape B,N,M
    model = get_model(A, S, Tx, Ty, P_, (N,M))
    S = S.reshape(K,N,M)

    return A, S, model, P_, Tx, Ty, errors