import numpy as np
from itertools import combinations
    

def gen_glv_random(N, sigma=1.0):
    """
    Generate GLV model with:
        A_ij ~ - abs(N(0, sigma^2)),
        et r_i ~ N(0, sigma^2).

    """
    r = np.random.normal(0.0, sigma, size=N)
    A = - abs(np.random.normal(0.0, sigma, size=(N, N)))

    return A, r

def gen_glv_connectance_sigma_negative(N, c=0.2, sigma=0.2):
    """
    Generate GLV model:
        A = -I + B
    with
        B_ij = - X_ij * U_ij,
        X_ij ~ Bernoulli(c),
        U_ij ~ N(0, sigma^2),
    and r = (1, ..., 1).

    """

    X = np.random.binomial(1, c, size=(N, N))
    U = - abs(np.random.normal(0.0, sigma, size=(N, N)))
    B = X * U

    np.fill_diagonal(B, 0.0)

    A = -np.eye(N) + B
    r = np.ones(N)

    return A, r

######### Steady states #########
    
def glv_steady(A, r, I, tol=1e-6):
    """
    Solve equilibrium restricted to subset I:
      A_II x_I = - r_I
    Returns:
      x vector with zeros outside I, or [] if no feasible equilibrium.
    """
    N = len(r)
    I = list(I)
    k = len(I)

    if k == 0:
        # empty equilibrium: x = 0
        return np.zeros(N)

    A_II = A[np.ix_(I, I)]
    r_I = r[I]

    # check invertibility
    if np.linalg.matrix_rank(A_II) < k:
        return []

    x_I = np.linalg.solve(A_II, -r_I)

    # feasibility: strictly positive on I
    if np.any(x_I <= tol):
        return []

    x = np.zeros(N)
    x[I] = x_I
    return x

######### Saturated states #########

def glv_saturated(A, r, x, tol=1e-6):
    """
    Saturated: for all extinct species i with x_i=0 g_i(x) <= 0.
    Returns:
      True if saturated, False otherwise
    """
    g = r + A @ x
    extinct = (x <= tol)
    if np.all(g[extinct] <= tol):
        return True
    return False

######### Stable states #########

def glv_stab(A, r, x, sat, tol=1e-6):
    """
    Asymptotic stability using Jacobian of GLV:
      f_i(x) = x_i (r_i + (A x)_i)
      J_ij = δ_ij (r_i + (A x)_i) + x_i A_ij

    Asymptotically stable if all eigenvalues have negative real parts.
    """
    if sat == False :
        return False
    
    N = len(x)
    g = r + A @ x

    J = np.zeros((N, N))
    # diagonal term 
    np.fill_diagonal(J, g)

    # add x_i A_ij term
    J += (x[:, None] * A)

    eigvals = np.linalg.eigvals(J)
    max_re = np.max(np.real(eigvals))

    if max_re < -tol:
        return True  # asymptotically stable
    elif max_re > tol:
        return False   # unstable
    else:
        return False  # borderline / numerically neutral


######### Classification function #########
    
def classification_glv(A, r, tol=1e-6, include_empty=True):
    """
    Compute all admissible GLV equilibria.

    Output:
      - 'steady'     : equilibrium x*
      - 'size'       : richness k 
      - 'saturated'  : True if saturated, False otherwise
      - 'stable'     : True stable, False otherwise
    """
    N = len(r)
    classifications = []

    # include empty set I=∅
    if include_empty:
        I = []
        x = np.zeros(N)
        sat = glv_saturated(A, r, x, tol=tol)
        stab = glv_stab(A, r, x, sat, tol=tol)
        classifications.append({
            "steady": x.tolist(),
            "size": 0,
            "saturated": sat,
            "stable": stab,
        })

    # non-empty subsets
    for k in range(1, N + 1):
        for I in combinations(range(N), k):
            x = glv_steady(A, r, I, tol=tol)
            if len(x) > 0 :
                sat = glv_saturated(A, r, x, tol=tol)
                stab = glv_stab(A, r, x, sat, tol=tol)

                classifications.append({
                    "steady": x.tolist(),
                    "size": k,
                    "saturated": sat,
                    "stable": stab,
                })

    return classifications

def classification_glv_params(params, tol=1e-6, include_empty=True):
    A, r = params
    return classification_glv(A, r, tol=tol, include_empty=include_empty)



