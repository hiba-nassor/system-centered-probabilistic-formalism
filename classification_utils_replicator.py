import numpy as np 
from itertools import combinations

def gen_replicator(N, mu=None, mean=0, var=1):
    """
    Generate fitness matrix for replicator model.
    """
    Lambda = np.random.normal(mean, np.sqrt(var), (N, N))
    if mu is None:
        np.fill_diagonal(Lambda, 0)
    
    if mu is not None:
        FIT = fitness(Lambda, mu)

        Lambda = FIT

    return  Lambda
            
######### Steady states #########

def replicator_steady(Lambda):
    """
    Solve equilibrium.
    Returns:
      z, or [] if no feasible equilibrium.
    """
    N = len(Lambda)
    
    C = np.concatenate((Lambda[0:N-1, :] - Lambda[1:N, :], np.ones((1, N))), axis=0)
    
    det = np.linalg.det(C)
    # check invertibility
    if det == 0:
        return []

    b = np.zeros((N, 1))
    b[N-1] = 1
    z = np.linalg.solve(C, b)

    # feasibility: positive solutions
    if len(z[z < 0]) >= 1:
        return []

    else:
        return z
    
def fitness(A,mu=1):
    """
    Compute the fitness of the reducted model.
    A is the matrix of interaction and mu is the ratio I/II.
    """
    n=len(A)
    return mu*(A.T-A)+A.T-np.matrix(np.ones((n,1)))*np.matrix(np.diagonal(A))

######### Saturated states #########

def replicator_saturated(Lambda, z, tol=1e-6):
    """
    Checks whether z is a saturated state.
    Returns:
    True if z is a saturated state.
    False otherwise.
    """
    N = len(Lambda)
    z = np.reshape(z, (N, 1))
    phi = (z.T @ Lambda @ z)[0, 0]
    support = np.nonzero(z.flatten() > tol)[0]

    for i in range(N):
        if i not in support:
            ei = np.zeros((N, 1))
            ei[i] = 1
            if (ei.T @ Lambda @ z)[0, 0] > phi + tol:
                return False
    return True

######### Stable states #########

def Jac(z, Lambda):
    """
    Compute the Jacobian. 
    """
    assert len(Lambda) != 0 
    n = len(Lambda)
    z = np.reshape(z,(n,1))
    JAC_coex = np.diag(z.T[0])@(Lambda - np.ones((n,n))@np.diag(np.array((Lambda + Lambda.T)@z).T[0]))
    bloc = np.array((Lambda)@z -((z.T)@Lambda@z)).T[0]
    JAC_trivial = np.diag(bloc)
    return JAC_coex + JAC_trivial

def Jac_red(z, Lambda):
    """
    Jacobian Matrix for a given system at a given point z reducted to the simplexe sum(z_i)=1.
    """
    assert len(Lambda) != 0 
    n = len(Lambda)
    P = np.concatenate((np.ones((1,n)),np.concatenate((np.ones((n-1,1)),-np.eye(n-1)),axis=1)),axis=0)
    JAC = Jac(z,Lambda)
    JACnew = np.linalg.inv(P)@JAC@P
    return JACnew[1:,1:]

def replicator_stab(Lambda, z, sat_val):
    """
    Compute the stability of steady state z.
    sat_val = True if z is saturated and False otherwise. If z is not saturated, then it is not stable.
    """
    if sat_val == False:
        return False
    M = Jac_red(z, Lambda)
    VP = np.linalg.eig(M)[0]
    U = True
    for v in VP:
        if v > 0: U =  False
        if (v == 0 & U == True): U = False
    return U


######### Evolutionary Stable States #########

def replicator_ESS(Lambda, z, stab_val, tol=1e-6):
    """
    Checks if z is an ESS by testing negative definiteness on V.
    Returns:
    True if z is an ESS equilibrium.
    False otherwise.

    stab_value = True if z is asym stable and False otherwise. If z is not stable, then it is not an ESS.    
    """
    if stab_val == False :
        return False
     
    A = 0.5 * (Lambda + Lambda.T)
    S = [i for i in range(len(z)) if z[i] > tol]
    if not S:
        return False
    A_S = A[np.ix_(S, S)]

    # Construct basis for V = {d / sum d_i = 0}
    m = len(S)
    if m == 1:
        return True  # Single strategy is always ESS if saturated
    B = np.zeros((m, m-1))
    for i in range(m-1):
        B[i, i] = 1
        B[m-1, i] = -1
    
    # Project A_S onto V
    A_V = B.T @ A_S @ B
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A_V).real
    if all(e < -tol for e in eigenvalues):
        return True
    return False


######### Classification Function #########

def classification_replicator(Lambda, tol=1e-6):
    """
        Classifies the states ( from steady to ESS ) for Lambda with i.i.d entries.
        A given dictionnary has the following entries :
        -- 'steady' : the value of the steady state z
        -- 'type' : the number of non zero coefficients in z. This gives the number of species which are living at z.
        -- 'saturated' : True if saturated, False otherwise.
        -- 'stable' : True if asymptoticaly stable, False otherwise.
        -- 'ESS' : True if ESS, False otherwise. 
        """


    N = len(Lambda)
    classifications = []
    for size in range(1, N + 1):
        for combination in combinations(range(N), size):
            Lambdasub = np.zeros((size, size))
            for i, row in enumerate(combination):
                for j, col in enumerate(combination):
                    Lambdasub[i, j] = Lambda[row, col]

            zsub = replicator_steady(Lambdasub)
            if len(zsub) > 0:
                z = np.zeros((N, 1))
                z[list(combination)] = zsub
                zs = z.reshape(N).tolist()
                nash_val = replicator_saturated(Lambda, z, tol)
                stab_val = replicator_stab(Lambda, z, nash_val)
                ESS_val = replicator_ESS(Lambda, z, stab_val, tol)
                classification = {
                    'steady': zs,
                    'size': size,
                    'saturated': nash_val,
                    'stable': stab_val,
                    'ESS': ESS_val
                }
                classifications.append(classification)

    return classifications

def classification_replicator_params(params, tol=1e-6):
    Lambda = params
    return classification_replicator(Lambda, tol)