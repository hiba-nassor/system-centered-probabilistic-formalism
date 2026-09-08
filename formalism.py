import numpy as np 
from collections import defaultdict, Counter

def analyze_one_system(N, params, classification, types):
    """
    One call to classification(params), then extracts for each type:
    - F_p vector size N+2: [∅, 0, 1, ..., N]
    - S_p = number of states of that type

    Returns:

    out[type_name] = {
        "F_p": np.array(N+2),
        "S_p": int,
        "S_p_k": Counter over k (k=0..N)
        }
    """

    # Call classification function -> returns dictionary of all admissible states with their classification
    states = classification(params) # Called once

    S_p_k_type = {name: Counter() for name in types} # number of states with exactly k species for each state type
    S_p_type = {name: 0 for name in types} # Mutiplicity of states of each type (their number)

    for state in states:
        k = int(state["size"]) # richness of state (biodiversity)
        for name, rule in types.items():
            test = True
            if rule is not None: 
                field, val = rule
                test = (state[field] == val) # Check if a state is saturated, stable, ESS 

            if test:
                S_p_k_type[name][k] += 1 
                S_p_type[name] += 1
    # For a given set of parameters, we have now the information about the richness of states and the multiplicity for each state type.
    # We use these variables to build the probability vector F_p
    
    out = {} 
    for name in types:
        S_p = S_p_type[name] # Multiplicity
        F_p = np.zeros(N + 2) # Probability vector 

        if S_p == 0:
            F_p[0] = 1.0  # F^∅
        else:
            for k, c in S_p_k_type[name].items():
                F_p[k + 1] = c / S_p

        out[name] = {
            "F_p": F_p,
            "S_p": S_p,
            "S_p_k": S_p_k_type[name]
        }

    return out


def monte_carlo_full(M, N, generate_params, classification, types):

    accum_F = {name: np.zeros(N + 2) for name in types}
    S = {name: Counter() for name in types}
    histo = {name: defaultdict(Counter) for name in types}

    for _ in range(M):
        params = generate_params()
        out = analyze_one_system(N, params, classification, types)

        for name in types:
            F_p = out[name]["F_p"]
            S_p = out[name]["S_p"]
            S_p_k = out[name]["S_p_k"]

            accum_F[name] += F_p
            S[name][S_p] += 1

            for k, c in S_p_k.items():
                histo[name][S_p][k] += c

    P_type = {name: accum_F[name] / M for name in types}

    return P_type, S, histo