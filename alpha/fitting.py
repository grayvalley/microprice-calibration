import json
import pandas as pd
import numpy as np

from solver import (
    ShortTermAlpha_Finite_Difference_Solver,
    ShortTermAlpha
)


def load_json(filename):

    with open(filename, "r") as f:
        data = json.load(f)

    return data


def save_excel(decisions, inventory_levels, signal_levels):
 
    opt_dec = pd.DataFrame(data=decisions, index=inventory_levels, columns=signal_levels)
    opt_dec.to_excel("solution.xlsx")

#%%
def main():
#%%
    params = load_json("appsettings.json")

    T            = 60
    lambda_p     = 5
    lambda_m     = 5
   
    dt = T / 1000

    short_term_alpha = ShortTermAlpha(0.5, 0.001, 0.001, 0.2)

    # Inventory Vector
    q = np.arange(-5, 5 + 1, 1)

    # Time Vector
    t = np.arange(0, T, dt)

    # The DPE is solved in the helper file
    h, lp, lm = ShortTermAlpha_Finite_Difference_Solver.solve_tob_postings(
        short_term_alpha, q, t, 0.5*0.05, 0.00001, 0.001, dt,
        lambda_p, lambda_m)
    
    p = lp[:, :, 100]
    m = lm[:, :, 100]
    
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.imshow(p, interpolation='nearest', cmap=cm.inferno, aspect='auto')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.imshow(m, interpolation='nearest', cmap=cm.inferno, aspect='auto')
    plt.show()
    
#%%
    # Create optimal posting matrix
    l_p = np.zeros((lp.shape[0], lp.shape[1]))
    l_m = np.zeros((lp.shape[0], lp.shape[1]))
    l_p[lp[:, :, 100] == True] = 1
    l_p[lp[:, :, 100] == False] = 0
    l_m[lm[:, :, 100] == True] = 2
    l_m[lm[:, :, 100] == False] = 0
    decisions = l_m + l_p

    save_excel(decisions, q, short_term_alpha.value)

    print("")

if __name__ == '__main__':
    main()
