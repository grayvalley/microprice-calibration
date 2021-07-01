import numpy as np
import numba as nb

class ShortTermAlpha:

    def __init__(self, zeta, epsilon, eta, beta):
        self.zeta = zeta # signal mean reversion speed
        self.eta = eta # signal volatility
        self.epsilon = epsilon # step size used in finite difference derivative
        self.dalpha = 0.01 # step size for alpha grid
        self.imbalance =  np.arange(-1, 1.0 + 0.1, 0.1)
        self.alpha = self.imbalance * beta
        self.alpha_up = self.alpha + self.epsilon
        self.alpha_down = self.alpha - self.epsilon

def counter(current_idx, total_idx):
    M = round(total_idx / 10)
    percent = int(round(current_idx / total_idx * 100, 0))
    if np.mod(current_idx, M) == 0:
        print("Processing: " + str(percent) + " percent completed.")


def nan_to_num(x, nan):
    """ Change the NaNs in a numpy array to the desired values.
    :param x: a numpy array
    :param nan: desired value
    :return: a deep copy of x with changed values
    """
    y = np.copy(x)
    y[np.isnan(y)] = nan
    return y

@nb.njit
def linear_extrapolation(x, y, e):
    """ Extrapolation and interpolation

    :param x: a numpy array
    :param y: a numpy array
    :param e: a numpy array, equivalent of x
    :return: a numpy array
    """
    new_x = np.sort(x)
    new_y = y[np.argsort(x)]

    def point_wise(ep):
        if ep < new_x[0]:
            return new_y[0] + (ep - new_x[0]) * (new_y[1] - new_y[0]) / (new_x[1] - new_x[0])
        elif ep > new_x[-1]:
            return new_y[-1] + (ep - new_x[-1]) * (new_y[-1] - new_y[-2]) / (new_x[-1] - new_x[-2])
        else:
            return np.interp([ep], x, y)[0]

    return np.array([point_wise(i) for i in e])

@nb.njit
def compute_h_delta_p(kappa, delta, alpha, h_1, h_2):
    h_delta = np.zeros(100)
    for depth in range(0, 100):
        h_delta[depth] = np.exp(-kappa*depth) * (delta + delta*depth - alpha + h_1 - h_2)
    return h_delta

@nb.njit
def compute_h_delta_m(kappa, delta, alpha, h_1, h_2):
    h_delta = np.zeros(100)
    for depth in range(0, 100):
        h_delta[depth] = np.exp(-kappa*depth) * (delta + delta*depth + alpha + h_1 - h_2)
    return h_delta

class ShortTermAlpha_Finite_Difference_Solver:
    
    @staticmethod
    def solve_tob_postings(
            short_term_alpha, q, t, Delta, term_penalty,
            phi, dt, lambda_p, lambda_m):
   
    
        # set up the time, inventory, and space grids
        dalpha     = short_term_alpha.dalpha
        alpha      = short_term_alpha.alpha
        eta        = short_term_alpha.eta
        zeta       = short_term_alpha.zeta
        Nalpha     = alpha.shape[0]
        
        # alpha values used to compute numerical derivatives for alpha
        # terms
        alpha_up = short_term_alpha.alpha_up
        alpha_down = short_term_alpha.alpha_down
    
        Nt = t.shape[0]
        Nq = q.shape[0]
    
        # stores the h as a function of q, alpha, t
        h = np.full((Nq, Nalpha, Nt), fill_value=np.nan)
    
        # stores the optimal posting strategies as a function of q, alpha, t
        lp = np.zeros((Nq, Nalpha, Nt), dtype=bool)
        lm = np.zeros((Nq, Nalpha, Nt), dtype=bool)
    
        # Terminal conditions for all q
        h[:, :, h.shape[-1] - 1] = np.around(
            -term_penalty * q * q, decimals=4)[:, np.newaxis]
    
        d_alpha_h = np.zeros(Nalpha)
    
        # Index of alpha smaller than 0
        idx_m = np.where(alpha < 0)[0]
        # Index of alpha greater than 0
        idx_p = np.where(alpha > 0)[0]
        # Index of alpha equals to 0
        idx_0 = np.where(alpha == 0)[0]
    
        # time dimension loop
        for i in range(h.shape[2] - 2, -1, -1):
    
            counter(Nt - i, Nt)
    
            # inventory dimension loop (computes for each alpha level)
            for k in range(0, h.shape[0], 1):
    
                # compute h(t, alpha + epsilon, q)
                h_p2 = linear_extrapolation(alpha, h[k, :, i + 1], alpha_up)
                
                if q[k] > -(Nq - 1) / 2:
                    
                    # compute h(t, alpha + epsilon, q - 1)
                    h_p1 = linear_extrapolation(alpha, h[k - 1, :, i + 1], alpha_up)
                    
                    # value function change if we post sell order
                    post_h_p = 0.5 * Delta + h_p1 - h_p2
                    
                else:
                    post_h_p = np.zeros(Nalpha)
    
                # compute h(t, alpha - epsilon, q)
                h_m2 = linear_extrapolation(alpha, h[k, :, i + 1], alpha_down)
                
                if q[k] < (Nq - 1) / 2:
                    
                    # compute h(t, alpha - epsilon, q + l)
                    h_m1 = linear_extrapolation(alpha, h[k + 1, :, i + 1], alpha_down)
                    
                    # value function change if we post buy order
                    post_h_m = 0.5 * Delta + h_m1 - h_m2
                    
                else:
                    post_h_m = np.zeros(Nalpha)
    
                lp[k, :, i + 1] = post_h_p > 0
                lm[k, :, i + 1] = post_h_m > 0
    
                # solve DPE for h function one time-step backwards
                d_alpha_h[idx_m] = (h[k, idx_m + 1, i + 1] - h[k, idx_m, i + 1]) / dalpha
                d_alpha_h[idx_p] = (h[k, idx_p, i + 1] - h[k, idx_p - 1, i + 1]) / dalpha
                d_alpha_h[idx_0] = (h[k, idx_0 + 1, i + 1] - h[k, idx_0 - 1, i + 1]) / (2 * dalpha)
                
                # central difference second derivative of h w.r.t alpha inside finite difference
                # grid (derivative boundary conditions are imposed below)
                d2_alpha_h = (
                    h[k, 2:h.shape[1], i + 1] - 2 * h[k, 1:(h.shape[1] - 1), i + 1] + h[k, 0:(h.shape[1] - 2), i + 1]) / (dalpha ** 2)
    
                h[k, 1:(h.shape[1] - 1), i] = h[k, 1:(h.shape[1] - 1), i + 1] \
                                              + dt * (
                                                  
                                                  # drift in imbalance signal
                                                  - zeta * alpha[1:(alpha.shape[0] - 1)] * d_alpha_h[1:(d_alpha_h.shape[0] - 1)]
                                                  
                                                  # alpha "gamma" term
                                                  + 0.5 * (eta ** 2) * d2_alpha_h
                                                  
                                                  + zeta * alpha[1:(alpha.shape[0] - 1)] * q[k]
                                                  
                                                  # inventory penalty
                                                  - phi * (q[k] ** 2)
                                                      
                                                      # impact of optimal decisions to value function
                                                      + lambda_p * np.maximum(post_h_p[1:(post_h_p.shape[0] - 1)], 0)
                                                      + lambda_m * np.maximum(post_h_m[1:(post_h_m.shape[0] - 1)], 0)
                                                      
                                                      # impact of arriving orders to signal and value function 
                                                      + lambda_p * (h_p2[1:(h_p2.shape[0] - 1)] - h[k, 1:(h.shape[1] - 1), i + 1])
                                                      + lambda_m * (h_m2[1:(h_m2.shape[0] - 1)] - h[k, 1:(h.shape[1] - 1), i + 1]))
    
                # impose second derivative vanishes along maximum and minimum values of alpha grid
                h[k, h.shape[1] - 1, i] = 2 * h[k, h.shape[1] - 2, i] - h[k, h.shape[1] - 3, i]
                h[k, 0, i] = 2 * h[k, 1, i] - h[k, 2, i]
    
        return h, lp, lm

