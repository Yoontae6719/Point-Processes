import numpy as np
import pandas as pd
from scipy.stats import lognorm,gamma
from scipy.optimize import brentq

# Synthetic data
np.random.seed(0)

def gen_stationary_poi():
    """
    the condition intensity function is given as λ(t | H_t) = 1
    """
    tau = np.random.exponential(size = 100000)
    cumsum_tau = tau.cumsum()
    score = 1
    return [cumsum_tau, score]
def gen_non_stationary_poi():
    """
    The condition intensity function is given as λ(t | H_t) = 0.99 * sin(2πt / 20000) + 1
    """
    L = 20000
    amp = 0.99
    l_t   = lambda t : amp * np.sin(2 * np.pi * t / L) + 1
    l_int = lambda t1,t2: - L/(2*np.pi)*( np.cos(2*np.pi*t2/L) - np.cos( 2*np.pi*t1/L) )*amp   + (t2-t1)

    while 1:
        cumsum_tau = np.random.exponential(size=210000).cumsum() * 0.5
        r = np.random.rand(210000)
        index = r < l_t(cumsum_tau) / 2.0

        if index.sum() > 100000:
            cumsum_tau = cumsum_tau[index][:100000]
            score = - (np.log(l_t(cumsum_tau[80000:])).sum() - l_int(cumsum_tau[80000 - 1], cumsum_tau[-1])) / 20000
            break

    return [cumsum_tau, score]

def gen_stationary_renewal():
    """
    Stationary renewal process : the inter-event intervals { τ_i = t_{i+1} - t_{i} }
    We herein use the log-normal distribution with a mean of 1.0 and std of 6.0 for p(τ).
    """
    s = np.sqrt(np.log(6 * 6 + 1))
    mu = -s * s / 2
    tau = lognorm.rvs(s=s, scale=np.exp(mu), size=100000)
    lpdf = lognorm.logpdf(tau, s=s, scale=np.exp(mu))
    cumsum_tau = tau.cumsum()
    score = - np.mean(lpdf[80000:])

    return [cumsum_tau, score]
def gen_nonstationary_renewal():
    """
    We first generate a sequence {t'} from stationary renewal process, and then we rescale the time according to t for a non-negative trend function r(t).
    in this process, an inter-event interval tends to be followed by the one with similar length, but the expected length gradually varies in times.
    """
    L = 20000
    amp = 0.99
    l_t = lambda t: np.sin(2 * np.pi * t / L) * amp + 1
    l_int = lambda t1, t2: - L / (2 * np.pi) * (np.cos(2 * np.pi * t2 / L) - np.cos(2 * np.pi * t1 / L)) * amp + (
                t2 - t1)

    cumsum_tau = []
    lpdf = []
    x = 0

    k = 4
    rs = gamma.rvs(k, size=100000)
    lpdfs = gamma.logpdf(rs, k)
    rs = rs / k
    lpdfs = lpdfs + np.log(k)

    for i in range(100000):
        x_next = brentq(lambda t: l_int(x, t) - rs[i], x, x + 1000)
        l = l_t(x_next)
        cumsum_tau.append(x_next)
        lpdf.append(lpdfs[i] + np.log(l))
        x = x_next

    cumsum_tau = np.array(cumsum_tau)
    lpdf = np.array(lpdf)
    score = - lpdf[80000:].mean()

    return [cumsum_tau, score]
def gen_self_correcting():
    """
    the conditional intensity function is given as λ(t | H_t) = exp(t - sum_{t_i < t} 1)
    """
    def self_correcting_process(mu, alpha, n):
        t = 0
        x = 0
        cumsum_tau = []
        log_l = []
        Int_l = []

        for i in range(n):
            e = np.random.exponential()
            tau = np.log(e * mu / np.exp(x) + 1) / mu  # e = ( np.exp(mu*tau)- 1 )*np.exp(x) /mu
            t = t + tau
            cumsum_tau.append(t)
            x = x + mu * tau
            log_l.append(x)
            Int_l.append(e)
            x = x - alpha

        return [np.array(cumsum_tau), np.array(log_l), np.array(Int_l)]

    [cumsum_tau, log_l, Int_l] = self_correcting_process(1, 1, 100000)
    score = - (log_l[80000:] - Int_l[80000:]).sum() / 20000

    return [cumsum_tau, score]
def gen_hawkes1():
    [cumsum_tau, LL] = simulate_hawkes(100000, 0.2, [0.8, 0.0], [1.0, 20.0])
    score = - LL[80000:].mean()
    return [cumsum_tau, score]
def gen_hawkes2():
    [cumsum_tau, LL] = simulate_hawkes(100000, 0.2, [0.4, 0.4], [1.0, 20.0])
    score = - LL[80000:].mean()
    return [cumsum_tau, score]
def simulate_hawkes(n, mu, alpha, beta):
    cumsum_tau = []
    LL = []

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0

    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential() / l
        x = x + step

        l_trg_Int1 += l_trg1 * (1 - np.exp(-beta[0] * step)) / beta[0]
        l_trg_Int2 += l_trg2 * (1 - np.exp(-beta[1] * step)) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0] * step)
        l_trg2 *= np.exp(-beta[1] * step)
        l_next = mu + l_trg1 + l_trg2

        if np.random.rand() < l_next / l:  # accept
            cumsum_tau.append(x)
            LL.append(np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int)
            l_trg1 += alpha[0] * beta[0]
            l_trg2 += alpha[1] * beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1

            if count == n:
                break

    return [np.array(cumsum_tau), np.array(LL)]


if __name__ == '__main__':
    print(gen_hawkes1())