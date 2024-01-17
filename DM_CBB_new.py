import sys
from scipy.stats import t
import collections
import pandas as pd
import numpy as np
from arch.bootstrap import CircularBlockBootstrap
from arch.bootstrap import optimal_block_length
import math
from arch.bootstrap import IIDBootstrap
import jax
import statsmodels.api as sm
import statistics as stat

def QS_Kernel(x):
    a = 6*x/5
    res = 3*(math.sin(math.pi*a)/(math.pi*a) - math.cos(math.pi*a))/(math.pi*a)**2
    return res

def autocovariance(Xi, N, k, Xs):
    autoCov = 0
    T = float(N)
    for i in np.arange(0, N-k):
        autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
    return (1/(T))*autoCov

def HAC_Variance(d, kernel = 'QS'):
    if len(d.shape)==2:
        T = d.shape[0]
        N = d.shape[1]
    elif len(d.shape)==1:
        T = len(d)

    if kernel == "QS":
        weight = []
        bw = math.floor(1.3*T**(1/5))
        for i in np.arange(1, T):
            ww = QS_Kernel(i/(bw))
            weight.append(ww)   
    elif kernel == "False":
        weight = np.repeat(1,T)

    if len(d.shape)==2:
        a=3
    elif len(d.shape)==1:
        mean_d = pd.Series(d).mean()
        gamma = []
        for lag in range(0,T):
            gamma.append(autocovariance(d,len(d),lag,mean_d)) # 0, 1, 2
        #V_d = (d_cov[0] + 2*sum(d_cov[1:]))/T

        #acf, _ = sm.tsa.acf(d, alpha=0.05, nlags = T-1, missing = 'drop')
        #dd_var = stat.variance(np.array(d))
        #d_cov = acf*dd_var
        
        if kernel == "False":
            d_var = gamma[0]
        else :
            d_var = (gamma[0] + 2*sum(gamma[1:]*np.array(weight)))
            #d_var = (((([0.5]+weight) * d_cov)*2).sum())/T
    return d_var


    
def dm_test(d_lst, h = 1, kernel = 'QS'):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_d = len(d_lst)

        # Check range of h
        if (h >= len_d):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        
    
    
    # Length of lists (as real numbers)
    T = float(len(d_lst))   
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    if kernel == 'False':
        gamma = []
        for lag in range(0,h):
            gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
        V_d = (gamma[0] + 2*sum(gamma[1:]))
        DM_stat1=(V_d/T)**(-0.5)*mean_d
        harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
        DM_stat = harvey_adj*DM_stat1
    elif kernel == 'QS':
        V_d = HAC_Variance(d_lst)
        DM_stat=(V_d/T)**(-0.5)*mean_d
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value mean_d V_d')
    rt = dm_return(DM=DM_stat, p_value=p_value, mean_d=mean_d, V_d=V_d)
    return rt

    
def CBB_stat_func(d):
    T = len(d)
    zeta_star = HAC_Variance(d)
    #stat = (d.mean())/((zeta_star/T)**(0.5))
    stat = (zeta_star/T)**(-0.5)*(d.mean())
    index = ["CBB_stat"]
    #return pd.Series(stat, index=index)
    return stat



def stochastic_function(seed, high = 1000000):
    rng = np.random.default_rng(seed)
    return rng.integers(high, size=1)



def get_rng_key(sample_size, theta, condition):
    '''
    Get the RNG key to use for one simulation replicate
    '''
    # seeds from random.org
    # organized by sample size and value of theta
    seeds = {
        'small': {
            '0.0': {
                'sigma_l': 278703,
                'sigma_h': 457475,
                'both': 531184
            },
            '1.0': {
                'sigma_l': 307503,
                'sigma_h': 596255,
                'both': 453558
            },
            '10.0': {
                'sigma_l': 252784,
                'sigma_h': 612776,
                'both': 905647
            },
        },
        'large': {
            '0.0': {
                'sigma_l': 568856,
                'sigma_h': 629931,
                'both': 685086
            },
            '1.0': {
                'sigma_l': 630072,
                'sigma_h': 680833,
                'both': 173016
            },
            '10.0': {
                'sigma_l': 551803,
                'sigma_h': 993769,
                'both': 637892
            }
        }  
    }

    # split into 1000 RNG keys, one per replicate at this combination of
    # sample size and theta
    rng = np.random.default_rng(seeds[sample_size][theta][condition])
    keys = rng.integers(high = 1000000, size=1000)
    return keys




def CBB_test(x, seed):
    sweep_d = x - x.mean()

    opt = optimal_block_length(x)
    opt_block = int(np.ceil(opt["circular"]))
    #CBB_stat_obs = CBB_stat_fun(np.array(x), opt_block)
    #if opt_block == 1:
    #    bs = IIDBootstrap(np.array(sweep_d), seed=seed)
    #    results = bs.apply(IIDBoots_stat, 2500)
    bs = CircularBlockBootstrap(opt_block, np.array(sweep_d), seed=seed)
    results = bs.apply(CBB_stat_func, 2500)
    CBB_stat = pd.DataFrame(results, columns=["CBB_stat"])

    #DM_stat, DM_p = dm_test(x, h=1, kernel = "False")
    DM_QS_stat, DM_QS_p, mean_d, V_d = dm_test(x, h=1, kernel = "QS")
    #CBB_p = (abs(np.array(CBB_stat)) > abs(DM_stat)).mean()
    #CBB_obs_p = (abs(np.array(CBB_stat)) > abs(CBB_stat_obs)).mean()
    CBB_QS_p = (abs(np.array(CBB_stat)) > abs(DM_QS_stat)).mean()
    #CBB_obs_p = (abs(np.array(CBB_stat)) > abs(obs_stat)).mean()
    #return CBB_stat, CBB_p, DM_stat, DM_p, CBB_stat_obs, CBB_obs_p, DM_QS_stat, CBB_QS_p, opt_block
    return CBB_stat, DM_QS_stat, DM_QS_p, CBB_QS_p, opt_block, mean_d, V_d



