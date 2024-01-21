import os
import time

import jax.numpy as jnp
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist


# generate an cholesky root of an AR(1) correlation matrix 
# The AR(1) correlation matrix has an explicit formula for the cholesky root 
# in terms of rho. 
# It is a special case of a general formula developed by V.Madar (2016)
def AR1Root(rho,p):
    R = jnp.zeros((p, p))
    R = R.at[0,].set(pow(rho, jnp.arange(p)))
    #R[0,] = pow(rho, jnp.arange(p))
    c = jnp.sqrt(1 - rho**2)
    R2 = c * R[0,:]
    
    for i in range(1,p):
        R = R.at[i, jnp.arange(i,p)].set(R2[jnp.arange(0,p-i)])
        #R[i, jnp.arange(i,p)] = R2[jnp.arange(0,p-i)]
    return jnp.transpose(R)
      


class ARCHDFA():
    '''
    Class representing a dynamic factor analysis model where the latent factors
    follow an autoregressive, conditional heteroskedastic [not yet implemented] model
    '''
    def __init__(self, num_timesteps=None, num_series=None, num_horizons = None,
                 num_factors=1, p=1, q=1, intercept_by_series=False, 
                 ar_constraint='[0,1]', sigma_factors_model='AR', 
                 loadings_constraint='positive'):
        '''
        Initialize an ARCHDFA model
        
        Parameters
        ----------
        num_timesteps: integer or None
            Number of time steps. If None, will be set at the time of fitting
        num_series: integer
            Number of observed series, e.g. number of locations in panel data
        num_horizons: integer
            Number of observed horizons; defaults to 1
        num_factors: integer
            Number of latent factors; defaults to 1
        p: integer
            Order of autoregressive processes for latent factors; defaults to 1
        q: integer
            Order of autoregressive processes for error of latent factors; defaults to 1
        intercept_by_series: boolean
            If True, estimate a separate intercept for each series. Otherwise,
            estimate a single intercept that is shared across all series.
            Defaults to False.
        ar_constraint: string
            Constraints on autoregressive coefficients. Either '[-1,1]' or '[0,1]'
        sigma_factors_model: string
            Constrains on sigma_factors, Either 'AR' or 'constant'
        loadings_constraint: string
            Constraints on factor loadings. Either 'simplex' or 'positive'.
            Defaults to 'positive'.

        Returns
        -------
        None
        '''
        if num_timesteps is not None and \
                (type(num_timesteps) is not int or num_timesteps <= 0):
            raise ValueError('num_timesteps must be None or a positive integer')
        self.num_timesteps = num_timesteps
        
        if type(num_series) is not int or num_series <= 0:
            raise ValueError('num_series must be a positive integer')
        self.num_series = num_series

        if type(num_horizons) is not int or num_horizons <= 0:
            raise ValueError('num_horizons must be a positive integer')
        self.num_horizons = num_horizons
        
        if type(num_factors) is not int or num_factors <= 0:
            raise ValueError('num_factors must be a positive integer')
        self.num_factors = num_factors
        
        if type(p) is not int or p <= 0:
            raise ValueError('p must be a positive integer')
        self.p = p

        if type(q) is not int or q <= 0:
            raise ValueError('q must be a positive integer')
        self.q = q

        if type(intercept_by_series) is not bool:
            raise ValueError('intercept_by_series must be a boolean')
        self.intercept_by_series = intercept_by_series
        
        if ar_constraint not in ['[-1,1]', '[0,1]']:
            raise ValueError("ar_constraint must be '[-1,1]' or '[0,1]'")
        self.ar_constraint = ar_constraint

        if sigma_factors_model not in ['AR', 'constant']:
            raise ValueError("sigma_factors must be 'AR' or 'constant'")
        self.sigma_factors_model = sigma_factors_model
        
        if loadings_constraint not in ['simplex', 'positive']:
            raise ValueError("loadings_constraint must be 'simplex' or 'positive'")
        self.loadings_constraint = loadings_constraint 
    
    def model(self, y=None, nan_inds=None, num_nans=None):
        '''
        Auto-regressive dynamic factor analysis model
        
        Parameters
        ----------
        y: array with shape (num_timesteps, num_series, num_horizons)
            Observed data
        '''
        # acquire and/or validate number of time steps and series
        if y is not None:
            if self.num_timesteps is not None and self.num_timesteps != y.shape[0]:
                raise ValueError('if provided, require num_timesteps = y.shape[0]')
            if self.num_series is not None and self.num_series != y.shape[1]:
                raise ValueError('if provided, require num_series = y.shape[1]')
            if self.num_horizons is not None and self.num_horizons != y.shape[2]:
                raise ValueError('if provided, require num_horizons = y.shape[2]')
            self.num_timesteps, self.num_series, self.num_horizons = y.shape
        
        if self.num_timesteps is None or self.num_series is None or self.num_horizons is None:
            raise ValueError('Must provide either y or three of num_timesteps, num_series and num_horizons')
        
        #if self.num_timesteps is None or self.num_series is None or self.num_horizons in None:
        #    raise ValueError('Must provide either y or all three of num_timesteps, num_series and num_horizons')
        
        # intercept for observation model, series-specific if requested
        # arranged as row vector for later broadcasting across timesteps
        if self.intercept_by_series:
            intercept_shape = (1, self.num_series)
        else:
            intercept_shape = (1,)

        intercept = numpyro.sample(
            'intercept',
            dist.Cauchy(0, 1),
            sample_shape=intercept_shape)
        
        # ar coefficients, shared across latent factors
        if self.ar_constraint == '[-1,1]':
            phi_l, phi_u = (-1, 1)
        elif self.ar_constraint == '[0,1]':
            phi_l, phi_u = (0, 1)
        phi = numpyro.sample(
            'phi',
            dist.Beta(3, 3),
            sample_shape=(1, self.p))
        
        # mean (ARVar_mu) and ar coefficients (alpha) of the variance of the error term in the latent factor analysis 
        ARVar_mu = numpyro.sample(
            'ARVar_mu', 
            dist.Normal(-1.5,1.0),
            sample_shape=(1, 1)
        )
        alpha = numpyro.sample(
            'alpha', 
            dist.Beta(15,2),
            sample_shape=(1, self.q)
        )

        alpha0 = ARVar_mu*(1-jnp.sum(alpha))

        # The variance of innovations in the AR process for the variance of the error term in the latent factor analysis
        sigma_nu = numpyro.sample(
            'sigma_nu',
            dist.Gamma(3,6),
            sample_shape=(1,1)
        )

        
        #The variance of different timepoints
        sigma_zeta = numpyro.sample(
            'sigma_zeta', 
            dist.Gamma(5,15),
            sample_shape=(1,1))
        
        beta0 = numpyro.sample(
            'beta0',
            dist.Normal(-0.5,0.8),
            sample_shape=(1,1)
        )

        beta1 = numpyro.sample(
            'beta1',
            dist.Normal(0.3,0.1),
            sample_shape=(1,1)
        )
                
        # factor loadings, shape (num_series, num_factors)
        if self.loadings_constraint == 'positive':
            factor_loadings = numpyro.sample(
                'factor_loadings',
                dist.Exponential(jnp.ones((self.num_series, self.num_factors))))
        elif self.loadings_constraint == 'simplex':
            factor_loadings = numpyro.sample(
                'factor_loadings',
                dist.Dirichlet(jnp.ones((self.num_factors,))),
                sample_shape=(self.num_series,))
        
        # initial values for factors, p time steps before time 0
        z_0 = numpyro.sample(
            'z_0',
            dist.Normal(0, 1),
            sample_shape=(self.p, self.num_factors))
        
        # initial values for error variance of latent factors, q time steps before time 0
        log_sigma_eta_0 = numpyro.sample(
            'log_sigma_eta_0',
            dist.HalfNormal(1),
            sample_shape=(self.q, 1)
        )

        h_rho = numpyro.sample(
            'h_rho',
            dist.Uniform(0.5,1),
            sample_shape=(1,)
        )

        Psi_a = numpyro.sample(
            'Psi_a',
            #dist.Normal(0,1),
            dist.Gamma(1.5,1),
            sample_shape=(1,)
        )

        Psi_b = numpyro.sample(
            'Psi_b',
            #dist.Normal(0,1),
            dist.Gamma(2,30),
            sample_shape=(1,)
        )

        
        # get horizon covariance based on the equations of sigma_h and matrixR 
        # (num_horizons by num_horizons)
        matrixR = AR1Root(h_rho, self.num_horizons)
        #sigma_h = 1+0.5*jnp.sqrt(jnp.arange(self.num_horizons))
        sigma_h = Psi_a+Psi_b*(jnp.arange(self.num_horizons))
        Sigma_eps_h_chol = jnp.matmul(jnp.diag(sigma_h), matrixR)

        #The variance of different series(locations)
        sigma_eps_l = numpyro.sample('sigma_eps_l',
            dist.Gamma(3,5), sample_shape=(self.num_series,1))      

        #get error variance of latent factors from AR(q) process
        def transition_ARvar(log_sigma_eta_prev, _):
            '''
            ARvar function for use with scan
            
            Parameters
            ----------
            log_sigma_eta_prev: array of shape (q, 1)
                error variance of the q time steps before time t
                the first row contains error variance of factor values for time t-1
            _: ignored, corresponds to integer time step value
            
            Returns
            -------
            log_sigma_eta: array of shape (q, 1)
                updated error variance of the q time steps ending at time t
                the first row contains error variance of factor values for time t
            log_sigma_eta_tt: array of shape (1,1)
                error variance at time t
            '''

            # calculate the mean for the error variance of the factors at time t, shape (1, 1, 1)
            log_sigma_mu_t = (jnp.matmul(alpha, log_sigma_eta_prev)+alpha0)

            # sample variances at time t, shape (1, 1, 1)
            log_sigma_eta = numpyro.sample('log_sigma_eta', dist.Normal(log_sigma_mu_t, sigma_nu))
            # updated variances for the q time steps ending at time t
            # shape (q, 1), first row is for time t
            log_sigma_eta_tt = jnp.concatenate((log_sigma_eta, log_sigma_eta_prev[:-1, :]), axis=0)

            return log_sigma_eta_tt, log_sigma_eta[0, :]
        
        
        timesteps = jnp.arange(self.num_timesteps)       

        # standard deviation of innovations in AR process for latent factors (num_timepoints, 1)
        if self.sigma_factors_model == 'constant':
            log_sigma_eta_t = numpyro.sample('log_sigma_eta_t', dist.HalfNormal(1))
            #log_sigma_eta_t = numpyro.sample('log_sigma_eta_t', dist.Normal(ARVar_mu, jnp.sqrt(sigma_nu**2/(1-alpha**2))))### Normal(ARVarmu, sigma_nu^2/(1-alpha^2))
            #log_sigma_eta_t = numpyro.sample('log_sigma_eta_t', dist.HalfNormal(1))
        elif self.sigma_factors_model == 'AR':
             _, log_sigma_eta_t = scan(transition_ARvar, log_sigma_eta_0, timesteps)
 

        # get latent factors from AR(p) process
        def transition_LF(factors_prev, timepoint):
            '''
            transition function for latent factor use with scan
            
            Parameters
            ----------
            factors_prev: array of shape (p, num_factors)
                latent factor values for the p time steps before time t
                the first row contains factor values for time t-1
            timepoint: corresponds to integer time step value
            
            Returns
            -------
            factors: array of shape (p, num_factors)
                updated latent factor values for the p time steps ending at time t
                the first row contains factor values for time t
            factors_t: array of shape (num_factors,)
                latent factor values at time t
            '''
            # calculate the mean for the factors at time t, shape (1, num_factors)
            m_t = jnp.matmul(phi, factors_prev)

            # sample factors at time t, shape (1, num_factors)
            if self.sigma_factors_model == 'constant':
                #factors_t = m_t + jnp.exp(log_sigma_eta_t) * factors_t_raw[timepoint, :]
                factors_t = numpyro.sample('factors', dist.Normal(m_t, jnp.exp(log_sigma_eta_t)))
            elif self.sigma_factors_model == 'AR':
                factors_t = numpyro.sample('factors', dist.Normal(m_t, jnp.exp(log_sigma_eta_t[timepoint,0])))


            # updated factors for the p time steps ending at time t
            # shape (p, num_factors), first row is for time t
            factors = jnp.concatenate((factors_t, factors_prev[:-1, :]), axis=0)
            
            return factors, factors_t[0, :]
        
        # scan over time steps; latent factors shape is (num_timepoints, num_factors)
        _, z_t = scan(transition_LF, z_0, timesteps)

        # observation model for y 
        ## zHmean is z times H (num_timepoints, num_series)
        ## zHmean add one more dimension for the horizon. It is duplicated becasue the means are the same 
        ## (num_horizons, num_timepoints, num_series)
        ## zHmean_trans transpose the order. (num_timepoints, num_series, num_horizons)
        zHmean = jnp.matmul(z_t, jnp.transpose(factor_loadings))
        zHmean = zHmean.reshape(zHmean.shape + (1,))
                   
        #mask = True 
        #if y is not None:
            # Mask nan values, then substitute a default value to
            # avoid nans in gradients
        #    mask = ~jnp.isnan(y)

        # indicate nan values, and replace to 0
        if y is not None:
        #if(jnp.sum(jnp.isnan(y))) :
            # isnan = jnp.isnan(y)
            y_impute = numpyro.param('y_impute', jnp.zeros(num_nans))
            y = y.at[nan_inds].set(y_impute)
            #print(y.shape)

        #with numpyro.handlers.mask(mask=mask):            
        #if self.sigma_factors_model == 'constant':
            #The variance of observation noise when factors model is constant
        #    log_sigma_eps = numpyro.sample(
        #        'log_sigma_eps',
        #        dist.HalfNormal(1),
        #        sample_shape=(self.num_timesteps, self.num_series, self.num_horizons))
        #    numpyro.sample(
        #        'y',
        #        dist.Normal(loc=intercept + zHmean, scale=log_sigma_eps),  
        #        obs=y)

        #elif self.sigma_factors_model == 'AR':
            #The variance of different timepoints
        if self.sigma_factors_model == 'constant':
            log_sigma_eps_t = numpyro.sample('log_sigma_eps_t',
                dist.HalfNormal(1), sample_shape=(1,1))
        elif self.sigma_factors_model == 'AR':
            log_sigma_eps_t = numpyro.sample('log_sigma_eps_t',
                dist.Normal(beta0+beta1*log_sigma_eta_t, sigma_zeta))
            

        sigma_tl = jnp.matmul(jnp.exp(log_sigma_eps_t), jnp.transpose(sigma_eps_l))  
        sigma_tl_reshape = sigma_tl.reshape(sigma_tl.shape + (1,1,))
        Sigma_eps_h_chol_reshape = Sigma_eps_h_chol.reshape((1,1,) + Sigma_eps_h_chol.shape)
        Omega_tl = sigma_tl_reshape * Sigma_eps_h_chol_reshape       
    
        numpyro.sample(
            'y',
            dist.MultivariateNormal(loc=intercept + zHmean, scale_tril=Omega_tl),
            obs=y)
                    

    
    def fit(self, y, rng_key, num_warmup=1000, num_samples=1000, num_chains=1,
            print_summary=False):
        '''
        Fit model using MCMC
        
        Parameters
        ----------
        y: array with shape (num_timesteps, num_series, num_horizons)
            Observed data
        rng_key: random.PRNGKey
            Random number generator key to be used for MCMC sampling
        num_warmup: integer
            Number of warmup steps for the MCMC algorithm
        num_samples: integer
            Number of sampling steps for the MCMC algorithm
        num_chains: integer
            Number of MCMC chains to run
        print_summary: boolean
            If True, print a summary of estimation results
        
        Returns
        -------
        array with samples from the posterior distribution of the model parameters
        '''
        start = time.time()
        sampler = numpyro.infer.NUTS(self.model)
        self.mcmc = numpyro.infer.MCMC(
            sampler,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False if 'NUMPYRO_SPHINXBUILD' in os.environ else True,
        )
        self.mcmc.run(rng_key, y=y, nan_inds=jnp.nonzero(jnp.isnan(y)), num_nans=int(jnp.isnan(y).sum()))
        print('\nMCMC elapsed time:', time.time() - start)
        
        if print_summary:
            self.mcmc.print_summary()
        return self.mcmc.get_samples()
    
    
    def sample(self, rng_key, condition={}, num_samples=1):
        '''
        Draw a sample from the joint distribution of parameter values and data
        defined by the model, possibly conditioning on a set of fixed values.
        
        Parameters
        ----------
        rng_key: random.PRNGKey
            Random number generator key to be used for sampling
        condition: dictionary
            Optional dictionary of parameter values to hold fixed
        num_samples: integer
            Number of samples to draw. Ignored if condition is provided, in
            which case the number of samples will correspond to the shape of
            the entries in condition.
        
        Returns
        -------
        dictionary of arrays of sampled values
        '''
        if condition == {}:
            predictive = numpyro.infer.Predictive(self.model,
                                                  num_samples=num_samples)
        else:
            predictive = numpyro.infer.Predictive(self.model,
                                                  posterior_samples=condition)
        
        return predictive(rng_key)

