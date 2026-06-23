import numpy as np
from scipy.stats import norm
from typing import Dict, Any

class BayesianAlphaValidator:
    """Uses Bayesian updating to determine the probability that alpha is still alive."""
    
    def __init__(self, prior_mu: float, prior_std: float):
        """
        prior_mu: Expected mean Sharpe or return from research.
        prior_std: Standard deviation of the expected metric.
        """
        self.prior_mu = prior_mu
        self.prior_std = prior_std
        
        self.posterior_mu = prior_mu
        self.posterior_std = prior_std
        
    def update(self, observed_mean: float, observed_std: float, n_samples: int) -> None:
        """
        Updates the posterior distribution using conjugate priors (Normal-Normal).
        """
        if n_samples == 0 or observed_std == 0:
            return
            
        # Variance of the sample mean
        var_sample_mean = (observed_std ** 2) / n_samples
        var_prior = self.posterior_std ** 2
        
        # Bayesian update formulas for normal conjugate prior
        self.posterior_mu = (var_sample_mean * self.posterior_mu + var_prior * observed_mean) / (var_sample_mean + var_prior)
        self.posterior_std = np.sqrt((var_prior * var_sample_mean) / (var_prior + var_sample_mean))
        
    def get_probabilities(self, zero_threshold: float = 0.0, degrade_threshold: float = None) -> Dict[str, float]:
        """
        Calculates probabilities of alpha states.
        zero_threshold: Threshold below which alpha is considered 'dead' (e.g., 0 Sharpe)
        degrade_threshold: Threshold below which alpha is 'degraded' (e.g., 0.5 * prior_mu)
        """
        if degrade_threshold is None:
            degrade_threshold = self.prior_mu * 0.5
            
        # P(alpha < zero_threshold)
        p_dead = norm.cdf(zero_threshold, loc=self.posterior_mu, scale=self.posterior_std)
        
        # P(alpha < degrade_threshold)
        p_below_degrade = norm.cdf(degrade_threshold, loc=self.posterior_mu, scale=self.posterior_std)
        
        p_degraded = p_below_degrade - p_dead
        p_alive = 1.0 - p_below_degrade
        
        return {
            "p_alive": float(p_alive),
            "p_degraded": float(p_degraded),
            "p_dead": float(p_dead),
            "posterior_mu": float(self.posterior_mu),
            "posterior_std": float(self.posterior_std)
        }
