# Bayev

A package to estimate Bayesian marginal likelihoods based on posterior samples.

## The estimators
The package implements two main methods to estimate the marginal likelihood:

* the estimator by introduced [Chib & Jeliazkov 2001](https://www.tandfonline.com/doi/abs/10.1198/016214501750332848) (implemented in module `chib`).

* the importance sampling estimator by [Perrakis, Ntzoufras, and Tsionas 2014](https://www.sciencedirect.com/science/article/abs/pii/S0167947314000814?via%3Dihub) (implemented in module `perrakis`).

In addition, a small module to compute the Harmonic estimator (described, for example, by Kass & Raftery 1995), is also provided.
