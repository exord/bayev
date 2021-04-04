# Bayev

A package to estimate Bayesian marginal likelihoods based on posterior samples.

## The estimators
The package implements two main methods to estimate the marginal likelihood:

* the estimator by introduced [Chib & Jeliazkov 2001](https://www.tandfonline.com/doi/abs/10.1198/016214501750332848) (implemented in module `chib`).

* the importance sampling estimator by [Perrakis, Ntzoufras, and Tsionas 2014](https://www.sciencedirect.com/science/article/abs/pii/S0167947314000814?via%3Dihub) (implemented in module `perrakis`).

In addition, a small module to compute the Harmonic estimator (described, for example, by Kass & Raftery 1995), is also provided.

## Usage

The package provides functions to compute the estimators in each of the packages (`chib.compute_cj_estimate` and `perrakis.compute_perrakis_estimate`), and a "meta-function" in the `run` that allow running a series of simulations (`run_montecarlo`) to estimate the variance of the estimator.

A more detailed documentation is not available (sorry!), but you're welcome to send me questions and comments to [rdiaz@unsam.edu.ar](mailto:rdiaz@unsam.edu.ar).

## Disclaimer and attribution

This is work in progress, and some small additions are constantly being implemented. But most of the work was done during my postdoc at the University of Geneva. So, if you use this code, please cite [Díaz et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...585A.134D/abstract).

```
@ARTICLE{2016A&A...585A.134D,
       author = {{D{\'\i}az}, R.~F. and {S{\'e}gransan}, D. and {Udry}, S. and {Lovis}, C. and {Pepe}, F. and {Dumusque}, X. and {Marmier}, M. and {Alonso}, R. and {Benz}, W. and {Bouchy}, F. and {Coffinet}, A. and {Collier Cameron}, A. and {Deleuil}, M. and {Figueira}, P. and {Gillon}, M. and {Lo Curto}, G. and {Mayor}, M. and {Mordasini}, C. and {Motalebi}, F. and {Moutou}, C. and {Pollacco}, D. and {Pompei}, E. and {Queloz}, D. and {Santos}, N. and {Wyttenbach}, A.},
        title = "{The HARPS search for southern extra-solar planets. XXXVIII. Bayesian re-analysis of three systems. New super-Earths, unconfirmed signals, and magnetic cycles}",
      journal = {\aap},
     keywords = {techniques: radial velocities, methods: data analysis, methods: statistical, planetary systems, Astrophysics - Earth and Planetary Astrophysics},
         year = 2016,
        month = jan,
       volume = {585},
          eid = {A134},
        pages = {A134},
          doi = {10.1051/0004-6361/201526729},
archivePrefix = {arXiv},
       eprint = {1510.06446},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2016A&A...585A.134D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
