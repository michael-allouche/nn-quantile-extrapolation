# Estimation of extreme quantiles from heavy-tailed distributions with neural networks
Implementation of the paper ["Estimation of extreme quantiles from heaby-tailed distributions with neural netwroks", 2022](LINK),
by Michaël Allouche, [Stéphane Girard](http://mistis.inrialpes.fr/people/girard/) and [Emmanuel Gobet](http://www.cmap.polytechnique.fr/~gobet/).

The repo contains the codes for comparing our proposed extreme quantile estimator with 7 other known estimators in the literature 
on simulated data.

## Abstract
We propose new parametrizations for neural networks in order to estimate extreme quantiles in both non-conditional and conditional heavy-tailed settings. All proposed neural network estimators feature a bias correction based on an extension of the usual second-order condition to an arbitrary order.
The convergence rate of the uniform error between extreme log-quantiles and their neural network approximation is established.
The finite sample performances of the non-conditional neural network estimator are compared to other bias-reduced extreme-value competitors on simulated data. Finally, 
the conditional neural network estimators are implemented to investigate the behaviour of extreme rainfalls as functions of their geographical location in the southern part of France.

## Dependencies
Clone the repo

```
git clone https://github.com/michael-allouche/nn-quantile-extrapolation.git
cd nn-quantile-extrapolation
```

Install the Python version and the library requirements used in this repo

```
conda create --name nnQextreme-session python=3.8.12
conda activate nnQextreme-session
conda install --file requirements.txt
```

## Simulated data
Seven heavy-tailed distributions are implemented in `./extreme/distribution.py`:

**Burr, NHW, Fréchet, Fisher, GPD, Inverse Gamma, Student**.

In `run_evt_estimators.py`, one can update the `dict_runner` with the desired parametrization. 
Next, run `run_evt_estimators.py` to compute all the quantile estimators at both quantile levels alpha=1/n and alpha=1/(2n) . 
For example, estimations applied to 1000 replications of 500 samples issued from a Burr distribution:

`python run_evt_estimators.py -d burr -r 1000 -n 500`

Once the run is finished, all the metrics for each estimator are saved in the folder `./ckpt`.

In the notebook, you can display a result table. For example

```
from extreme.estimators import evt_estimators 
evt_estimators(n_replications=1000, params={"evi":0.125, "rho": -1.},
                distribution="burr", 
               n_data=500, n_quantile="2n")
```
```
Estimators     W	RW	CW	CH	CHp	PRBp	CHps	PRBps

RMSE	      0.0471	0.0095	0.0063	0.0155	0.0149	0.015	0.0135	0.0164
```
You can also plot the bias, the variance and the RMSE

```
from extreme import visualization as statsviz
statsviz.evt_quantile_plot(n_replications=1000, 
   		           params={"evi":0.125, "rho": -0.125}, 
                           distribution="burr", 
                           n_data=500, 
                           n_quantile="2n")
```
![simulations](imgs/simulations_test.jpg)

## Conditional extension - rainfall data on the Cevennes-Vivarais region (France)


## Citing
