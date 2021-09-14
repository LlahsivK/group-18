# Group 18 - Ceilometer PBL height analysis Project - Git Repository

## Team members

* Thomas Koh - 329888
* Vishall Krishnan - 1018473
* Shaohua Liu - 1150336
* Yujing Yang - 979613 

## Problem statement

Analysis of planetary boundary layer (PBL) height in Lidcombe and Merriwa in NSW.

## Background

Department of Planning, Infrastructure and Environment (DPIE) of the NSW government have implemented a new device (CL51 ceilometer) that measures the PBL hegihts in two locations in NSW. We have been comissioned to:
* Investigate the efficacy of the new device in producing PBL heights; and
* Investigate methods of improving on the modelling of future PBL heights taking into account information from the new device.

## Analysis 1

Visualisation and comparison of PBL height as captured by the CL51 ceilometer against two status quo models:
* The CCAM-CTM model; and
* The WRF model

Here we perform visual and statistical comparisons of the PBL heights against the two status quo models.

## Analysis 2

Comparison of PBL heights against metereological factors that may influence the PBL height. Here we look to establish if there are causal relationships between PBL heights and these "external" factors.

## Analysis 3

Modelling of PBL height taking into account factors in Analysis 2 and actual historical PBL heights as derived from the CL51 ceilometer. Here our aim is to see if incorporation of historical PBL heights can improve upon the prediction of PBL heights as compared to the CCAM-CTM and WRF models.

We consider 4 broad methods for modelling the PBL heights:
* Regression methods;
* Time series methods;
* Ensemble methods; and
* Neural network methods.

## Software used

Python and R

### Packages used

##### Python libraries
Library name| Version
-------------|--------
Pandas| 1.3.2
Numpy| 1.20.3
scikit-learn| 0.24.2
matplotlib| 3.4.2
fastdtw| 1.1.0
xgboost| 1.4.0
lightgbm| 3.2.1
seaborn| 0.11.0
datetime

##### R libraries
Library name| Version
-------------|--------
ggplot2| 3.3.5
dplyr| 1.0.7
dtw| 1.22
tidyr| 1.1.3
tidyverse| 1.3.1
forecast| 8.15
