.. EddyFlux documentation master file, created by
   sphinx-quickstart on Tue Jul  1 21:35:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EddyFlux Documentation
=======================

This project focuses on exploring and modeling Net Ecosystem Exchange (NEE) using eddy covariance data. It includes:

- Visual analysis of flux and meteorological variables
- Feature selection and symbolic modeling techniques
- Predictive modeling of NEE using advanced ML methods

Notebook Summaries
-------------------

**1. Explore_Dataset.ipynb**  
This notebook focuses on data loading, format conversion to `parquet`, and comprehensive visual exploration of environmental drivers of CO₂ flux. Key steps include:

- Histograms and time series plots of NEE
- Comparisons across towers and time periods
- Diurnal patterns in soil/air temperature, radiation, and evapotranspiration
- Analysis of radiation balance and GPP vs RECO
- Correlation matrix construction
- Principal Component Analysis (PCA)
- Integration of external datasets such as GPM precipitation

**2. NEE_Modeling.ipynb**  
This notebook develops predictive models for NEE based on selected environmental features. Highlights include:

- Directional analysis of shortwave radiation
- Mutual information-based feature relevance ranking
- Use of symbolic regression to derive interpretable NEE equations
- Deployment of Kolmogorov–Arnold Networks (KANs) to discover symbolic nonlinear patterns
- Several complex, interpretable expressions approximating NEE from multiple variables (e.g., VPD, Ta, RH, SW, etc.)

.. toctree::
   :maxdepth: 1
   :caption: Notebooks


