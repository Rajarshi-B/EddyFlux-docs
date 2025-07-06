.. EddyFlux documentation master file, created by
   sphinx-quickstart on Tue Jul  1 21:35:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EddyFlux Documentation
=======================

This project focuses on exploring and modeling Net Ecosystem Exchange (NEE) using eddy covariance data. It includes:

- Explore_Dataset.ipynb opens the Eddy Covariance Dataset and visualizes it.

- HLS_Generate_NDVI.ipynb needs a boundary-geometry (geojson file) within which NDVI it extracts the satellite derived NDVI.

- Interpolate_Plot_NDVI_vs_GPP.ipynb interpolates the sparse NDVI data to 30 min resolution and visualizes alongside GPP.

- NEE_Modelling.ipynb finds the most important features and fits various machine learning model/tunes them.

- Explore_Grazing.ipynb finds the impact of grazing ungrazed vs grazed scenarios using machine learning models.

Notebook Summaries
-------------------

.. dropdown:: 1. Explore_Dataset.ipynb
  :open:

  This notebook performs exploratory data analysis (EDA) on eddy covariance datasets. The goal is to prepare the data for modeling Net Ecosystem Exchange (NEE) by cleaning, transforming, and visualizing relevant variables.

  .. dropdown:: 1. Loading the Data
    :open:

    The dataset, originally in Excel format, is loaded using `pandas.ExcelFile`. Each sheet represents data from a different tower site.

    Key steps:
    - Load each sheet into a DataFrame
    - Inspect columns, datatypes, and record counts

    .. code-block:: python

        file = pd.ExcelFile('Data_Quanterra.xls')
        df_HurstWest = pd.read_excel(file, sheet_name='HurstWest')

    Metadata such as column count and structure is printed using `.info()` and `.columns`.

  ---

  .. dropdown:: 2. Saving as Parquet for Efficient I/O
    :open:

    To improve I/O speed, the large Excel files are converted to `.parquet` format. This also standardizes time columns using a helper function based on literature.

    Highlights:
    - Datetime formatting
    - Feature extraction (hour, DOY)
    - Data type enforcement (e.g., float conversion)

    .. code-block:: python

        df_FergusonNE = add_time_vars(df_FergusonNE, 'MIDPOINT_DATETIME')
        df_HurstWest.to_parquet('HurstWest.parquet')

  ---

  .. dropdown:: 3. Visualizing the Data & Finding Predictors
    :open:

    This section visualizes key flux and meteorological variables and looks for meaningful patterns.

  ---

  .. dropdown:: 3.1 Visualizing NEE
    :open:

    - Histogram and time-series plots of NEE
    - Checking for yearly transitions or gaps
    - Comparison of NEE across towers

    .. code-block:: python

        fig, axes = plt.subplots()
        axes.plot(HE['TIMESTAMP'], HE['NEE'])

  ---

  .. dropdown:: 3.2 Temperature and Radiation Effects
    :open:

    - Examine air/soil temperature and shortwave/longwave radiation
    - Net radiation calculated as:  
      `Net Radiation = Incoming - Outgoing`

    .. code-block:: python

        HE['NET_RAD'] = HE['SW_IN'] - HE['SW_OUT']

  ---

  .. dropdown:: 3.3 Ecosystem Respiration and GPP
    :open:

    - Seasonal and diurnal analysis of RECO and GPP
    - Function of soil temperature and light availability

    Questions posed:
    - How is RECO calculated?
    - What explains variation in ET and LE?

  ---

  .. dropdown:: 3.4 Evapotranspiration Calculations
    :open:

    - Evapotranspiration is derived from latent heat flux:  
      `ET = LE / λ`

    - The latent heat of vaporization is temperature dependent

    Referenced method:  
    [StackExchange explanation on latent heat](https://earthscience.stackexchange.com/questions/20733)

  ---

  .. dropdown:: 3.5 Correlation Matrix
    :open:

    - Mean correlation plots across variables
    - Special attention to: VPD, RH, Air Temp

    VPD = \( e_s - e \), where \( e_s \) is from Clausius-Clapeyron relation.

  ---

  .. dropdown:: 3.6 Principal Component Analysis (PCA)
    :open:

    - PCA is applied to reduce dimensionality and interpret variable groupings.

    .. code-block:: python

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(df[numeric_columns])

  ---

  .. dropdown:: 3.7 External Precipitation Data
    :open:

    - Plots daily precipitation from GPM
    - Compared with soil/air moisture

    .. code-block:: python

        plt.plot(GPM['time'], GPM['precip'])

.. dropdown:: 2. HLS_Generate_NDVI.ipynb
  :open:

  This notebook walks through the complete workflow for generating **NDVI (Normalized Difference Vegetation Index)** and **EVI (Enhanced Vegetation Index)** maps from Harmonized Landsat and Sentinel (HLS) surface reflectance data. It demonstrates how to search for HLS data using NASA Earthdata, load selected spectral bands directly from cloud-hosted GeoTIFFs, apply spatial subsetting and quality filtering, compute NDVI/EVI, and export results as stacked time series or GeoTIFFs.

  .. dropdown:: 1. Background
    :open:

    The Harmonized Landsat and Sentinel (HLS) project provides surface reflectance products that merge Sentinel-2 and Landsat-8 observations at 30 m spatial resolution and ~2–3 day revisit. These are distributed as Cloud Optimized GeoTIFFs (COGs), allowing selective access to bands (e.g., Red, NIR, Blue) directly in the cloud.

    In this notebook, we use the cloud-native geospatial Python stack to:

    - Query specific HLS granules using spatial and temporal filters
    - Load and subset Red, NIR, and Blue bands
    - Apply cloud masks using the Fmask layer
    - Compute NDVI and EVI vegetation indices
    - Stack multi-date raster scenes into a temporal composite
    - Export GeoTIFFs and summary statistics for further analysis

    **Vegetation Indices:**

    - **NDVI** = (NIR - Red) / (NIR + Red): Indicates vegetation greenness.
    - **EVI**  = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1): More robust in high biomass or shadowed areas.

  .. dropdown:: 2. Getting Started
    :open:

    ### 2.1 Import Packages

    Load required Python libraries for geospatial, visualization, and cloud access tasks.

    .. code-block:: python

      import os
      from datetime import datetime
      import numpy as np
      import pandas as pd
      import geopandas as gp
      from skimage import io
      import matplotlib.pyplot as plt
      from osgeo import gdal
      import xarray as xr
      import rioxarray as rxr
      import hvplot.xarray
      import hvplot.pandas
      import earthaccess

    ### 2.2 Earthdata Login Authentication

    Use `earthaccess` to log in with your NASA Earthdata credentials.

    .. code-block:: python

      auth = earthaccess.login()
      auth

  .. dropdown:: 3. Finding HLS Data
    :open:

    Search for Sentinel-2 based HLS surface reflectance products over a region and time period of interest.

    .. code-block:: python

      results = earthaccess.search_data(
          short_name="HLSS30",
          cloud_hosted=True,
          bounding_box=(-106.623, 35.056, -106.491, 35.154),
          temporal=("2021-04-01", "2021-11-01")
      )

  .. dropdown:: 4. Accessing HLS COG Data in the Cloud
    :open:

    ### 4.1 Subset by Band

    Load URLs for individual bands needed for NDVI/EVI calculation: Red (B04), NIR (B8A), Blue (B02), and Fmask.

    .. code-block:: python

      assets = earthaccess.load_asset_urls(results, asset_types=["B04", "B8A", "B02", "Fmask"])

    ### 4.2 View Browse Image

    Visual preview of scene using `browse` URL.

    .. code-block:: python

      io.imshow(assets[0]["browse"])

    ### 4.3 Load COGs into Memory

    Read bands into memory using `rioxarray`.

    .. code-block:: python

      red = rxr.open_rasterio(assets[0]["B04"], masked=True).squeeze()
      nir = rxr.open_rasterio(assets[0]["B8A"], masked=True).squeeze()
      blue = rxr.open_rasterio(assets[0]["B02"], masked=True).squeeze()
      fmask = rxr.open_rasterio(assets[0]["Fmask"], masked=True).squeeze()

    ### 4.4 Subset Spatially

    Clip raster bands to a shapefile-defined area of interest.

    .. code-block:: python

      boundary = gp.read_file("data/boundary.shp")
      red_clip = red.rio.clip(boundary.geometry, boundary.crs, drop=True)

    ### 4.5 Apply Scale Factor

    Convert reflectance values to real scale by dividing by 10,000.

    .. code-block:: python

      red_scaled = red_clip / 10000
      nir_scaled = nir_clip / 10000
      blue_scaled = blue_clip / 10000

  .. dropdown:: 5. Processing HLS Data
    :open:

    ### 5.1 Calculate NDVI and EVI

    Compute both vegetation indices for comparison and flexibility.

    .. code-block:: python

      ndvi = (nir_scaled - red_scaled) / (nir_scaled + red_scaled)
      evi = 2.5 * (nir_scaled - red_scaled) / (nir_scaled + 6 * red_scaled - 7.5 * blue_scaled + 1)

    NDVI is more widely used and interpretable for general vegetation mapping, while EVI performs better in dense canopies or when atmospheric noise is high.

    ### 5.2 Quality Filtering

    Use Fmask to remove cloudy or shadowed pixels:

    .. code-block:: python

      mask = fmask_clip.where((fmask_clip == 0) | (fmask_clip == 1))
      ndvi_filtered = ndvi.where(mask)
      evi_filtered = evi.where(mask)

    ### 5.3 Export to COG

    Save results to GeoTIFF format for GIS or web map usage.

    .. code-block:: python

      ndvi_filtered.rio.to_raster("output/NDVI.tif")
      evi_filtered.rio.to_raster("output/EVI.tif")

  .. dropdown:: 6. Automation
    :open:

    Batch process multiple scenes by looping through assets for each date, applying the same steps (load → clip → scale → compute → filter → export) to produce NDVI and EVI time series.

    This supports applications like seasonal monitoring, crop stress analysis, or vegetation phenology studies.

  .. dropdown:: 7. Stacking HLS Data
    :open:

    ### 7.1 Open and Stack COGs

    Combine all exported NDVI or EVI files into one time-aware `xarray.DataArray` for temporal analysis.

    .. code-block:: python

      stack_ndvi = xr.concat([rxr.open_rasterio(fp) for fp in ndvi_filepaths], dim="time")
      stack_evi = xr.concat([rxr.open_rasterio(fp) for fp in evi_filepaths], dim="time")

    ### 7.2 Visualize Stacked Time Series

    View how NDVI or EVI evolves over time across your AOI.

    .. code-block:: python

      stack_ndvi.hvplot(x='x', y='y', groupby='time', cmap='YlGn')
      stack_evi.hvplot(x='x', y='y', groupby='time', cmap='viridis')

  .. dropdown:: 8. Export Statistics
    :open:

    Compute spatial statistics such as mean or maximum NDVI/EVI across time and export for reporting or machine learning.

    .. code-block:: python

      mean_ndvi = stack_ndvi.mean(dim='time')
      mean_evi = stack_evi.mean(dim='time')

      mean_ndvi.rio.to_raster("output/mean_NDVI.tif")
      mean_evi.rio.to_raster("output/mean_EVI.tif")



.. dropdown:: 3. NEE_Modeling.ipynb
  :open:

  This notebook explores various modeling approaches for estimating Net Ecosystem Exchange (NEE) from eddy covariance data. It includes techniques ranging from mutual information analysis and symbolic regression to Gaussian processes and neural networks. The objective is to build interpretable and high-performing models that capture the directional influences of meteorological drivers on NEE.

  .. dropdown:: 1. Directional Influence
    :open:

    This section investigates how different environmental variables influence NEE depending on wind direction. By segmenting the data based on wind direction, the analysis visualizes and quantifies the varying relationships between NEE and climate variables. This helps reveal directional dependencies and site-specific heterogeneity in ecosystem flux responses.

  .. dropdown:: 2. PLOT Shortwave
    :open:

    Shortwave radiation is a primary driver of photosynthesis and NEE. This section visualizes shortwave radiation over time, showing its diurnal and seasonal dynamics. Plots include time series and scatter relationships with NEE to highlight their interdependence.

  .. dropdown:: 3. Check Mutual Information
    :open:

    Mutual Information (MI) is computed between NEE and potential predictor variables to measure the strength of both linear and non-linear relationships. The MI scores are used to rank variables and inform the feature selection process for subsequent modeling techniques. The analysis aids in identifying the most informative meteorological and environmental variables.

  .. dropdown:: 4. Symbolic Regression
    :open:

    Symbolic regression is applied to derive compact, interpretable mathematical expressions that model NEE from selected input variables. The focus is on identifying non-linear structures and explicit formulae that relate NEE to environmental drivers.

    .. dropdown:: 4.1 KAN (Kolmogorov–Arnold Networks)
      :open:

      Kolmogorov–Arnold Networks (KANs) are explored for symbolic modeling. These networks use structured layers of functional transformations (like sin, cos, log, etc.) to learn symbolic representations of the data. This section discusses training and visualizing KAN models for NEE estimation.

    .. dropdown:: 4.2 PySR and GPLearn
      :open:

      This section implements two symbolic regression libraries—`PySR` and `gplearn`—to discover closed-form expressions for NEE. The results are compared based on accuracy and interpretability.

      .. dropdown:: 4.2.1 GPLearn
        :open:

        The `gplearn` library uses genetic programming to evolve symbolic expressions. This subsection shows how models are trained using a function set tailored to ecological data and evaluates the resulting formulas.

  .. dropdown:: 5. Polynomial Ridge Regression
    :open:

    A polynomial regression model with Ridge (L2) regularization is built to capture complex non-linear patterns in NEE without overfitting. The polynomial terms allow the model to account for higher-order interactions, while the Ridge penalty helps maintain generalizability. Model diagnostics, predictions, and performance metrics are visualized and discussed.

  .. dropdown:: 6. Gaussian Process Regression
    :open:

    Gaussian Process Regression (GPR) is used to model NEE with uncertainty quantification. Due to its non-parametric nature, GPR is well-suited for modeling non-linear, noisy ecological data, especially when sample sizes are small. This section includes kernel selection, model fitting, prediction with uncertainty bounds, and performance evaluation.

  .. dropdown:: 7. NN Learns GPP and RECO
    :open:

    A neural network is trained to estimate Gross Primary Production (GPP) and Ecosystem Respiration (RECO) separately, from which NEE is inferred as the difference (GPP - RECO). This biologically grounded decomposition approach enhances the interpretability of the model and aligns with ecological theory. The architecture, training process, and model evaluation are covered in detail.
  .. dropdown:: 8. Tune
    :open:

    This section attempts hyperparameter tuning for neural networks or other models previously applied. The process helps improve model performance by selecting optimal architecture and training parameters.

  .. dropdown:: 9. Tune Properly
    :open:

    A refined approach to hyperparameter tuning, this section ensures methodical validation techniques and possibly includes early stopping, regularization, or cross-validation strategies.

  .. dropdown:: 10. 1D CNN
    :open:

    Implements a 1D Convolutional Neural Network (CNN) architecture for learning temporal patterns in NEE or flux-related features. CNNs help extract local dependencies within time series inputs, improving modeling power.

  .. dropdown:: 11. SVM
    :open:

    This section applies Support Vector Machines (SVM) for regression or classification related to NEE. It investigates kernel selection (e.g., linear or RBF) and model performance on ecological data.

  .. dropdown:: 12. RBF
    :open:

    Focuses on Radial Basis Function (RBF) kernel usage, particularly in SVM or other models like kernel ridge regression. This enhances the non-linear modeling capabilities.

  .. dropdown:: 13. Ridge Regression + NEE Equation as Feature
    :open:

    Ridge Regression is applied with an additional input feature derived from a known NEE equation. This hybrid approach blends empirical data and process-based knowledge.

  .. dropdown:: 14. Random Forest
    :open:

    Trains a Random Forest model on selected features. Includes tuning and comparison against previous models. Model interpretation (e.g., feature importances) is also discussed.

    .. dropdown:: 14.1 Running for loaded hyperparameters
      :open:

      Loads and uses pre-optimized hyperparameters to train the final Random Forest model.

    .. dropdown:: 14.2 Bootstrap correct
      :open:

      Applies bootstrap resampling for uncertainty estimation and performance robustness.

    .. dropdown:: 14.3 Bootstrap Correct (New)
      :open:

      An updated implementation of bootstrap evaluation to assess Random Forest generalization across different samples.

    .. dropdown:: 14.4 RF with NEE equation as input
      :open:

      Enhances the Random Forest model by adding symbolic NEE equation output as an additional feature.

  .. dropdown:: 15. Bootstrap
    :open:

    General overview of bootstrap procedures applied across models to estimate uncertainty and model stability.

    .. dropdown:: 15.1 Diagnose
      :open:

      Diagnoses issues with model variance and bias through analysis of bootstrap sample distributions.

  .. dropdown:: 16. XGBoost
    :open:

    Gradient-boosted decision trees are applied using the XGBoost library. It includes hyperparameter tuning, interpretation, and performance evaluation.

    .. dropdown:: 16.1 Load Hyperparameters and Run
      :open:

      Loads optimal parameters and executes the XGBoost model for final predictions.

    .. dropdown:: 16.2 NEE equation as input
      :open:

      Integrates symbolic regression outputs or process-based NEE formulations as input to XGBoost.

    .. dropdown:: 16.3 XGBoost Bootstrap
      :open:

      Applies bootstrap sampling to evaluate uncertainty and generalizability of the XGBoost model.

    .. dropdown:: 16.4 RF vs XGB
      :open:

      A comparative evaluation of Random Forest and XGBoost models in terms of accuracy and stability.

    .. dropdown:: 16.5 XGB SHAP
      :open:

      Explores SHAP (SHapley Additive exPlanations) values for interpreting feature contributions to XGBoost predictions.

  .. dropdown:: 17. LSTM
    :open:

    Long Short-Term Memory (LSTM) networks are applied for time series prediction of NEE. This section defines architecture, handles sequence formatting, and evaluates results.

  .. dropdown:: 18. Bias Correction
    :open:

    Investigates post-processing techniques to correct model bias, often by learning residuals or blending predictions with known equations.

  .. dropdown:: 19. NEE Equation used
    :open:

    Introduces the specific NEE equation(s) used for feature generation, correction, or hybrid modeling.

    .. dropdown:: 19.1 Direct Equation
      :open:

      Directly implements a known NEE equation based on biophysical relationships for estimation or model input.

    .. dropdown:: 19.2 NEE + Error Correction
      :open:

      Combines equation-based estimates with error modeling for refined predictions.

    .. dropdown:: 19.3 NEE prediction as input
      :open:

      Uses outputs from symbolic or empirical NEE models as features for downstream learning algorithms.

      .. dropdown:: 19.3.1 NEE Equation as input (Latest)
        :open:

        Latest approach for integrating symbolic NEE expressions into machine learning workflows.

      .. dropdown:: 19.3.2 SHAP
        :open:

        SHAP analysis for this hybrid-input model to interpret how symbolic NEE estimates influence model predictions.

      .. dropdown:: 19.3.3 SHAP with XGB
        :open:

        Detailed SHAP plots specifically for the XGBoost model using symbolic NEE features.

    .. dropdown:: 19.4 NEE*(1+rf)
      :open:

      A modeling idea that scales symbolic NEE by a correction factor learned by a random forest model.

    .. dropdown:: 19.5 Visualizing NEE
      :open:

      Plots and diagnostics showing modeled NEE time series and residuals for final visualization and interpretation.


.. toctree::
   :maxdepth: 1
   :caption: Notebooks


