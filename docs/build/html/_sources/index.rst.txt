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


.. toctree::
   :maxdepth: 1
   :caption: Notebooks


