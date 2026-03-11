# ADCP-internship (ongoing):

This repository contains processing for upward-looking Acoustic Doppler Current Profiler (ADCP) data and export for further analysis of coastal circulation. 
The data was collected outside the coast of Algarve (Armona) during May - November 2024.

Implements Python 3.12, DOLfYN, Matplotlib, NumPy, Xarray and SciPy.

## List of content:

### my_functions.py: custom functions 

### raw_data_processing.ipynb: Jupyter Notebook containing the workflow.
  #### Processing summary:
- Removing data before deployment
- Instrument attitude control
- Removing surface interference
- Filtering of tides
- Applying correlation filter
- Removing outliers from signal amplitude
- Controlling coordinate system and setting declination
- Assigning quality flags to data
- Exporting data as netcdf

### data_analysis.ipynb: Jupyter Notebook containing further analyis of the processed data.