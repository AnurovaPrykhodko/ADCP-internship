# ADCP-internship (ongoing):

This repository contains processing for Acoustic Doppler Current Profiler (ADCP) data, analysis of coastal circulation and quality flagging according to IODE Quality Flag Standard. 
The data was collected outside the coast of Algarve (Armona) during May - November 2024. The ADCP was upward-looking.

Implements Python 3.12, DOLfYN, Matplotlib, NumPy, Xarray and SciPy.

## List of content:

### my_functions.py: Custom functions.

### raw_data_processing.ipynb: Notebook containing the processing of the raw data.
  #### Summary:
- Removing data before deployment
-	Adding transducers height
- Instrument attitude control
-	Controlling coordinate system
-	Setting magnetic declination
-	Applying correlation filter
-	Removing outliers from signal amplitude
-	Exporting data into NetCDF format

### data_analysis.ipynb: Notebook containing data analysis of the processed data.
  #### Summary:
-	Filtering of tides
- Removing surface interference and values above surface
-	Filling in gaps
-	Averaging velocity with depth and comparison with temperature

### quality_flagging.ipynb: Notebook containing quality flagging of data with minimal processing.
  #### Numeric flags are added according to:
- Passing all tests
- Unkown
- Above surface
- Contaminated from surface interference
- Below correlation threshold 64
- Signal amplitude outlier
- Missing data