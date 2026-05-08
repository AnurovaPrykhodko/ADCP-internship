# ADCP-internship (ongoing):

This repository contains processing for Acoustic Doppler Current Profiler (ADCP) data, analysis of coastal circulation and quality flagging according to the IODE Quality Flag Standard. 
The data was collected on the coastal shelf of Armona, Algarve, southern Portugal, during May - November 2024. The ADCP was TRDI WorkHorse Sentinel 600kHz, and upward-looking.

Implements Python 3.12, DOLfYN, Utide, Matplotlib, NumPy, Xarray and SciPy.

## List of content:

### my_functions.py: Custom functions.

### raw_data_processing.ipynb: Notebook containing minimal processing of the raw data.
  #### Summary:
- Removing data before deployment
-	Adding transducers height
-	Exporting data into NetCDF format

### data_analysis.ipynb: Notebook containing data analysis of the flagged data.
  #### Summary:
-	Removing bad data
- tidal reconstruction with harmonic analysis
- computing the residual current
- rotating to alongshore and crossshore directions based on principal heading
- avereging velocity with depth and different layers
- comparing alongshore velocity with temperature
- exporting to csv

### quality_flagging.ipynb: Notebook containing quality flagging of data with minimal processing.
  #### The numeric flagging system includes a primary flag indicating the quality and a secondary flag with a description.