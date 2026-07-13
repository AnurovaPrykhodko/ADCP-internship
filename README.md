# ADCP-internship:

This repository contains quality flagging, processing and analysis of Acoustic Doppler Current Profiler (ADCP) data. The data was collected on the coastal shelf of Armona, Algarve, southern Portugal, during May - November 2024 with a TRDI WorkHorse Sentinel 600kHz.

Implements Python 3.12, DOLfYN, Utide, Matplotlib, NumPy, Xarray and SciPy.

## List of content:

### raw_data_processing.ipynb: Notebook containing minimal processing of the raw data.
- removing data before deployment
-	adding transducers height
-	exporting data into NetCDF format

### quality_flagging.ipynb: Notebook containing quality flagging of data with minimal processing.
 - The numeric flagging system includes a primary flag indicating the quality and a secondary flag with a description.

   <img width="1113" height="656" alt="scheme" src="https://github.com/user-attachments/assets/8c036fd0-d0db-4eaf-bd21-2fd211bc394f" />

  
### data_analysis.ipynb: Notebook containing data analysis of the flagged data.

-	removing data flagged as Bad 
- tidal reconstruction with harmonic analysis
- computing the residual current
- rotating to alongshore and crossshore directions based on principal heading
- avereging velocity with depth and different layers
- comparing alongshore velocity with temperature
- exporting to csv

### my_functions.py: Custom functions.
