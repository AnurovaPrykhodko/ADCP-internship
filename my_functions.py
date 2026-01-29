import numpy as np
import xarray as xr

def print_data_removed(ds_before, ds_after):
    """
    Prints velocity data removed and the percentage. 
    """
    before = np.isfinite(ds_before.vel.values).sum()  
    after = np.isfinite(ds_after.vel.values).sum()
    removed = before - after
    percent_removed = (removed / before) * 100

    return print(f"Data removed: {removed} ({percent_removed:.2f}%)")


def remove_outliers(ds, bottom_threshold, surface_threshold):
     """
     Remove signal outliers based on linear decay of signal. If less than 3 beams are valid, velocity set to NaN.

     Parameters
     ----------
     ds : xarray.Dataset
     bottom_threshold : float
          Amplitude threshold at the bottom (first bin). Bins with amplitude above
          this value are considered outliers.
     surface_threshold : float
          Amplitude threshold at the surface (last bin). Threshold is linearly
          interpolated between bottom and surface.

     Returns:
     -------
     ds_cleaned : xarray.Dataset
          Copy of the input dataset with:
          - 'amp' : beam amplitudes above threshold set to NaN
          - 'vel' : velocities set to NaN where fewer than 3 beams are valid.
     """
     n_bins = ds.sizes['range']
     threshold = np.linspace(bottom_threshold, surface_threshold, n_bins)
     threshold_da = xr.DataArray(threshold, dims=['range'], coords={'range': ds.range})

     ds_cleaned = ds.copy()

     mask_per_beam = ds['amp'] < threshold_da
     beam_mask = mask_per_beam.sum(dim='beam') >= 3 # keep only if 3 or more beams valid

     ds_cleaned['amp'] = ds['amp'].where(mask_per_beam)

     if 'vel' in ds:
           ds_cleaned['vel'] = ds['vel'].where(beam_mask)

     return ds_cleaned