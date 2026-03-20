from mhkit import dolfyn
from mhkit.dolfyn.adp import api

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d

def print_data_removed(ds_before, ds_after, variable):
    """
    Prints variable data removed and the percentage. 
    """
    before = np.isfinite(ds_before[variable].values).sum()  
    after = np.isfinite(ds_after[variable].values).sum()
    removed = before - after
    percent_removed = (removed / before) * 100

    print(f"Data removed for {variable}: {removed} ({percent_removed:.2f}%)")


def detect_outliers(ds, bottom_threshold, surface_threshold, set_to_NaN=True):
     """
     Detect signal outliers based on linear decay of signal. Signal counts as outlier if less than 3 beams are valid.
     
     Parameters
     ----------
     ds : xarray.Dataset
     bottom_threshold : float
          Amplitude threshold at the bottom (first bin). Bins with amplitude above
          this value are considered outliers.
     surface_threshold : float
          Amplitude threshold at the surface (last bin). Threshold is linearly
          interpolated between bottom and surface.
     set_to_NaN : Boolean
          Adjust to return copy of dataset with velocity set to NaN according to outliers (True)
          or return mask with true for valid values (False)

     Returns:
     -------
     ds_cleaned : xarray.Dataset
          Copy of the input dataset with:
          - 'amp' : beam amplitudes above threshold set to NaN
          - 'vel' : velocities set to NaN where fewer than 3 beams are valid.
     or 
     beam_mask : xarray.Dataset
          Mask with True for valid values.
     """
     n_bins = ds.sizes['range']
     threshold = np.linspace(bottom_threshold, surface_threshold, n_bins)
     threshold_da = xr.DataArray(threshold, dims=['range'], coords={'range': ds.range})

     ds_cleaned = ds.copy()

     mask_per_beam = ds['amp'] < threshold_da
     beam_mask = mask_per_beam.sum(dim='beam') >= 3 # keep only if 3 or more beams valid

     ds_cleaned['amp'] = ds['amp'].where(mask_per_beam)

     if 'vel' in ds and set_to_NaN == True:
           ds_cleaned['vel'] = ds['vel'].where(beam_mask)
           return ds_cleaned
     elif set_to_NaN == False:
           return beam_mask
     else:
           print('dataset does not contain variable vel')


def interp_profile(vel, coord, coord_new):
    """Interpolate a single velocity profile from time varying coordinate to constant coordinate."""
    f = interp1d(coord, vel, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(coord_new)


def apply_qc_mask(mask, true_values, qc1, qc2):
    """
    Apply a mask to update primary and secondary quality flag values.
    
    Parameters:
        mask: xarray DataArray boolean mask
        true_values: tuple (qc1_value, qc2_value) where mask is True
        qc1: existing qc1 array (kept where mask is False)
        qc2: existing qc2 array (kept where mask is False)
    
    Returns:
        qc1, qc2: xarray DataArrays with updated values
    """
    qc1 = xr.where(mask, true_values[0], qc1)
    qc2 = xr.where(mask, true_values[1], qc2)
    
    return qc1, qc2

def plot_masked(mask, array, mask_name):
    """
    Plot masked array values.
    
    Parameters:
        mask: xarray DataArray boolean mask
        array: xarray DataArray to plot
        mask_name: str, description of the mask for the title
    """
    array[0].where(mask).plot(x='time')
    plt.title(f'Values flagged as {mask_name} (dir=E)')
    plt.show()

def plot_qc_primary(ds, direction=0):
    """
    Plot primary QC flags for chosen direction.
    
    Parameters:
        ds: xarray Dataset containing 'vel_qc_primary' or 'vel_filt_qc_primary'
        direction: int, index for direction (0=E, 1=N, 2=U1, 3=U2)
    """
    dir_labels = {0: 'E', 1: 'N', 2: 'U1', 3: 'U2'}
    
    # Define discrete colormap for QC values
    colors = [
        '#228B22',  # 1: Good   
        '#808080',  # 2: Unknown   
        '#E69F00',  # 3: Questionable   
        '#D55E00',  # 4: Bad   
        '#000000',  # 9: Missing   
    ]
    cmap = ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 9.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot with discrete colors
    if 'vel_qc_primary' in ds:
        ds['vel_qc_primary'][direction].plot(
            cmap=cmap, 
            norm=norm,
            cbar_kwargs={
                'ticks': [1, 2, 3, 4, 9],
                'label': 'QC Flag'
            }
        )
    else:
        ds['vel_filt_qc_primary'][direction].plot(
            cmap=cmap, 
            norm=norm,
            cbar_kwargs={
                'ticks': [1, 2, 3, 4, 9],
                'label': 'QC Flag'
            }
        )

    
    # Rename colorbar tick labels
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.set_yticklabels(['Good', 'Unknown', 'Questionable', 'Bad', 'Missing'])
    
    plt.title(f'Primary QC Flags (dir={dir_labels.get(direction, direction)})')
    plt.show()

def plot_qc_secondary(ds, direction=0):
    """
    Plot secondary QC flags.
    
    Parameters:
        ds: xarray Dataset containing 'vel_qc_secondary' or 'vel_filt_qc_secondary'
        direction: int, index for direction (0=E, 1=N, 2=U1, 3=U2)
        num_flags: int, number of QC flags (7 or 8)
    """
    dir_labels = {0: 'E', 1: 'N', 2: 'U1', 3: 'U2'}
    
    colors = [
        '#228B22',  # 1: Passed     
        '#808080',  # 2: Unknown     
        '#F0E442',  # 3: Above surface     
        '#E69F00',  # 4: Surface interference     
        '#56B4E9',  # 5: Below correlation     
        '#CC79A7',  # 6: Amplitude outliers     
        '#000000',  # 7: Missing     
    ]
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    ticks = [1, 2, 3, 4, 5, 6, 7]
    labels = [
        'Passed all tests',
        'Unknown', 
        'Above surface',
        'Surface interference',
        'Below correlation threshold',
        'Signal amplitude outliers',
        'Missing data'
    ]
    
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot with discrete colors
    ds['vel_qc_secondary'][direction].plot(
        cmap=cmap, 
        norm=norm,
        cbar_kwargs={
            'ticks': ticks,
            'label': 'QC Flag'
        }
    )
    
    # Rename colorbar tick labels
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.set_yticklabels(labels)
    
    plt.title(f'Secondary QC Flags (dir={dir_labels.get(direction, direction)})')
    plt.show()


def create_masks(ds, corr_threshold=64, bottom_threshold=225, surface_threshold=100):
     """
     Create quality control masks for ADCP data.
          
     Parameters
     ----------
     ds : xarray.Dataset
         Dataset containing 'range', 'depth', 'corr', and 'vel' variables.
         corr_threshold : float, optional
         Correlation threshold. Default is 100.
         bottom_threshold : float, optional
            Amplitude threshold at bottom for outlier detection. Default is 225.
         surface_threshold : float, optional
            Amplitude threshold at surface for outlier detection. Default is 100.
        
     Returns
     -------
     dict
        Dictionary containing all masks:
        - 'above_surface': True where range > depth
        - 'surface_interference': True in upper 15% of water column
        - 'below_corr_thresh': True where correlation below threshold
        - 'outlier': True where amplitude outliers detected
        - 'NaN': True where velocity is NaN
        - 'passed': True where all quality checks pass
     """

     # values above surface, True where range > depth
     mask_above_surface = ds['range'] > ds['depth']   
 
     # values contaminated from surface interference (upper 15%)
     mask_surface_interference = (ds['range']/ds['depth'] >= 0.85) & (ds['range']/ds['depth'] <= 1)

     # values below correlation threshold
     mask_below_corr_thresh = (
        (ds['corr'] <= corr_threshold).any(dim='beam')
        & ~mask_above_surface
        & ~mask_surface_interference
        )

     # amplitude outliers
     mask_outlier = (
        ~detect_outliers(ds, bottom_threshold, surface_threshold, set_to_NaN=False)
        & ~mask_above_surface
        & ~mask_surface_interference
        & ~mask_below_corr_thresh
        )

     # NaN values
     mask_NaN = np.isnan(ds['vel'])

     # passed all checks
     mask_passed = (
        ~mask_above_surface
        & ~mask_surface_interference
        & ~mask_below_corr_thresh
        & ~mask_outlier
        & ~mask_NaN
        )
        
     return {
        'above_surface': mask_above_surface,
        'surface_interference': mask_surface_interference,
        'below_corr_thresh': mask_below_corr_thresh,
        'outlier': mask_outlier,
        'NaN': mask_NaN,
        'passed': mask_passed
        }