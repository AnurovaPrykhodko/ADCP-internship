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
    """
    dir_labels = {0: 'E', 1: 'N', 2: 'U1', 3: 'U2'}
    
    colors = [
        '#228B22',  # 1: Passed
        '#808080',  # 2: Unknown
        '#F0E442',  # 3: Above surface
        '#E69F00',  # 4: Surface interference
        '#56B4E9',  # 5: Below correlation
        '#CC79A7',  # 6: Amplitude outliers
        '#9467BD',  # 7: Compass heading deviation
        '#D62728',  # 8: Pressure failure
        '#17BECF',  # 9: Temperature sensor
        '#BCBD22',  # 10: Time
        '#000000',  # 11: Missing
    ]
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    labels = [
        'Passed all tests',
        'Unknown',
        'Above surface',
        'Surface interference',
        'Below correlation threshold',
        'Signal amplitude outliers',
        'Compass heading deviation',
        'Pressure failure',
        'Unrealistic temperature',
        'Clock drift',
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

def detect_const_pressure(ds, pressure_diff_thresh=0.001):
    """
    Detect three consecutive constant pressure values in a dataset. Constant is defined as a 
    value +- pressure_diff_thresh to allow for rounded values of the last digit.
    
    Parameters:
        ds: xarray Dataset containing 'pressure' variable
        pressure_diff_thresh: float, threshold for detecting constant values (default 0.001)
    
    Returns:
        mask_pressure_constant: xarray DataArray, True where pressure is constant
    """
    # Calculate difference between consecutive values
    pressure_diff = np.abs(ds['pressure'] - ds['pressure'].shift(time=1)).fillna(0)
    
    # Check where difference is below threshold
    pressure_constant = pressure_diff <= pressure_diff_thresh
    
    # Find where we have more than 3 consecutive constant values
    consecutive_constant = pressure_constant.rolling(time=3, min_periods=3).sum() >= 3
    
    # Shift back to flag all points of constant runs
    mask_pressure_constant = (
        consecutive_constant | 
        consecutive_constant.shift(time=-1, fill_value=False) | 
        consecutive_constant.shift(time=-2, fill_value=False)
    )
    
    return mask_pressure_constant

def summarize_qc(ds, qc_var):
    qc = ds[qc_var]
    
    # Read attributes
    flag_values = qc.attrs.get('flag_values', [])
    flag_meanings = qc.attrs.get('flag_meanings', '').split()
    
    # Build mapping
    mapping = dict(zip(flag_values, flag_meanings))
    
    # Count occurrences
    counts = qc.to_series().value_counts()
    total_count = counts.sum()
    
    # Print summary
    print(f"QC Summary for '{qc_var}':")
    for flag in flag_values:
        count = (qc == flag).sum().item()
        label = mapping.get(flag, "unknown")
        pct = (count / total_count) * 100
        print(f"{label}: {count} ({pct:.2f}%)")
    
    print("\n")