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


def remove_outliers(ds, bottom_threshold, surface_threshold):
     """
     Remove signal outliers based on linear decay of signal. If less than 3 beams are valid, velocity filtered set to NaN.

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
          - 'vel_filt' : velocities set to NaN where fewer than 3 beams are valid.
     """
     n_bins = ds.sizes['range']
     threshold = np.linspace(bottom_threshold, surface_threshold, n_bins)
     threshold_da = xr.DataArray(threshold, dims=['range'], coords={'range': ds.range})

     ds_cleaned = ds.copy()

     mask_per_beam = ds['amp'] < threshold_da
     beam_mask = mask_per_beam.sum(dim='beam') >= 3 # keep only if 3 or more beams valid

     ds_cleaned['amp'] = ds['amp'].where(mask_per_beam)

     if 'vel_filt' in ds:
           ds_cleaned['vel_filt'] = ds['vel_filt'].where(beam_mask)

     return ds_cleaned


def interp_profile(vel, coord, coord_new):
    """Interpolate a single velocity profile from time varying coordinate to constant coordinate."""
    f = interp1d(coord, vel, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(coord_new)

def correlation_filter(ds, thresh=50, inplace=False):
    """
    Function adapted from mhkit.dolfyn.adp.api.clean to remove vel_filt instead of vel values based on correlation.

    Filters out data where correlation is below a threshold in the
    along-beam correlation data.

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean.
    thresh : numeric
      The maximum value of correlation to screen, in counts or %.
      Default = 50
    inplace : bool
      When True the existing data object is modified. When False
      a copy is returned. Default = False

    Returns
    -------
    ds : xarray.Dataset
      Elements in velocity filtered, correlation, and amplitude are removed if below the
      correlation threshold

    Notes
    -----
    Does not edit correlation or amplitude data.
    """

    if not inplace:
        ds = ds.copy(deep=True)

    # 4 or 5 beam
    tag = []
    if hasattr(ds, "vel_filt"):
        tag += [""]
    if hasattr(ds, "vel_b5"):
        tag += ["_b5"]
    if hasattr(ds, "vel_avg"):
        tag += ["_avg"]

    # copy original ref frame
    coord_sys_orig = ds.coord_sys

    # correlation is always in beam coordinates
    dolfyn.rotate2(ds, "beam", inplace=True)
    # correlation is always in beam coordinates
    for tg in tag:
        mask = ds["corr" + tg].values <= thresh

        for var in ["vel_filt", "corr", "amp"]:
            try:
                ds[var + tg].values[mask] = np.nan
            except:
                ds[var + tg].values[mask] = 0
            ds[var + tg].attrs["Comments"] = (
                "Filtered of data with a correlation value below "
                + str(thresh)
                + ds["corr" + tg].units
            )

    dolfyn.rotate2(ds, coord_sys_orig, inplace=True)

    if not inplace:
        return ds


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
        ds: xarray Dataset containing 'vel_qc_primary'
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
    ds['vel_qc_primary'][direction].plot(
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

def plot_qc_secondary(ds, direction=0, num_flags=7):
    """
    Plot secondary QC flags.
    
    Parameters:
        ds: xarray Dataset containing 'vel_qc_secondary'
        direction: int, index for direction (0=E, 1=N, 2=U1, 3=U2)
        num_flags: int, number of QC flags (7 or 8)
    """
    dir_labels = {0: 'E', 1: 'N', 2: 'U1', 3: 'U2'}
    
    if num_flags == 7:
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
    elif num_flags == 8:
        colors = [
            '#228B22',  # 1: Passed     
            '#808080',  # 2: Unknown     
            '#F0E442',  # 3: Above surface     
            '#E69F00',  # 4: Surface interference     
            '#56B4E9',  # 5: Below correlation     
            '#CC79A7',  # 6: Amplitude outliers     
            '#0072B2',  # 7: Interpolated data
            '#000000',  # 8: Missing     
        ]
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        ticks = [1, 2, 3, 4, 5, 6, 7, 8]
        labels = [
            'Passed all tests',
            'Unknown', 
            'Above surface',
            'Surface interference',
            'Below correlation threshold',
            'Signal amplitude outliers',
            'Interpolated data',
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