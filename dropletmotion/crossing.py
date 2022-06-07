'''
This module contains functions used the analysis of scaling experiments

Functions
---------
center_peak
    Center main detected crossing peak
peak_spacing
    Calculate spacing and position of accelaration peaks
crossing_point
    Calculate trace and probe crossing point from reference image
extract_acceleration
    Calculate velocity and acceleration and save it to csv file
'''

# Import packages
import pathlib
import numpy as np
import pandas as pd
import cv2
from scipy import signal
from dropletmotion import velocity
from dropletmotion import utility
from dropletmotion import core


__all__ = [
    'center_peak', 'peak_spacing', 'crossing_point', 'extract_acceleration'
]


def center_peak(data, init_region=1.5, refine_region=5):
    """
    Center velocity peaks to allow for easier comparison and analysis

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe with velocity and position data
    init_region : float
        Region (in mm) around crossing point to which initial
        maximum estimate is restricted (default = 1.5)
    refine_region : float
        Region (in mm) around first max estimate to which
        refined peak finding is restricted (default = 5)

    Returns
    -------
    x_zeroed : pd.Series
        Droplet position with peak center as zero
    """

    # Select first probe droplet
    data_first = data.query("droplet==1 & seq_ID==3")

    # Restrict to region around crossing and provide first estimate
    data_init = data_first.loc[
        (data_first.x_cent>-init_region) & (data_first.x_cent<init_region)
    ]

    # Find maximum
    max_init = data_init['v'].idxmax()
    x_init = data_init.loc[max_init, 'x_cent']

    # Restrict to region around detected maximum to refine peak search
    refine_bounds = [x_init - refine_region, x_init + refine_region]
    data_refine = data_first.loc[
        (data_first.x_cent > refine_bounds[0]) &
        (data_first.x_cent < refine_bounds[1])
    ]

    # Supply very low minimum peak height and width.
    # This is necessary to have quantities available in output
    peaks = signal.find_peaks(data_refine['v'], height=0.01, width=20)

    # Find tallest peak
    max_peak = np.argmax(peaks[1]['peak_heights'])

    # Calculate center
    max_bounds = (
        data_refine.iloc[int(peaks[1]['left_ips'][max_peak])]['x_cent'],
        data_refine.iloc[int(peaks[1]['right_ips'][max_peak])]['x_cent'])
    max_x = sum(max_bounds) / 2

    # Calculate new centered x
    x_zeroed = data['x_cent'] - max_x

    return x_zeroed


def peak_spacing(data, roi=3):
    """
    Calculate spacing and position of accelaration peaks

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe with acceleration and position data
    roi : float
        Region (in mm) on each side of detected center of main peak
        to which analysis is limited (default=3)

    Returns
    -------
    series_out : pd.Series
        Pandas series with:
        - 'spacing' : peak spacing
        - 'max_x' : position of maximum
        - 'min_x' : position of minimum
        - 'max_a' : acceleration at maximum
        - 'min_a' : acceleration at minimum
        - 'max_x_adv' : position of advancing front at maximum
        - 'min_x_adv' : position of advancing front at mimimum
        - 'max_x_rec' : position of receding front at maximum
        - 'min_x_rec' : position of receding front at mimimum
    """

    # Restrict analysis to region around peak
    data = data.loc[(data.x_zeroed>-roi) & (data.x_zeroed<roi)]

    # Divide dataframe in positive and negative region
    data_neg = data.loc[data.x_zeroed <= 0]
    data_pos = data.loc[data.x_zeroed >= 0]

    # Find maxima in negative region and minima in negative region
    maxima = signal.find_peaks(data_neg.a, prominence=0, height=0)
    minima = signal.find_peaks(-data_pos.a, prominence=0, height=-0)

    # Find global maximum and global minimum in region of interest
    # In negative region, correct for very similar maxima
    num_max = len(maxima[0])
    if num_max > 1:
        max_df = pd.DataFrame(maxima[1]).sort_values(
            'peak_heights', ascending=False)
        # If maxima sufficiently close in height take leftmost
        if max_df.iloc[1].peak_heights / max_df.iloc[0].peak_heights > 0.9:
            max_ix = min(maxima[0][max_df.index[:2]])
        else:
            max_ix = maxima[0][max_df.index[0]]
    else:
        max_ix = maxima[0][np.argmax(maxima[1]['peak_heights'])]

    min_ix = minima[0][np.argmax(minima[1]['peak_heights'])]

    # Find their position relative to the crossing point
    results = {}

    results['max_x'] = data_neg['x_cent'].iloc[max_ix]
    results['min_x'] = data_pos['x_cent'].iloc[min_ix]

    results['max_x_adv'] = data_neg['x_cent_adv'].iloc[max_ix]
    results['min_x_adv'] = data_pos['x_cent_adv'].iloc[min_ix]

    results['max_x_rec'] = data_neg['x_cent_rec'].iloc[max_ix]
    results['min_x_rec'] = data_pos['x_cent_rec'].iloc[min_ix]

    results['max_a'] = data_neg['a'].iloc[max_ix]
    results['min_a'] = data_pos['a'].iloc[min_ix]

    # Calculate spacing between detected peaks
    results['spacing'] = results['min_x'] - results['max_x']

    series_out = pd.Series(results)

    return series_out


def crossing_point(file_path) -> pd.DataFrame:
    '''
    Calculate probe and trace crossing point in crossing experiments
    from reference images; also calculate trace base diameter, unless
    otherwise specified by an 'ignore_diameter' in the file name.

    Parameters
    ----------
    file_path : str
        Path to the reference image. Sample information and processing
        flags are inferred from file name

    Returns
    -------
    results : pd.DataFrame
        Dataframe with experiment identifiers, and extracted crossing
        point and trace base diameter
    '''

    # Extract sample information and calculate scale
    file_name = pathlib.Path(file_path).stem
    sample_info = utility.extract_info(file_name, video_info_skip=True)

    # Check for data processing flags
    ignore_diameter = bool('ignore_diameter' in file_name)

    # Read image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    _, image_th = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    cont, _ = cv2.findContours(
        image_th,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # Keep only the largest contours, i.e. drop
    cont_drop = sorted(cont, key=cv2.contourArea, reverse=True)[0]

    # Calculate crossing point from droplet centroid
    moments = cv2.moments(cont_drop)
    x_cross = (moments["m10"] / moments["m00"]) * sample_info['scale']

    # Calculate droplet base diameter
    if not ignore_diameter:
        x_left, x_right, _ = core.drop_profile(cont_drop, sample_info['scale'])
        base_diameter = x_right - x_left
    else:
        base_diameter = np.nan

    # Build results dataframe
    results = pd.DataFrame({
        'name': sample_info['name'],
        'spot_ID': sample_info['spot_ID'],
        'x_cross': [x_cross],
        'base_diameter': [base_diameter],
        'vol_trace': sample_info['volume']
    })

    return results


def extract_acceleration(file_path, reg_param=1e-1) -> None:
    """
    Calculate velocity and acceleration of crossing experiments from
    a given csv file (output of dropletmotion.core.DropletTrack.motion_save)
    and save it to a csv file located in the same folder.

    Parameters
    ----------
    file_path : str
        Path to position vs time csv file
    reg_param : float
        Regularization parameter to be passed to
        dropletmotion.velocity.regularized_deritvative (default=1e-1)

    Returns
    -------
    None
    """

    file_name = pathlib.Path(file_path).stem

    # Read csv file
    data = pd.read_csv(file_path)

    try:
        # Assign fps for velocity calculation
        fps = utility.extract_info(file_name)['fps']

        # Calculate velocity
        data['v'] = (
            data.groupby('droplet')
            .x.transform(
                velocity.regularized_derivative,
                fps=fps, reg_param=reg_param
            )
        )

        # Calculate acceleration
        data['a'] = (
            data.groupby('droplet')
            .v.transform(
                velocity.regularized_derivative,
                fps=fps, reg_param=reg_param
            )
        )

        # Assign experimental time interval, that cannot
        # be reconstructed from data_out
        data_out = data.merge(utility.experimental_t_int(data))

        # Save csv file
        data_out.to_csv(
            file_path.replace('position', 'acceleration'),
            index=False
        )

    except (KeyError, AttributeError):
        print(f'{file_name} not approriately formatted')
