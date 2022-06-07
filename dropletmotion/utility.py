"""
This module contains general utility functions used throughout the package

Functions
---------
extract_info
    Create dictionary of sample and video info from file name
experimental_t_int
    Calculate experimental time interval between droplets
r_adj_calc
    Calculate adjusted R squared
"""

# Import required packages
import re
import pandas as pd
import numpy as np


def extract_info(
    file_name, ref_mm=25.0, ref_px_err=5, ref_mm_err=0.05, video_info_skip=False
) -> dict:
    """
    Create dictionary of sample and video info from file name.

    Parameters
    ----------
    file_name : str
        Name of the file to be analysed
    ref_mm : float
        Lenght (in mm) of the reference object used for calculation of video
        scale (default=25.0)
    ref_px_err : int
        Uncertainty on the length (in pixels) of the reference object used for
        calculation of video scale (default=5)
    ref_mm_err : float
        Uncertainty on the length (in mm) of the reference object used for
        calculation of video scale (default=0.05)
    video_info_skip : bool
        Flag to skip requirement of angle and fps calculation. They are
        fundamental information for video analysis, but might not be necessary
        for other purposes, hence they might not be present in filename
        (default=False)

    Returns
    -------
    sample_info : dict
        Dictionary with extracted information
    """

    sample_info = {'name': file_name.split(' ')[0]}

    def _get_info(prop, search_key, num_type, alternative=None, required=False):
        """
        Subfunction to extract properties and assign default values if
        name does not match correct patterns. Raise AttributeError if
        alternative is None.
        """
        try:
            match = (
                re.findall(
                    rf'\d+\.\d+{search_key}|\d+{search_key}|{search_key}\d+',
                    file_name
                )[0]
                .replace(search_key, '')
            )
            sample_info[prop] = int(match) if num_type == 'int' else float(match)
        except IndexError as missing_prop:
            if alternative is not None:
                sample_info[prop] = alternative
            elif required:
                raise AttributeError(
                    f'Required property {prop} not in file name'
                ) from missing_prop


    name_split = sample_info['name'].split('_')

    if 'R' in name_split[0]:
        sample_info['structure'] = 'rods'
    elif 'S' in name_split[0]:
        sample_info['structure'] = 'SNFs'

    sample_info['viscosity'] = name_split[1]
    sample_info['sample_ID'] = name_split[2]

    _get_info('angle', 'deg', 'float', required=not video_info_skip)
    _get_info('t_int', 'secint', 'int')
    _get_info('spot_ID', 'spot', 'int', alternative=1)
    _get_info('seq_ID', 'seq', 'int', alternative=1)
    _get_info('volume', 'uL', 'int', alternative=10)

    _get_info('fps', 'fps', 'float')
    # If no fps specified in filename, attempt its calculation from
    # total time and number of frames, if specified in filename.
    if 'fps' not in sample_info and not video_info_skip:
        try:
            msec = int(re.findall(r'\d+ms', file_name)[0].replace('ms', ''))
            frames = int(re.findall(r'\d+fr', file_name)[0].replace('fr', ''))
            sample_info['fps'] = frames / msec * 1000
        except IndexError as missing_fps:
            raise AttributeError(
                "Framerate missing from filename"
            ) from missing_fps

    # Set video scale and its uncertainty
    try:
        ref_px = int(re.findall(r'\d+px', file_name)[0].replace('px', ''))
        sample_info['scale'] = ref_mm / ref_px
        sample_info['scale_err_rel'] = (
            (ref_mm_err / ref_mm) ** 2 +
            (ref_px_err / ref_px) ** 2) ** 0.5
    except IndexError as missing_px:
        raise AttributeError(
            "Reference size in pixel missing from filename"
        ) from missing_px

    return sample_info


def experimental_t_int(x_df) -> pd.DataFrame:
    """
    Calculate experimental time interval between droplets of a given
    measurement

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe as output by core.DropletTrack.motion_save. The dataframe
        must contain a single experiment, otherwise the function would yield
        incorrect values.

    Returns
    -------
    x_out : pd.DataFrame
        Dataframe with the experimental time interval between droplets.

    Raises
    ------
    ValueError
        If x_df does not refer to a unique experiment
    """

    # Check if x_df refers to a single experiment
    if 'name' in x_df:
        for label in ['name', 'angle', 'spot_ID', 'seq_ID']:
            if not x_df[label].is_unique:
                raise ValueError(
                    "Non-unique data labels. Operation is being likely " +
                    "attempted on multiple experiments.\nGroup values " +
                    "approrpiately so that only one experiment is considered"
                )

    x_out = (
        x_df
        .groupby('droplet')
        .t
        .apply(lambda x: x.iloc[0])
        .diff()
        .reset_index()
    )

    x_out.rename({'t': 't_int_exp'}, axis=1, inplace=True)

    return x_out


def r_adj_calc(x, y, model, beta, free_params) -> float:
    """
    Calculate the adjusted R squared of a given model on a dataset, according
    to the Wherry-1 formula in Yin and Fan, J Exp Educ, 69, 2, 2001.

    Parameters
    ----------
    x : np.array
        Independent variable of the model
    y : np.array
        Measured values of the the dependent variable
    model : func
        Function of the model to evaluate
    beta : sequence
        Values of the parameters to be passed to model
    free_params : int
        Number of the free parameters, which might be different
        from the number of items in beta

    Returns
    -------
    r_adj : float
        Adjusted R squared of the regression
    """

    # Calculated R squared
    r_squared = 1 - np.var(y - model(beta, x)) / np.var(y)

    # Perfomed adjusted R squared correction
    n_points = len(y)
    r_adj = 1 - (1 - r_squared) * (n_points - 1) / (n_points - free_params - 1)

    return r_adj
