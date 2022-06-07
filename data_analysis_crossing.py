'''
Analysis of crossing experiments, to extract aggregated data about crossing
peaks.
'''

# Import packages
import pathlib
import glob
import numpy as np
import pandas as pd
import dropletmotion as dm

def build_cross_data(data_relative_path='') -> pd.DataFrame:
    """
    Build raw data database for crossing analysis.

    Parameters
    ----------
    data_relative_path : str
        Relative path to the Data folder from the current working directory.

    Returns
    -------
    data : pd.DataFrame
        Database with raw data used for crossing analysis.
    """

    glob_path = data_relative_path + "Data/**/*exp_C*acceleration.csv"

    # Load data and build database
    data_raw = []
    for file in glob.glob(glob_path, recursive=True):
        if 'Excluded' not in file:
            data_read = pd.read_csv(file)
            sample_info = dm.utility.extract_info(pathlib.Path(file).stem)
            data_raw.append(data_read.assign(**sample_info))

    data_raw = pd.concat(data_raw, ignore_index=True).dropna(subset=['v'])
    data_raw.rename(columns={'volume': 'vol_probe'}, inplace=True)

    # Calculate average probe base diameter
    data_raw['drop_diam'] = data_raw.x_adv - data_raw.x_rec
    data_raw['probe_diam'] = (
        data_raw
        .groupby('vol_probe')
        .drop_diam
        .transform(np.mean)
    )
    data_raw['probe_diam_err'] = (
        data_raw
        .groupby('vol_probe')
        .drop_diam
        .transform(np.std, ddof=1)
    )

    # Read offset database and assign values to data
    offset = pd.read_csv(
        data_relative_path + 'Data/crossing_offset_database.csv'
    )
    offset['trace_diam'] = (
        offset
        .groupby('vol_trace')
        .base_diameter
        .transform(np.mean)
    )
    offset['trace_diam_err'] = (
        offset
        .groupby('vol_trace')
        .base_diameter
        .transform(np.std, ddof=1)
    )
    data = data_raw.merge(offset)

    # Scale position to crossing region
    data['x_cent'] = data.x - data.x_cross
    data['x_cent_adv'] = data.x_adv - data.x_cross
    data['x_cent_rec'] = data.x_rec - data.x_cross

    # Calculate position zeroed on center of crossing peak
    data['x_zeroed'] = (
        data
        .groupby(['name', 'spot_ID'], as_index=False, group_keys=False)
        .apply(dm.crossing.center_peak)
    )

    return data


if __name__ == "__main__":

    data = build_cross_data()

    # Analyse main peak
    grouping = [
        'structure', 'vol_trace', 'vol_probe',
        'probe_diam', 'probe_diam_err', 'trace_diam', 'trace_diam_err'
    ]

    # Limit analysis to first probe droplet
    data_probe = data.query("seq_ID==3 & droplet==1")
    peaks = (
        data_probe
        .groupby(['name', 'spot_ID'] + grouping)
        .apply(dm.crossing.peak_spacing, roi=3)
        .reset_index()
    )

    agg_names = ['spacing', 'max_x_adv', 'min_x_adv', 'max_x_rec', 'min_x_rec']
    col_names = sorted([i.replace('_x', '') for i in agg_names])
    col_names += [i + '_err' for i in col_names]

    peaks_agg = pd.pivot_table(
        peaks,
        values=agg_names,
        index=grouping,
        aggfunc=[np.mean, lambda x: np.std(x, ddof=1)]
    )

    peaks_agg.columns = col_names
    peaks_agg = peaks_agg.reset_index()

    # Save output
    peaks_agg.to_csv("Results/crossing_peaks.csv", index=False)
    print('Crossing analysis complete')
