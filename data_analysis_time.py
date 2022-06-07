'''
Analysis of time interval experiments, to extract aggregated data.
'''

# Import packages
import pathlib
import glob
import numpy as np
import pandas as pd
import dropletmotion as dm


# Load data and build database
data_raw = []
for file in glob.glob("Data/**/*exp_T*constant_velocity.csv", recursive=True):
    if 'Excluded' not in file:
        data_read = pd.read_csv(file)
        sample_info = dm.utility.extract_info(pathlib.Path(file).stem)
        data_raw.append(data_read.assign(**sample_info))

data_raw = pd.concat(data_raw, ignore_index=True).dropna(subset=['v'])

# Remove droplets after 8, if present in some dataset, to allow for comparison
# of droplets present in ALL datasets
data_raw = data_raw.query("droplet<=8").copy()

# Normalize droplet velocity
data_raw['v_norm'] = (
    data_raw.groupby(['name', 't_int', 'spot_ID'], group_keys=False)
    .apply(lambda df: df.query('droplet>=4').v.mean() / df.v)
)

# Aggregate first droplet data
data_agg = pd.pivot_table(
    data_raw.query("droplet==1"), values='v_norm',
    index=['structure', 't_int'],
    aggfunc=[np.mean, lambda x: np.std(x, ddof=1)]
)
data_agg.columns = ['v_norm', 'v_norm_err']
data_agg = data_agg.reset_index()

# Save aggregated data
data_agg.to_csv("Results/time_interval_aggregated.csv", index=False)
print('Time interval analysis complete')
