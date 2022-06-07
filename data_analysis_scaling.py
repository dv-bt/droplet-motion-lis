'''
Analysis of scaling experiments, to extract aggregated data and fit
scaling laws
'''

# Import packages
import pathlib
import glob
import numpy as np
import pandas as pd
import dropletmotion as dm


# Define aggregation function
def aggregate_data(data):
    ''' Aggregate selected dataset '''

    # Define aggregation keys
    index_agg = [
        'structure', 'viscosity',
        'angle', 'sin', 'sin_err', 'force', 'force_err',
    ]
    values_agg = ['Ca', 'friction', 'v']
    columns_agg = values_agg + [i + '_err' for i in values_agg]

    # Perform aggregation
    data_out = pd.pivot_table(
        data, values=values_agg, index=index_agg,
        aggfunc=[np.mean, lambda x: np.std(x, ddof=1)]
    )
    data_out.columns = columns_agg
    data_out = data_out.reset_index()

    return data_out


# Load data and build database
data_raw = []
for file in glob.glob("Data/**/*exp_S*constant_velocity.csv", recursive=True):
    if 'Excluded' not in file:
        data_read = pd.read_csv(file)
        sample_info = dm.utility.extract_info(pathlib.Path(file).stem)
        data_raw.append(data_read.assign(**sample_info))

data_raw = pd.concat(data_raw, ignore_index=True).dropna(subset=['v'])
data_raw['viscosity'] = pd.to_numeric(data_raw['viscosity'])

# Define constants used in analysis
G = 9.806  # acceleration of gravity [m/s^2]
RHO = 950  # Density of silicone oil [kg/m^3]
GAMMA_OW = 40e-3  # Silicone oil-water interfacial tension [N/m]
RADIUS = 1.34e-3  # Nominal radius of a spherical 10 µL droplet [mm]
ANGLE_ERR = 0.2  # Uncertainty on angle positioning [deg]

# Calculate sine of inclination angle, force (in µN), capillary number and
# scaling form of friction (in µN)
data_raw['sin'] = np.sin(np.deg2rad(data_raw.angle))
data_raw['sin_err'] = (
    np.cos(np.deg2rad(data_raw.angle)) * np.deg2rad(ANGLE_ERR)
)
data_raw['force'] = (
    RHO * G * (4 / 3 * np.pi * RADIUS ** 3) * data_raw.sin * 1e6
)
data_raw['force_err'] = (
    RHO * G * (4 / 3 * np.pi * RADIUS ** 3) * data_raw.sin_err * 1e6
)
data_raw['viscosity_dyn'] = data_raw.viscosity * 1e-6 * RHO
data_raw['Ca'] = data_raw.viscosity_dyn * data_raw.v * 1e-3 / GAMMA_OW
data_raw['friction'] = (
    2 * np.pi * GAMMA_OW * RADIUS * data_raw.Ca ** (2 / 3) * 1e6
)

# Aggregate data
data_agg_first = aggregate_data(data_raw.query('droplet==1'))
data_agg_first['kind'] = 'first'

data_agg_plateau = aggregate_data(data_raw.query('droplet>=4'))
data_agg_plateau['kind'] = 'plateau'

data_agg = pd.concat([data_agg_first, data_agg_plateau])

# Save aggregated data
data_agg.to_csv("Results/scaling_aggregated.csv", index=False)

# Perform regression
data_fit = (
    data_agg
    .groupby(
        ['structure', 'viscosity', 'kind'],
        as_index=False
    )
    .apply(dm.scaling.fit_score)
)

data_fit.to_csv("Results/scaling_regression.csv", index=False)
print('Scaling analysis complete')
