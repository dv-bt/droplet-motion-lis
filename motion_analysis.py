"""
===============
motion_analysis
===============

This script performs a batch analysis of the motion of droplets in recorded
videos and extracts the portion of motion with constant velocity, or, if
dealing with crossing experiments, calculates the velocity and acceleration
over the whole recorded dataset. The script also updates the database storing
the information about crossing points.
All results are saved to csv files.

Optional arguments
------------------
-m --mode : 'new' or 'all'
    Analysis mode: restrict analysis to new files (i.e. those without a
    corresponding analysed file) or perform for all suitable files
-dr --detect_roi : float
    Region of interest for detection to be passed to
    dropletmotion.core.DropletTrack.droplet_detect, fraction from left
-ns --needle_check_skip
    Skip routing for checking if needle is in video frame
-v --velocity_only
    Perform only velocity calculation
"""

import pathlib
import glob
import argparse
import runpy
import logging
import tqdm
import pandas as pd
import dropletmotion as dm


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--mode', required=False, default='new',
                help="Analysis mode: 'new' videos or 'all' videos")
ap.add_argument('-dr', '--detect_roi', required=False,
                help="Region of interest for detection, fraction from left")
ap.add_argument('-ns', '--needle_check_skip', required=False,
                action='store_true', help="Skip needle in frame check routine")
ap.add_argument('-v', '--velocity_only', required=False,
                action='store_true', help="Perform only velocity calculation")
args = vars(ap.parse_args())

# Script variables
DATA_PATH = 'Data_example/'  # Path to the data folder

# Build keyword aguments to pass to dm.core.DropletTrack.droplet_detect from
# parsed values
kwargs_list = ['detect_roi', 'needle_check_skip']
kwargs = {i:j for i, j in args.items() if j is not None and i in kwargs_list}
if 'needle_check_skip' in kwargs:
    kwargs['needle_check'] = not kwargs.pop('needle_check_skip')
if 'detect_roi' in kwargs:
    kwargs['detect_roi'] = float(kwargs['detect_roi'])

# Initialize logger
logging.basicConfig(
    filename='motion_analysis.log',
    level = logging.INFO,
    format='%(asctime)s :: %(name)s :: %(levelname)-7s :: %(message)s'
)

# %% Video analysis
if not args['velocity_only']:
    video_search = DATA_PATH + '**/*.mp4'
    video_list = glob.glob(video_search, recursive=True)

    # Initialize counting variables
    video_count = 0
    error_count = 0

    for file in tqdm.tqdm(video_list, desc='Analysing videos'):

        video_path = pathlib.Path(file)

        analysed_file = (
            video_path.parent / 'Extracted data' /
            (video_path.stem + ' position.csv')
        )

        # Skip if file already analysed
        if not (args['mode'] == 'new' and analysed_file.is_file()):
            video = dm.core.DropletTrack(video_path)
            break_flag = video.droplet_detect(**kwargs)
            video_count += 1
            if break_flag:
                error_count += 1

    print(f'Video analysis completed. {video_count} files analyzed')
    if error_count > 0:
        print(error_count, 'errors. See log for details')

# %% Velocity and acceleration analysis
file_search = DATA_PATH + '**/*position.csv'

# Exclude files with these keywords from the analysis
file_list = glob.glob(file_search, recursive=True)

# Initialize counting variables
file_count = 0

for file_path in tqdm.tqdm(file_list, desc='Analysing position files'):

    analysed_file = (
        file_path.replace('position', 'acceleration') if 'exp_C' in file_path
        else file_path.replace('position', 'constant_velocity')
    )

    if not (args['mode'] == 'new' and pathlib.Path(analysed_file).is_file()):
        if 'exp_C' in file_path:
            dm.crossing.extract_acceleration(file_path)
        else:
            dm.velocity.extract_velocity(file_path)
        file_count += 1

print(f'Velocity analysis completed. {file_count} files analyzed')

# %% Update crossing point database
crossing_search = DATA_PATH + 'Crossing_offset/*cross_offset.tiff'
crossing_list = glob.glob(crossing_search)

# Initialize loop variables
crossing_results_list = []
crossing_count = 0

for file_path in tqdm.tqdm(crossing_list, desc='Analysing crossing points'):
    crossing_results_list.append(dm.crossing.crossing_point(file_path))
    crossing_count += 1

crossing_results = pd.concat(crossing_results_list, ignore_index=True)

# Read manually obtained result
try:
    crossing_results_manual = pd.read_csv(
        DATA_PATH + 'Crossing_offset/crossing_offset_manual.csv'
    )
    crossing_results = pd.concat(
        [crossing_results, crossing_results_manual], ignore_index=True
    )
except FileNotFoundError:
    pass

crossing_results.to_csv(DATA_PATH + 'crossing_offset_database.csv', index=False)

# %% Run data analysis script

globals_pass = dict(data_path=DATA_PATH)

runpy.run_path('data_analysis_scaling.py', init_globals=globals_pass)
runpy.run_path('data_analysis_time.py', init_globals=globals_pass)
runpy.run_path(
    'data_analysis_crossing.py', init_globals=globals_pass, run_name='__main__'
)
