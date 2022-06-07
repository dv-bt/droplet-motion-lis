# Droplet Motion Analysis – dropletmotion

A Python package to analyse the motion of water droplets on Liquid Infused Surfaces (LIS)

### Table of contents

- [Introduction](#introduction)
- [Structure](#structure)
- [Usage](#usage)
  - [core](#core)
  - [crossing](#crossing)
  - [scaling](#scaling)
  - [utility](#utility)
  - [velocity](#velocity)
  - [Notes on file naming](#notes-on-file-naming)
- [Dependencies](#dependencies)
- [References](#references)

## Introduction

The `dropletmotion` package was developed to support the analysis of droplet dynamics on LIS, fabricated on silicone (polysilsesquioxane) nanostructures, described in [[Bottone 2022]](#Bottone2022). As such, it contains the code used for the following data pipeline:

   0. Experiments yield videos showing a series of droplets moving on a surface (**raw data**)
   1. Videos are processed to extract droplet **position** $x$ *vs* time $t$ signals
   2. **Velocity** $v$ and, if required, **acceleration** $a$ are calculated from the $x$ *vs* $t$ signals
   3. The $v$ and $a$ signals are analysed for the features relevant for the kind of experiment performed

Additionally, the package contains also functions to perform analyses on aggregated data.

The code is necessarily highly specific for its application, and particularly so when it comes to step 1, where it had to be tailored to the optical characteristics of the experimental setup used. Nevertheless, its adaptation to suit a wider range of similar experimental problems should be relatively straightforward.

## Structure

The package contains five modules, structured as follows:

```
dropletmotion
├── __init__.py
├── README.md
├── core.py
├── crossing.py
├── scaling.py
├── utility.py
└── velocity.py
```

The `core` module contains code for the analysis of raw video data, while `utility` provides utility functions used throughout the package. The remaining modules mainly deal with processing of extracted position and velocity data.

## Usage

The classes and functions of this package are mainly intended to be used through the scripts in the [droplet-motion-lis]() repository, that contain the instructions for correct data processing, including data aggregation.

A concise description of each module and its contents follows below. 

### core

This module contains the main code used to extract the droplets' position from recorded videos.

- `DropletTrack`: class that stores a video and relevant metadata, taking the path to the video as the argument. Metadata is extracted from the [video name](#notes-on-file-naming) by `utility.extract_info`. It has two methods:
  - `droplet_detect`: performs video analysis and extract droplet position; `motion_save` is automatically called by default.
  - `motion_save`: saves the detected position to a csv file named "(video name) position.csv" in a subfolder "Extracted data".
- `drop_profile`: analyses the droplet profile and extracts the position of the advancing and receding front, as well as, optionally, an estimate of the droplet volume.

### crossing

This module contains functions used for the analysis of crossing experiments.

- `center_peak`: function that detects the centre of the velocity peak in the crossing region and scales position to this value; this is to ensure a homogeneous processing of signals, that is less sensitive to possible errors in crossing point calibration (see `crossing_point`).
- `peak_spacing`: calculates spacing and position of acceleration peaks, with respect to droplet centre of mass, advancing, and receding position.
- `crossing_point`: calculates the nominal crossing position, as well as trace droplet base diameter, from reference images, returning a database entry. The database is build using metadata extracted from file names by [`utility.extract_info`](), which must then follow the same [naming guidelines as videos](#notes-on-file-naming); such files do not require information about *tilt angle* and *video fps*, but require a *spot identifier* and *droplet volume*.
- `extract_acceleration`: calculates velocity and acceleration from position data "(video name) position.csv", and saves it to "(video name) acceleration.csv".

### scaling

This module contains classes and functions used for the analysis of scaling experiments.

- `RegimeTransition`: class that calculates and stores the transition between two given power laws.
- `odr_power`: power law of general form $y = c x^n$; used by `odr_power_fit`.
- `odr_power_fit`: perform orthogonal distance regression (ODR) on a given dataset with given uncertainties in both the dependent and the independent variables. The power law exponent $n$ in `odr_power` is fixed by the required argument `n` of the function.
- `fit_score`: performs regression on the whole scaling data set and finds the best pair of regressions for the upper and lower part of the set. This is accomplished by using a cumulative score $S$ for each pair of regressions that takes into account the accuracy of both regressions, as:
  $$
  S = \overline{R}^2_\mathrm{low}\ln N_\mathrm{low} + \overline{R}^2_\mathrm{high}\ln N_\mathrm{high}\, ,
  $$
  where $N$ is the number of points and $\overline{R}^2$ is the adjusted coefficient of determination of each regression.

### utility

This module contains utility functions used throughout the package.

- `extract_info`: extracts sample and video information from file name, which must be [appropriately formatted](#notes-on-file-naming).
- `experimental_t_int`: calculates the experimental time interval between droplets from position *vs* time data.
- `r_adj_calc`: calculates adjusted coefficient of determination for a given model on a data set, using the Wherry-1 formula in [[Yin & Fan 2001]](#YinFan2001)

### velocity

This module contains functions used for the calculation of velocity from position data.

- `regularized_derivative`: Calculates the derivative of an evenly spaced signal with a Total Variation Regularization (TVR) approach [[Chartrand 2011]](#Chartrand2011).
- `constant_velocity`: Calculates the constant velocity portion of a position *vs* time signal.
- `extract_velocity`: Extracts the constant velocity portion from a position csv file "(video name) position.csv" using `constant_velocity` and saves output to "(video name) constant_velocity.csv". 

### Notes on file naming

The file name of **all videos** needs to be appropriately formatted and include the following minimum information, separated by space, for correct processing by `utility.extract_info`; `{}` indicate a numeric field, while `[]` indicate an alphabetic one:

- *Sample name*, in the format `[structure code]_{viscosity in cSt}_{specimen ID #}`. The currently accepted values for `[structure code]` are:
  - **RW** for rods
  - **SW** for SNFs
  
  **Example**: `RW_20_2` indicates specimen #2 of rods infused with 20 cSt oil.

- *Tilt angle*, in the format `{angle in °}deg`.
- *Video fps*, in the format `{fps}fps`.
- *Reference size in pixels*, in the format `{size in pixels}px`. This is used to calculate the scale in mm/px of the video, using the `ref_mm` argument of `utility.extract_info` as the size in mm of the reference object (default = 2.90 mm).  
  
  **Example**: `1080px` yields a scale 2.90 mm / 1080 px = 2.68 × 10<sup>-3</sup> mm/px

**Example**: a minimum working video name could be `RW_20_2 15deg 200fps 1080px.mp4`.

Video names should also include additional information that, while not required by `utility.extract_info`, is necessary for the correct processing of the results further down the pipeline:

- **Mandatory for all experiments**
  - *Experiment identifier*, in the format `exp_[experiment code]`. Used to identify the kind of experiment performed in the video. The currently accepted values for `[experiment code]` are:
    - **S** for scaling experiments
    - **T** for time interval experiments
    - **C** for crossing experiments
- **Mandatory for time interval experiments**
  - *Time interval* between consecutive droplets, in the format `{time in s}secint`. This represents the *nominal* time interval, while the *experimental* time interval is calculated for each video from position *vs* time data
- **Mandatory for crossing experiments**
  - *Spot identifier*, in the format `spot{spot #}`. Used to distinguish between repetitions of the same experiments, or (crossing experiments only) to match videos with calibration images.
  - *Sequence identifier*, in the format `seq{sequence #}`. Used to identify sequential experiments having the same spot identifier. For crossing experiments, the values have the following meaning:
    - 1: reference droplet;
    - 3: probe droplets.
  - *Droplet volume*, in the format `{volume in µL}uL`.

**Example**: a complete video name could be, for crossing experiments, `SW_20_1 10deg 200fps 1079px spot5 seq3 10uL exp_C.mp4`.

## Dependencies

The `dropletmotion` package requires an installation of Python 3.9 or newer.

Additionally, `dropletmotion` depends on the following packages:
- numpy 1.21.2
- pandas 1.3.2
- scipy 1.7.0
- opencv 4.5.2
- tvregdiff ([GitHub page](https://github.com/dv-bt/tvregdiff))

## References

<a id="Bottone2022">[Bottone 2022]</a>
Bottone, D. & Seeger, S.
Droplet memory on liquid infused surfaces, *in preparation*

<a id="Chartrand2011">[Chartrand 2011]</a>
Chartrand, R.
Numerical differentiation of noisy, nonsmooth data. *ISRN Applied Mathematics* **2011**, 1–11 (2011).

<a id="YinFan2001">[Yin & Fan 2001]</a>
Yin, P. & Fan, X.
Estimating *R*<sup>2</sup> Shrinkage in Multiple Regression: A Comparison of Different Analytical Methods.
*The Journal of Experimental Education* 69, 203–224 (2001).
