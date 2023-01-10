# droplet-motion-lis

Analysis of droplet motion on liquid-infused surface (LIS).

### Table of contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [Methods](#methods)
  - [General data pipeline](#general-data-pipeline)
  - [Step 1: droplet detection and tracking](#step-1-droplet-detection-and-tracking)
  - [Step 2: velocity and acceleration calculation](#step-2-velocity-and-acceleration-calculation)
    - [Constant velocity validation](#constant-velocity-validation)
  - [Step 3: Feature analysis](#step-3-feature-analysis)
    - [Automation of regression bounds](#automation-of-regression-bounds)
- [Requirements](#requirements)
- [Authorship](#authorship)
- [References](#references)

## Introduction

The code in this repository was developed to support the analysis of experimental data of water droplets moving on inclined liquid infused surfaces (LIS). The context, goal, and results of this investigation are reported in detail in [[Bottone 2023]](#Bottone2023).

## Usage

The `motion_analysis.py` script performs the entire analysis pipeline on the experimental videos in the `Data` folder. Results of the analysis are then saved to the `Results` folder. A reduced dataset is provided for illustrative purposes, and the full dataset is available upon request.

The analysis relies on the code provided in the `dropletmotion` package. For correct operation, files must be [appropriately named](dropletmotion/README.md#notes-on-file-naming).

A detailed description of the analytical steps is contained in the [Methods](#methods) section.

## Repository structure

```
droplet-motion-lis
├── dropletmotion/
├── Data/
│   ├── Crossing_offset/
│   ├── Videos/
│   │   └── Extracted data/
│   └── crossing_offset_database.csv
├── Results/
│   ├── crossing_peaks.csv
│   ├── scaling_aggregated.csv
│   ├── scaling_regression.csv
│   └── time_interval_aggregated.csv
├── README.md
├── data_analysis_crossing.py
├── data_analysis_scaling.py
├── data_analysis_time.py
├── motion_analysis.py
└── LICENSE
```

## Methods

### General data pipeline

Raw data is processed according to the following pipeline:

   1. Experimental videos (showing a series of droplets moving on a surface) are processed to extract droplet **position** $x$ *vs* time $t$ signals
   2. **Velocity** $v$ and, if required, **acceleration** $a$ are calculated from the $x$ *vs* $t$ signals
   3. The $v$ and $a$ signals are analysed to extract the features relevant for the kind of experiment performed, which are described in detail in [[Bottone 2023]](#Bottone2023)

### Step 1: droplet detection and tracking

Droplet detection and tracking from the recorded videos are largely based on the [OpenCV library](https://github.com/opencv/opencv), and are executed through the `dropletmotion.core.DropletTrack` class.

Droplets are detected as soon as they fully enter the frame (moving from left to right) by template matching; the appropriate template is based on the first droplet in the video, which is found by a tailored blob analysis. Once detected, droplets are tracked with a CSRT (Channel and Spatial Reliability Tracker) tracking algorithm, originally developed by [[Lukežič 2018]](#Lukežič2018).

The position of the droplet centre of mass $x$ as a function of time is simply calculated from the image moments of the droplet contour, while the position of the advancing and the receding fronts of the droplet is extracted from an analysis of the droplet profile implemented in `dropletmotion.core.drop_profile`.

### Step 2: velocity and acceleration calculation

Velocity $v$ is calculated from position $x$ *vs* time $t$ signals with the Total Variation Regularization (TVR) algorithm developed by [[Chartrand 2011]](#Chartrand2011), using the [tvregdiff](https://github.com/stur86/tvregdiff) Python implementation by Simone Sturniolo. The functions used for this purpose are contained in the `dropletmotion.velocity` module.

The value of the regularization parameter $\lambda$ was chosen based on the data density $\delta$ – defined as the number of data points per mm of droplet travel – as:

$$
\begin{cases}
    \lambda = 10^{-4}\delta^{2.3} \quad &\text{if}\ \delta \geq 10\ \mathrm{pts\ mm^{-1}}\, ,\\
    \lambda = 10^{-3} &\text{if}\ \delta < 10\ \mathrm{pts\ mm^{-1}}\, .
\end{cases}
$$

Higher values of $\delta$ are representative of a noisier signal, and a higher $\lambda$ is consequently appropriate.

In crossing experiments (see [[Bottone 2023]](#Bottone2023)) $\lambda$ is fixed at 10<sup>−1</sup>, lower than what predicted by the above criterion for the relevant signals, in order to preserve detail at the cost of a slightly noisier derivative. Acceleration $a$ of these signals is calculated using the same approach and parameters by differentiating $v$.

#### Constant velocity validation

Some experiments rely on the existence of a constant terminal velocity of the droplets. In practice, it's possible that the recorded signals contain non-constant parts. The signals, then, have to be appropriately filtered and validated. This routine is implemented in `dropletmotion.velocity.constant_velocity`.

Each signal is divided in 10 pieces and each combination of contiguous pieces containing at least 3 elements is validated.
This is achieved by imposing two conditions, both of which have to be fulfilled:

1. The average relative gradient $\lang \partial_x v \rang$ evaluated over the relevant portion of the signal is lower than 1%, as:

    $$
    \lang \partial_xv \rang = \frac{1}{N} \sum_i^N\frac{\partial v}{\partial x}\frac{1}{v}\bigg|_{x_i} < 0.01\, ,
    $$

    where $N$ is the number of recorded points in the relevant portion of the signal. The $\partial v / \partial x$ term is calculated by centred finite differences.

2. The standard deviation of velocity $\sigma_v$, relative to the average velocity $\lang v \rang$, is lower than 4%, as:
   $$
   \frac{\sigma_v}{\lang v \rang} < 0.04\, .
   $$

The longest portion of signal fulfilling both conditions was taken as the constant velocity for that droplet; in case of a tie, the portion with the lower $\lang \partial_x v \rang$ was taken. If no suitable portion of the signal was found, the signal was discarded.

### Step 3: Feature analysis

The $v$ and $a$ signals obtained in step 2 are analysed to extract the features relevant for the performed experiment. A thorough description of these features is provided in [[Bottone 2023]](#Bottone2023) and the methods for their extraction are documented in the [droplemotion package readme](dropletmotion/README.md). 

Additional details are provided here for the more complex procedures.

#### Automation of regression bounds

Two scaling regimes are found in the driving force $F_\mathrm{d}$ _vs_ friction $F_\mathrm{f}$ curves: one at low _v_ (and $F_\mathrm{f}$) and one at high _v_ [[Keiser 2017]](#Keiser2017). The selection of bounds used for regression of the two regimes was automated on the basis of the following considerations:

1. Fitting at very low and very high $F_\mathrm{f}$ is usually very accurate;
1. A fit with more data points is better than a fit with fewer data points;
1. Regression accuracy should be as high as possible;
1. The upper limit for the low _v_ regime $F_\mathrm{f,low}$ can at most coincide with the lower limit for the high _v_ regime $F_\mathrm{f,high}$:
    $$
    F_\mathrm{f, low} \leq F_\mathrm{f, high}\,;
    $$
1. More strictly, the critical transition friction $F_\mathrm{f, crit}$ must be between the two limits of condition 4:
    $$
    F_\mathrm{f, low} \leq F_\mathrm{f, crit} \leq F_\mathrm{f, high}\, .
    $$

Taking into account the above conditions, both regimes are fitted at the same time for each combination of points that fulfils **condition 4** and that includes at least the 3 respective extreme points (**condition 1**). $F_\mathrm{f, crit}$ is calculated for each of these pairs, as the intersection between the two regimes, and a score $S_i$ is calculated for those that fulfil **condition 5** as:

$$
S_i = \overline{R}^2_{\mathrm{low},i}\ln N_{\mathrm{low},i} + \overline{R}^2_{\mathrm{high},i}\ln N_{\mathrm{high},i}\, ,
$$

where $N_{\mathrm{low}, i}$ and $N_{\mathrm{high}, i}$ are the number of points in the $i^\mathrm{th}$ pair of regressions, while $\overline{R}^2_{\mathrm{low},i}$ and $\overline{R}^2_{\mathrm{high},i}$ are the adjusted coefficients of determination of the $i^\mathrm{th}$ pair of regressions.

The $\overline{R}^2$ terms makes sure that **condition 3** is considered. On the other hand, the $\ln N$ terms takes into account **condition 2**, but with diminishing returns as the regression moves away from the extremes, as per **condition 1**. The pair of regressions with the highest score is selected. This procedure is implemented in `dropletmotion.scaling.fit_score`.

## Requirements

A complete list of the packages in the virtual environment used for the development of this code is included in [environment.yml](environment.yml).

## Authorship

All the custom code was written by [Davide Bottone](mailto:davide.bottone@chem.uzh.ch) at the University of Zurich. This research was supported by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 722497—LubISS

## References

<a id="Bottone2023">[Bottone 2023]</a>
Bottone, D. & Seeger, S.
Droplet memory on liquid infused surfaces, *in preparation*

<a id="Chartrand2011">[Chartrand 2011]</a>
Chartrand, R.
Numerical differentiation of noisy, nonsmooth data. *ISRN Applied Mathematics* **2011**, 1–11 (2011).

<a id="Keiser2017">[Keiser 2017]</a>
Keiser, A., Keiser, L., Clanet, C. & Quéré, D.
Drop friction on liquid-infused materials.
*Soft Matter* **13**, (2017).

<a id="Lukežič2018">[Lukežič 2018]</a>
Lukežič, A., Vojíř, T., Čehovin Zajc, L., Matas, J. & Kristan, M.
Discriminative Correlation Filter Tracker with Channel and Spatial Reliability.
*Int J Comput Vis* **126**, 671–688 (2018).