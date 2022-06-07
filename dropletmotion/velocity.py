"""
This module conatins functions used to calculate the velocity
and acceleration of droplets from the detected position data

Functions
---------
regularized_derivative
    Calculate derivative of evenly spaced signal with TVR approach
constant_velocity
    Calculate constant velocity portion of position vs time signal
extract_velocity
    Extract constant velocity portion from position csv file and
    save output
"""

# Import required packages
import pathlib
import pandas as pd
import numpy as np
from tvregdiff import tvregdiff
from dropletmotion import utility


__all__ = ['regularized_derivative', 'constant_velocity', 'extract_velocity']


# Public functions

def regularized_derivative(signal, fps, n_iter=2, reg_param='auto') -> pd.Series:
    """
    Calculates the derivative of an evenly spaced time series signal
    using the Total Variation Regularization approach described by
    Rick Chartrand (R. Chartrand, ISRN Applied Mathematics, 2011,
    doi:10.5402/2011/164564) and implemented by Simone Sturniolo
    (https://github.com/stur86/tvregdiff).

    Parameters
    ----------
    signal : pd.DataFrame
        Series with the signal to be differentiated
    fps : int or float
        Sampling rate of the signal (i.e. frames per second)
    n_iter : int
        Number of iterations to pass to the differentiation
        algorithm. Going beyond 2 does not usually yield significant
        improvements (default=2)
    reg_param : float or str
        regularization parameter to pass to the differentiation
        algorithm. If 'auto', it's determined automatically based
        on the normalised data density, i.e. number of datapoints
        over maximum of the signal, by a call to reg_param_assign.
        NOTE: this approach might be unsuitable with anything that
        is not droplet position vs time. (default='auto')

    Returns
    -------
    der_out : pd.Series
        Series containing the calcualted derivative and having the same
        indices as x
    """

    sampling_int = 1 / fps

    if reg_param=='auto':
        reg_param = _reg_param_assign(len(signal) / signal.max())

    der = tvregdiff.TVRegDiff(
        signal, n_iter, reg_param, dx=sampling_int, scale='large', diffkernel='sq',
        plotflag=False, diagflag=False)

    der_out = pd.Series(der, index=signal.index)

    return der_out


def constant_velocity(data, min_group_size, v_grad_mean, v_std) -> pd.DataFrame:
    """
    Calculate the constant part of a velocity signal, with given tolerance

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a position (label 'x') and velocity (label 'v') data.
        Note: it should refer to a single droplet in one experiment
    min_group_size : int
        Minimum size of group in which data is divided for the detection of
        a constant velocity. It should be at most equal to 10.
    v_grad_mean : float
        Maximum average relative velocity gradient of the constant portion
        of velocity
    v_std : float
        Maximum average relative standard deviation of the constant portion
        of velocity

    Returns
    -------
    data_const : pd.DataFrame
        Slice of data with constant velocity, according to the given
        constraints. Returns empty dataframe if no constant portion detected
    """

    # Discard first 10% of the dataset, because of the higher incidence of
    # numerical errors and the proximity to deposition position
    data = data.tail(round(0.9 * len(data))).copy()

    # Calculate relative velocity gradient in the x-direction
    # Suppress numpy nan warnings, as they cannot be avoided with data filtering
    # and anyway result in nan propagation, which is appropriate in this context
    with np.errstate(invalid='ignore', divide='ignore'):
        data['v_grad'] = np.gradient(data.v, data.x) / data.v

    # Divide dataframe in 10 equal groups
    indices = np.array_split(data.index, 10, axis=0)
    for i, index in enumerate(indices):
        data.loc[index,'group'] = i

    grads = []

    # Loop through groups of minimum size min_group_size and calculate total
    # relative gradient and standard deviation of velocity
    for start in range(10):
        stop = start + min_group_size - 1
        while stop <= 9:
            data_sub = data.loc[(data.group>=start) & (data.group<=stop)]
            grads.append([
                start, stop, stop - start + 1,
                abs(data_sub.v_grad.mean()),
                data_sub.v.std() / data_sub.v.mean()
            ])
            stop += 1

    grads = pd.DataFrame(
        grads, columns=['start', 'stop', 'num', 'v_grad_mean', 'v_std']
    )

    # Find largest group that fulfills imposed conditions
    try:
        best_match = (
            grads.loc[(grads.v_grad_mean<v_grad_mean) & (grads.v_std<v_std)]
            .sort_values(by=['v_grad_mean'], ascending=True)
            .sort_values(by=['num'], ascending=False)
        ).iloc[0]

        data_const = data.loc[
            (data.group>=best_match.start) & (data.group<=best_match.stop),
             ['t', 'x', 'v']
        ]

    # If no group fulfills conditions returns empty dataframe
    except IndexError:
        data_const = pd.DataFrame([], columns=['t', 'x', 'v'])

    return data_const


def extract_velocity(
    file_path, min_group_size=3, v_grad_mean=0.01, v_std=0.04
) -> None:
    """
    Extract constant velocity portion of position vs time signal from a given
    csv file (output of dropletmotion.core.DropletTrack.motion_save) and save
    it to a csv file located in the same folder.

    Parameters
    ----------
    file_path : str
        Path to position vs time csv file
    min_group_size : int
        Argument passed to constant_velocity. Minimum size of group in which
        data is divided for the detection of a constant velocity. It should be
        at most equal to 10. (default=3)
    v_grad_mean : float
        Argument passed to constant_velocity. Maximum average relative velocity
        gradient of the constant portion of velocity (default=0.01)
    v_std : float
        Argument passed to constant_velocity. Maximum average relative standard
        deviation of the constant portion of velocity (default=0.04)

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
            .x.transform(regularized_derivative, fps=fps)
        )

        data_out = (
            data
            .groupby('droplet')
            .apply(
                constant_velocity, min_group_size=min_group_size,
                v_grad_mean=v_grad_mean, v_std=v_std
            )
            .droplevel(1)
            .reset_index()
        )

        # Assign experimental time interval, that cannot
        # be reconstructed from data_out
        data_out = data_out.merge(utility.experimental_t_int(data))

        # Save csv file
        data_out.to_csv(
            file_path.replace('position', 'constant_velocity'),
            index=False
        )

    except (KeyError, AttributeError):
        print(f'{file_name} not approriately formatted')


# Private functions

def _reg_param_assign(data_density) -> float:
    """ Calculate regularization parameter based on data density """
    if data_density < 10:
        reg_param = 1e-3
    else:
        reg_param = data_density ** 3.2 * 1e-4

    return reg_param
