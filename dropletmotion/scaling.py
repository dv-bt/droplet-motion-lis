"""
This module contains functions used in the analysis of scaling experiments

Classes
-------
RegimeTransition
    Indentify transition between lower and upper scaling regimes

Functions
---------
odr_power
    Power law model to be used in ODR fit
odr_power_fit
    Perform ODR fit with power law with fixed exponent
fit_score
    Identify best pair of lower and upper fit of data
"""

# Import requuired packages
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import odr
from dropletmotion import utility


@dataclass
class RegimeTransition():
    '''
    Calculate the transition between regimes at high and low angles and
    store the result.

    Parameters
    ----------
    regression_low : odr.Output
        ODR regression object of low-angle power law
    regression_high : odr.Output
        ODR regression object of the high-angle power law

    Attributes
    -------
    regression_low : odr.Output
        ODR regression object of low-angle power law
    regression_high : odr.Output
        ODR regression object of the high-angle power law
    x_trans : float
        Transition x
    x_trans_err : float
        Error on the transition x, calculated by error propagation
    y_trans : float
        Transition y
    y_trans_err : float
        Error on transition y, calculated by error propagation
    '''
    regression_low : odr.Output
    regression_high : odr.Output

    def __post_init__(self) -> None:
        ''' Calculate transition between regimes '''
        c_low = self.regression_low.beta[0]
        c_high = self.regression_high.beta[0]
        n_low = self.regression_low.beta[1]
        n_high = self.regression_high.beta[1]
        c_low_err = self.regression_low.sd_beta[0]
        c_high_err = self.regression_high.sd_beta[0]

        self.y_trans = (
            (c_low ** n_high / c_high ** n_low) ** (1 / (n_high - n_low))
        )
        self.y_trans_err = (
            self.y_trans * abs(n_low * n_high / (n_low - n_high)) *
            np.sqrt(
                (c_low_err / c_low / n_low) ** 2 +
                (c_high_err / c_high / n_high) ** 2
            )
        )

        self.x_trans = (c_low / c_high) ** (1 / (n_high - n_low))
        self.x_trans_err = (
            self.x_trans / abs(n_low - n_high) *
            np.sqrt(
                (c_low_err / c_low) ** 2 + (c_high_err / c_high) ** 2
            )
        )


def odr_power(params, x) -> np.array:
    """ Power law with exponent fixed by B[1] """
    return params[0] * x ** params[1]


def odr_power_fit(x, y, xerr, yerr, n) -> tuple[odr.Output, float]:
    """
    Finds the ODR for data {x, y} and returns the result with power law.
    Power law exponent must be fixed, and fit has only one linear
    parameter B[0].

    Parameters
    ----------
    x : np.array
        Values of the independent variable
    y : np.array
        Values of the dependent variable
    xerr : np.array
        Values of the uncertainties on the independent variable
    yerr : np.array
        Values of the uncertainties on the dependent variable
    n : float
        Exponent of the power law fit

    Returns
    -------
    output : odr.Output
        Output of the ODR fit
    r_adj : float
        Value of the adjusted r squared for the fit
    """
    power_law = odr.Model(odr_power)
    mydata = odr.RealData(x, y, sx=xerr, sy=yerr)
    beta0 = [50, n]
    ifixb = [1, 0]
    myodr = odr.ODR(mydata, power_law, beta0=beta0, ifixb=ifixb)
    output = myodr.run()
    r_adj = utility.r_adj_calc(x, y, odr_power, output.beta, free_params=1)
    return output, r_adj


def fit_score(data,  x='friction', y='force', n_low=1, n_high=0.5) -> pd.DataFrame:
    '''
    Perform regression on data and find best pair of upper and
    lower fit based on cumulative score.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with aggregated values containing central values and
        uncertainties.
        NOTE: there should be no duplicated angle entries, accomplished
        by either appropriate use of group_data, or by subsetting the
        dataframe
    x : str
        Label of the column containing the x values in df
        (default='friction')
    y : str
        Label of the column containing the y values in df
        (default='force')
    n_low : float or int
        Exponent of the power law valid at low angles (default=1)
    n_high : float or int
        Exponent of the power law valid at high angles (default=0.5)

    Returns
    -------
    score_out : pd.DataFrame
        Dataframe with the results of the fit, continaing the two
        regressions and sorted by best match.
    '''

    # Copy dataframe not to modify input
    data = data.copy()
    
    # Set error labels
    x_err = x + '_err'
    y_err = y + '_err'

    # Use angles as indices
    angle_list = data.angle.to_list()
    data.index = data.angle
    if data.angle.duplicated().sum() > 0:
        raise ValueError(
            'Duplicate angle indices: incorrect DataFrame grouping'
        )

    score_list = []
    regressions = {}
    r_adj = {}
    score_columns = [
        'angle_low', 'angle_high', 'score', 'prefactor_low',
        'prefactor_low_err', 'r_adj_low', 'prefactor_high',
        'prefactor_high_err', 'r_adj_high'
    ]

    # Loop for regression and scoring.
    # Fit must be evaluated on at least 3 points.
    for angle_low in angle_list[2:]:

        # Perform lower regression with n1
        data_low = data.loc[:angle_low]
        regressions['low'], r_adj['low'] = odr_power_fit(
            data_low[x],
            data_low[y],
            data_low[x_err],
            data_low[y_err],
            n=n_low
        )

        # Perform higher regression with n2
        for angle_high in reversed(
            angle_list[angle_list.index(angle_low):-2]
        ):
            data_high = data.loc[angle_high:]
            regressions['high'], r_adj['high'] = odr_power_fit(
                data_high[x],
                data_high[y],
                data_high[x_err],
                data_high[y_err],
                n=n_high
            )

            # Calculate transition between regimes
            transition = RegimeTransition(
                regressions['low'], regressions['high']
            )

            # Transition point must be between upper bound of low regression
            # and lower bound of high regression
            if (
                transition.x_trans > data_low.iloc[-1][x] and
                transition.x_trans < data_high.iloc[0][x]
            ):
                score = (
                    r_adj['low'] * np.log(len(data_low)) +
                    r_adj['high'] * np.log(len(data_high))
                )

                score_df_values = [
                    angle_low, angle_high, score,
                    regressions['low'].beta[0],
                    regressions['low'].sd_beta[0], r_adj['low'],
                    regressions['high'].beta[0],
                    regressions['high'].sd_beta[0], r_adj['high']
                ]

                score_list.append(
                    pd.DataFrame({
                        key:[values] for key, values
                        in zip(score_columns, score_df_values)
                    })
                )

    try:
        score_df = pd.concat(score_list, ignore_index=True)
        score_df = score_df.sort_values(by='score', ascending=False)

        # Select pair with highest score
        score_out = score_df.iloc[0, :]

    # Assign empty dataframe if no suitable results
    except ValueError:
        score_out = pd.Series(index=score_columns, dtype='object')

    return score_out
