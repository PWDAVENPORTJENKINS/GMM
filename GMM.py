# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:45:18 2019

P. W. Davenport-Jenkins
Charlie Reynolds is a gimp
University of Manchester
MSc Econometrics
N.B. Python 3.5+ is required
"""

from numba import jit
import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np


NO_PMC = 10


# An m vector of moment restrictions
# defined by the user.
def moment_conditions_matrix(beta, z):
    """
    There are m moment restrictions.
    There are N data points, called z.
    For each data point the m moment restrictions are computed.
    The result is an N x m matrix with each column representing
    a different moment restriction.
    """
    # In this example we do the normal distribution with location 3 and
    # scale 5.
    # central moments are defined as E[(z-mu)^k]
    mu = beta[0]
    sigma = beta[1]
    moments_matrix = np.concatenate(
        (
            z - mu,
            sigma**2 - (z - mu)**2,
            (z - mu)**3,
            3 * sigma**4 - (z - mu)**4,
            (z - mu)**5,
            15 * sigma**6 - (z - mu)**6,
            (z - mu)**7,
            105*sigma**8 - (z - mu)**8,
            (z - mu)**9,
            945*sigma**10 - (z - mu)**10
        ),
        axis=1
    )

    return moments_matrix


# This could perhaps be done more efficiently.
# einstien notation?
@jit
def GMM_weighting_matrix(moment_conditions_matrix):

    number_of_restrictions = len(moment_conditions_matrix[0])

    sample_size = len(moment_conditions_matrix)

    omega = np.zeros((number_of_restrictions, number_of_restrictions))

    for moment_vector in moment_conditions_matrix:

        omega = omega + np.outer(moment_vector, moment_vector)

    return np.linalg.inv(omega/sample_size)


def GMM_objective_function(x, sample, weighting_matrix):

    sample_moments_vector = moment_conditions_matrix(x, sample).mean(axis=0)

    return sample_moments_vector.T @ weighting_matrix @ sample_moments_vector


def CUE_GMM_objective_function(x, sample):

    sample_moments_vector = moment_conditions_matrix(x, sample).mean(axis=0)

    return sample_moments_vector.T @ \
        GMM_weighting_matrix(moment_conditions_matrix(x, sample)) @ \
        sample_moments_vector



def GMM(beta_initial, z, m, estimator):

    if estimator == "2-Step" or estimator == "K-Step":

        # First Stage: calculate initial beta.
        beta_1 = optimize.minimize(
                GMM_objective_function,
                beta_initial,
                args=(z, np.identity(m)),
                method="BFGS"
        )

        beta_1 = beta_1.x

        # Use this value to compute the optimal weighting matrix
        weighting_matrix = GMM_weighting_matrix(
                                    moment_conditions_matrix(beta_1, z)
        )

        # Second stage:: use the optimal weighting matrix to compute 2S-GMM
        # estimator of beta
        beta_2 = optimize.minimize(
                GMM_objective_function,
                beta_1,
                args=(z, weighting_matrix),
                method="BFGS"
        )

        if estimator == "2-Step":
            return beta_2.x
        else:
            beta_2 = beta_2.x
            tolerance = 0.000000000000000005
            assert 0 < tolerance
            weighting_matrix = GMM_weighting_matrix(
                                    moment_conditions_matrix(beta_2, z)
            )

            beta_new = optimize.minimize(
                    GMM_objective_function,
                    beta_2,
                    args=(z, weighting_matrix),
                    method="BFGS"
            )

            beta_new = beta_new.x
            beta_old = beta_2
            print(beta_new)
            while(np.max(np.abs(beta_new - beta_old)) >= tolerance):

                beta_old = beta_new

                # Use this value to compute the optimal weighting matrix
                weighting_matrix = GMM_weighting_matrix(
                                    moment_conditions_matrix(beta_old, z)
                )

                beta_new = optimize.minimize(
                        GMM_objective_function,
                        beta_old,
                        args=(z, weighting_matrix),
                        method="BFGS"
                )

                beta_new = beta_new.x
            return beta_new

    elif estimator == "CUE":
        beta = optimize.minimize(
            CUE_GMM_objective_function,
            beta_initial,
            args=(z),
            method="BFGS"
        )

        return beta.x

    else:
        print("You need to select one of '2-Step', 'K-Step', or 'CUE'.")


normal_random_variable = stats.norm(3, 5)

# generate n random values from above distribution
sample = normal_random_variable.rvs(size=(500, 1))

beta = [2, 4]

two_step = GMM(beta, sample, NO_PMC, estimator="2-Step")
k_step = GMM(beta, sample, NO_PMC, estimator="K-Step")
cue = GMM(beta, sample, NO_PMC, estimator="CUE")
