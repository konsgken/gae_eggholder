#!/usr/bin/env python3

"""
 eggHolder.py: Implements an evolutionary algorithm for the eggholder
problem.
"""
__author__ = 'Konstantinos Gkentsidis'
__license__ = 'BSDv3'

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

matplotlib.use('Qt5Agg')


def egg_holder_ea() -> None:
    """ Main function of the evolutionary algorithm solving the egg holder
    problem.
    """
    global lambda_, int_max
    alpha = 0.05  # mutation probability
    lambda_ = 100  # population and offspring size
    k = 3  # tournament selection
    int_max = 500  # Boundary of the domain, not intended to be changed

    # initialize population
    population = int_max * np.random.rand(lambda_, 2)

    plot_population(population)

    for idx in range(0, 20):
        selected = selection(population, k)
        offspring = crossover(selected)
        joined_population = np.vstack((mutation(offspring, alpha), population))
        population = elimination(joined_population, lambda_)

        # show progress
        print(f'Iteration: {idx}, Mean fitness: {np.mean(objf(population))}')

        plot_population(population)

    return


def objf(x: np.array) -> np.array:

    """ Computes the objective fucnction at the vector of (x,y) values

    :param x: population
    :type x: numpy.array
    :return: 1D array containing the cost of the population samples
    :rtype: numpy.array
    """

    x = x.reshape((-1, 2))
    sas = np.sqrt(np.abs(x[:, 0] + x[:, 1]))
    sad = np.sqrt(np.abs(x[:, 0] - x[:, 1]))
    f_x = -x[:, 1] * np.sin(sas) - x[:, 0] * np.sin(sad)

    return f_x


def plot_population(population: np.array) -> None:
    """ Plot the population

    :param population: The population to plot
    :type population: numpy.array
    """

    x = np.linspace(0, int_max, 500).reshape(-1)
    y = np.linspace(0, int_max, 500).reshape(-1)

    X, Y = np.meshgrid(x, y)

    F = -Y * np.sin(np.sqrt(np.abs(X + Y))) - X * np.sin(np.sqrt(np.abs(X - Y)))

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.view_init(elev=72, azim=-137)
    ax.plot_surface(X, Y, F, rstride=8, cstride=8, cmap="jet", shade=False, alpha=0.7, linewidth=1)
    ax.scatter(population[:, 0],
               population[:, 1],
               objf(population) + 50, color='red', marker="X")
    print(population)
    plt.show()


def selection(population: np.array, k: int) -> np.array:
    """ Perform k-tournament selection to select pairs of parents

    :param population: population
    :type population: numpy.array
    :param k: the number of selected pairs from population
    :type k: integer
    :return: the selected candidates
    :rtype: numpy.array
    """

    selected = np.zeros((2 * len(population), 2))

    for idx in range(0, 2 * lambda_):
        ri = random.sample(range(0, lambda_), k)
        mi = np.argmin(objf(population[ri, :]))
        selected[idx, :] = population[ri[mi], :]

    return selected


def crossover(selected: np.array) -> np.array:
    """ Perform crossover

    :param selected: selected (np.array): The parents to produce the offsprings from
    :type selected: numpy.array
    :return: The offsprings produced from the crossover operation
    :rtype: numpy.array
    """

    weights = 3 * np.random.rand(lambda_, 2) - 1
    offspring = np.zeros((lambda_, 2))

    for idx in range(0, len(offspring)):
        offspring[idx, 0] = min(int_max, max(0, selected[2 * idx, 0] + weights[idx, 0] * (selected[2 * idx + 1, 0] - selected[2 * idx, 0])))
        offspring[idx, 1] = min(int_max, max(0, selected[2 * idx, 1] + weights[idx, 1] * (selected[2 * idx + 1, 1] - selected[2 * idx, 1])))

    return offspring


def mutation(offspring: np.array, alpha: float) -> np.array:
    """

    :param offspring: The produced offspring of the latest iteration
    :type offspring: numpy.array
    :param alpha: Offset of offspring selection
    :type alpha: float
    :return: Mutated offspring
    :rtype: numpy.array
    """

    indices = np.where(np.random.rand(len(offspring), 1) <= alpha)[0]
    if len(indices) != 0:

        offspring[indices, :] = offspring[indices, :] + 10 * np.random.randn(len(indices), 2)
        offspring[indices, 0] = min(int_max, max(0, max(offspring[indices, 0])))
        offspring[indices, 1] = min(int_max, max(0, max(offspring[indices, 1])))

    return offspring


def elimination(joined_population: np.array, keep: int) -> np.array:
    """ Eliminate the unfit candidate solutions

    :param joined_population: The population with the new offspring
    :type joined_population: numpy.array
    :param keep: The number of samples to keep from the population
    :type keep: integer
    :return: The samples that survived the elimination process
    :rtype: numpy.array
    """

    fvals = objf(joined_population)
    perm = np.argsort(fvals)
    survivors = joined_population[perm[0:keep], :]

    return survivors


if __name__ == '__main__':
    print('Running the eggholder evolutionary algorithmic solver')
    egg_holder_ea()
