import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
    DTLZ1, DTLZ2, \
    BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
import pyDOE
from cross_val_hyperp import cross_val_krg
from joblib import dump, load
import time
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
    MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, ego_fitness, EI
from matplotlib.lines import Line2D

import os
import copy
import multiprocessing as mp
import pygmo as pg
import EI_problem
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds

def hv_convergeplot():

    print(0)
def get_paretofront(problem, n):
    from pymop.factory import get_uniform_weights
    if problem.name() == 'DTLZ2' or problem.name() == 'DTLZ1':
        ref_dir = get_uniform_weights(n, 2)
        return problem.pareto_front(ref_dir)
    else:
        return problem.pareto_front(n_pareto_points=n)

def get_hv(front, ref):
    '''
    this function calcuate hv with respect to ref
    :param front:
    :param ref:
    :return:
    '''
    front = np.atleast_2d(front)
    new_front = []
    n = front.shape[0]
    m = front.shape[1]

    # eliminate front instance which is beyond reference point
    for i in range(n):
        if np.any(front[i, :] - ref >= 0):
            continue
        else:
            new_front = np.append(new_front, front[i, :])

    # elimiate case where nd is far way off
    if len(new_front) == 0:
        hv = 0
    else:
        new_front = np.atleast_2d(new_front).reshape(-1, m)
        # calculate hv
        hv_class = pg.hypervolume(new_front)
        hv = hv_class.compute(ref)

    return hv

def mat2csv(target_problems, matrix, matrix_sts, methods, seedmax):
    '''
    this function iterate through problems and create csv with headers and index
    :param target_problems:
    :param matrix:
    :return:
    '''
    num_methods = len(methods)
    num_prob = len(target_problems)

    # write header
    path = os.getcwd()
    path = path + '\\paper1_results\\'
    savefile = path + 'paper1_resconvert\\results_convert.csv'

    with open(savefile, 'w') as f:
        # write header
        f.write(',')
        for i in range(num_prob):
            for j in range(num_methods):
                m = eval(target_problems[i]).name() + '_' + methods[j]
                f.write(m)
                f.write(',')
        f.write('\n')

        for seed in range(seedmax):
            # write seed
            f.write(str(seed))
            f.write(',')

            # write matrix
            for i in range(num_prob * num_methods):
                f.write(str(matrix[seed, i]))
                f.write(',')
            f.write('\n')

        # write statistics
        sts = ['mean', 'std', 'median', 'median_id']
        for s in range(4):
            f.write(sts[s])
            f.write(',')
            for i in range(num_prob * num_methods):
                f.write(str(matrix_sts[s, i]))
                f.write(',')
            f.write('\n')


def hv_medianplot():
    '''
    this function plot the final results of each problem
    :param seed:
    :return:
    '''
    import json

    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    num_methods = len(methods)


    prob_id = 2
    problem = target_problems[prob_id]
    method_selection = methods[2]
    seed = 19
    problem = eval(problem)

    path = os.getcwd()
    path = path + '\paper1_results'
    savefolder = path + '\\' + problem.name() + '_' + method_selection
    savename = savefolder + '\\nd_seed_' + str(seed) + '.csv'
    nd_front = np.loadtxt(savename, delimiter=',')
    nd_front = np.atleast_2d(nd_front)
    pf = get_paretofront(problem, 1000)

    plt.ion()
    fig, ax = plt.subplots()
    ax.scatter(pf[:, 0], pf[:, 1], s=1)
    ax.scatter(nd_front[:, 0], nd_front[:, 1], facecolors='white', edgecolors='red', alpha=1)
    plt.title(problem.name())
    plt.legend(['PF', 'alg results'])
    plt.xlabel('f1')
    plt.ylabel('f2')

    #-----
    path = os.getcwd()
    savefolder = path + '\\paper1_results\\plots'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + '_' + method_selection + '_final_' + str(seed) + '.eps'
    savename2 = savefolder + '\\' + problem.name() + '_' + method_selection + '_final_' + str(seed) + '.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)

    plt.pause(2)
    plt.close()


def hv_summary2csv():
    '''
    this function reads parameter files to track experimental results
    read nd front from each seed, and load pareto front to generated reference point
    use this pf reference to calculate hv for each seed
    :return: two csv files (1) raw data hv collection, (2) mean median csv collection
    '''
    import json

    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]
    seedmax = 29

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    num_methods = len(methods)
    hv_raw = np.zeros((seedmax, num_pro * 3))
    path = os.getcwd()
    path = path + '\paper1_results'
    # plt.ion()
    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        pf = problem.pareto_front(n_pareto_points=100)
        nadir = np.max(pf, axis=0)
        ref = nadir * 1.1
        for j in range(num_methods):
            method_selection = methods[j]
            savefolder = path + '\\' + problem.name() + '_' + method_selection
            for seed in range(seedmax):
                savename = savefolder + '\\nd_seed_' + str(seed) + '.csv'
                print(savename)
                nd_front = np.loadtxt(savename, delimiter=',')
                nd_front = np.atleast_2d(nd_front)
                '''
                fig, ax = plt.subplots()
                ax.scatter(pf[:, 0], pf[:, 1])
                ax.scatter(nd_front[:, 0], nd_front[:, 1])
                plt.title(problem.name())
                plt.show()
                plt.pause(0.5)
                plt.close()
                '''
                hv = get_hv(nd_front, ref)
                print(hv)
                hv_raw[seed, problem_i * num_methods + j] = hv
    # (2) mean median collection
    hv_stat = np.zeros((4, num_pro*3))
    for i in range(num_pro*3):
        hv_stat[0, i] = np.mean(hv_raw[:, i])
        hv_stat[1, i] = np.std(hv_raw[:, i])
        sortlist = np.argsort(hv_raw[:, i])
        hv_stat[2, i] = hv_raw[:, i][sortlist[int(29/2)]]
        hv_stat[3, i] = sortlist[int(29/2)]


    plt.ioff()
    path = path + '\paper1_resconvert'
    if not os.path.exists(path):
        os.mkdir(path)
    saveraw = path + '\\hvraw.csv'
    np.savetxt(saveraw, hv_raw, delimiter=',')

    target_problems = hyp['MO_target_problems']
    mat2csv(target_problems, hv_raw, hv_stat, methods, seedmax)

    savestat = path + '\\hvstat.csv'

    np.savetxt(savestat, hv_stat, delimiter=',')

    print(0)

def igd_summary2csv():
    '''
    this function calculate igd similar to hv above
    create two files
    :return:
    '''


if __name__ == "__main__":
    hv_summary2csv()
    # hv_medianplot()
    print(0)