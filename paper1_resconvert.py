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

import os
import copy
import multiprocessing as mp
import pygmo as pg
import EI_problem
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds

def hv_convergeplot():

    print(0)

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
        if np.any(front[i, :] - ref) >= 0:
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


def hv_summary2csv():
    '''
    this function reads parameter files to track experimental results
    read nd front from each seed, and load pareto front to generated reference point
    use this pf reference to calculate hv for each seed
    :return: two csv files (1) raw data hv collection, (2) mean median csv collection
    '''
    import json
    # problems_json = 'p/zdt_problems_hvnd.json'
    problems_json = 'p/zdt_problems_hvndr.json'
    # problems_json = 'p/zdt_problems_hv.json'
    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    seedmax = 29

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    num_methods = len(methods)
    hv_raw = np.zeros((seedmax, num_pro * 3))
    path = os.getcwd()
    path = path + '\paper1_results'
    plt.ion()
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

                fig, ax = plt.subplots()
                ax.scatter(pf[:, 0], pf[:, 1])
                ax.scatter(nd_front[:, 0], nd_front[:, 1])
                plt.title(problem.name())
                plt.show()
                plt.pause(0.5)
                plt.close()
                hv = get_hv(nd_front, ref)
                hv_raw[seed, problem_i * num_pro + j] = hv
        a = 0
    plt.ioff()
    path = path + '\paper1_resconvert'
    if not os.path.exists(path):
        os.mkdir(path)
    saveraw = path + '\\hvraw.csv'
    np.savetxt(saveraw, hv_raw, delimiter=',')


    print(0)

def igd_summary2csv():
    print(0)

if __name__ == "__main__":
    hv_summary2csv()
    print(0)