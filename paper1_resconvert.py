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

def hv_convergeplot(k):
    '''this function needs specify a problem internally
    and plot its hv convergence
    find the following line to decide which problem to process
    problem = eval(target_problems[2])
    ** warning: dependent on running of hv_summary2csv or trainy_summary2csv
    ** to create seed selection
    *** or plot same seed change the following line
    '''
    import json
    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']

    method_selection = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    problem = eval(target_problems[k])
    pf = get_paretofront(problem, 100)
    nadir = np.max(pf, axis=0)
    ref = nadir * 1.1

    n_vals = problem.n_var
    if 'ZDT' in problem.name():
        number_of_initial_samples = 11 * n_vals - 1
        max_eval = 100
    if 'WFG' in problem.name():
        max_eval = 250
        number_of_initial_samples = 200

    num_plot = max_eval - number_of_initial_samples
    hvplot = np.zeros((3, num_plot))



    path = os.getcwd()
    path = path + '\\paper1_results\\paper1_resconvert\\median_id.joblib'
    median_id = load(path)
    seed = [int(median_id[problem.name() + '_' + method_selection[0]]),
            int(median_id[problem.name() + '_' + method_selection[1]]),
            int(median_id[problem.name() + '_' + method_selection[2]])]
    seed = [1] * 3

    path = os.getcwd()
    path = path + '\paper1_results'
    for m in range(3):
        savefolder = path + '\\' + problem.name() + '_' + method_selection[m]
        savename = savefolder + '\\trainy_seed_' + str(seed[m]) + '.csv'
        print(savename)
        trainy = np.loadtxt(savename, delimiter=',')
        trainy = np.atleast_2d(trainy)
        for i in range(number_of_initial_samples, max_eval):
            f = trainy[0:i, :]
            hvplot[m, i-number_of_initial_samples] = get_f2hv(f, ref)


    # m = ['normalization f', 'normalization nd', 'nd and ideal search']
    m = ['normalization nd', 'normalization nd and ideal search']

    style = ['dotted', '--']
    plt.ion()
    fig, ax = plt.subplots()
    # algs = {}
    for i in range(2):
        ax.plot(hvplot[i+1, :], linestyle=style[i])
    ax.legend(m)
    ax.set_xlabel('iterations')
    ax.set_ylabel('Hypervolume')
    plt.title(problem.name())
    plt.pause(1)


    path = os.getcwd()
    savefolder = path + '\\paper1_results\\process_plot'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + '_hvconverge.eps'
    savename2 = savefolder + '\\' + problem.name() + '_hvconverge.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)

    plt.ioff()


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

def get_paretofront(problem, n):
    from pymop.factory import get_uniform_weights
    if problem.name() == 'DTLZ2' or problem.name() == 'DTLZ1':
        ref_dir = get_uniform_weights(n, 2)
        return problem.pareto_front(ref_dir)
    else:
        return problem.pareto_front(n_pareto_points=n)
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
        pf = get_paretofront(problem, 100)
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

    #--- save
    median_id = {}
    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        for j in range(num_methods):
            method_selection = methods[j]
            name = problem.name() + '_' + method_selection
            median_id[name] = hv_stat[3, problem_i * 3 + j]
    saveName = path + '\\' + 'median_id.joblib'
    dump(median_id, saveName)




    savestat = path + '\\hvstat.csv'

    np.savetxt(savestat, hv_stat, delimiter=',')

    print(0)


def get_ndfront(train_y):
    '''
       :param train_y: np.2d
       :return: nd front points extracted from train_y
       '''
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    ndf_index = ndf[0]
    nd_front = train_y[ndf_index, :]
    return nd_front

def get_f2hv(f, ref):
    '''
    this function takes training data as input calculate hv with ref
    :param f:
    :param ref:
    :return: one value
    '''
    nd_front = get_ndfront(f)
    hv = get_hv(nd_front, ref)
    return hv

def trainy_summary2csv():
    '''
        this function reads parameter files to track experimental results
        read saved training data  from each seed, and cut the right number of evaluation
        calculate pareto front to generated reference point
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
        pf = get_paretofront(problem, 100)
        nadir = np.max(pf, axis=0)
        ref = nadir * 1.1
        if 'ZDT' in problem.name():
            evalnum = 100
        if 'WFG' in problem.name():
            evalnum = 250
        for j in range(num_methods):
            method_selection = methods[j]
            savefolder = path + '\\' + problem.name() + '_' + method_selection
            for seed in range(seedmax):
                savename = savefolder + '\\trainy_seed_' + str(seed) + '.csv'
                print(savename)
                trainy = np.loadtxt(savename, delimiter=',')
                trainy = np.atleast_2d(trainy)
                '''
                fig, ax = plt.subplots()
                ax.scatter(pf[:, 0], pf[:, 1])
                ax.scatter(nd_front[:, 0], nd_front[:, 1])
                plt.title(problem.name())
                plt.show()
                plt.pause(0.5)
                plt.close()
                '''
                # fix bug
                trainy = trainy[0:evalnum, :]
                hv = get_f2hv(trainy, ref)
                print(hv)
                hv_raw[seed, problem_i * num_methods + j] = hv
    # (2) mean median collection
    hv_stat = np.zeros((4, num_pro * 3))
    for i in range(num_pro * 3):
        hv_stat[0, i] = np.mean(hv_raw[:, i])
        hv_stat[1, i] = np.std(hv_raw[:, i])
        sortlist = np.argsort(hv_raw[:, i])
        hv_stat[2, i] = hv_raw[:, i][sortlist[int(29 / 2)]]
        hv_stat[3, i] = sortlist[int(29 / 2)]

    plt.ioff()
    path = path + '\paper1_resconvert'
    if not os.path.exists(path):
        os.mkdir(path)
    saveraw = path + '\\hvraw.csv'
    np.savetxt(saveraw, hv_raw, delimiter=',')

    target_problems = hyp['MO_target_problems']
    mat2csv(target_problems, hv_raw, hv_stat, methods, seedmax)

    # --- save
    median_id = {}
    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        for j in range(num_methods):
            method_selection = methods[j]
            name = problem.name() + '_' + method_selection
            median_id[name] = hv_stat[3, problem_i * 3 + j]
    saveName = path + '\\' + 'median_id.joblib'
    dump(median_id, saveName)

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
    # trainy_summary2csv()
    for k in range(0,12):
        hv_convergeplot(k)
    # hv_summary2csv()
    # hv_medianplot()
