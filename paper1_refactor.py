import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, \
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



def init_xy(number_of_initial_samples, target_problem, seed):
    n_vals = target_problem.n_var
    n_sur_cons = target_problem.n_constr

    # initial samples with hyper cube sampling
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples, criterion='maximin')  #, iterations=1000)

    xu = np.atleast_2d(target_problem.xu).reshape(1, -1)
    xl = np.atleast_2d(target_problem.xl).reshape(1, -1)

    train_x = xl + (xu - xl) * train_x

    # test
    # lfile = 'sample_x' + str(seed) + '.csv'
    # train_x = np.loadtxt(lfile, delimiter=',')

    out = {}
    target_problem._evaluate(train_x, out)
    train_y = out['F']

    if 'G' in out.keys():
        cons_y = out['G']
        cons_y = np.atleast_2d(cons_y).reshape(-1, n_sur_cons)
    else:
        cons_y = None

    # test
    '''
    lfile = 'sample_x' + str(seed) + '.csv'
    train_x_1 = np.loadtxt(lfile, delimiter=',')
    out = {}
    target_problem._evaluate(train_x_1, out)
    train_y_1 = out['F']

    plt.scatter(train_y[:, 0], train_y[:, 1])
    plt.scatter(train_y_1[:, 0], train_y_1[:, 1])
    plt.legend(['python', 'matlab'])
    plt.show()
    '''

    return train_x, train_y, cons_y


def confirm_search(new_y, train_y):
    obj_min = np.min(train_y, axis=0)
    diff = new_y - obj_min
    if np.any(diff < 0):
        return True
    else:
        return False

def normalization_with_self(y):
    '''
    normalize a y matrix, with its own max min
    :param y:
    :return:  normalized y
    '''
    y = check_array(y)
    min_y = np.min(y, axis=0)
    max_y = np.max(y, axis=0)
    return (y - min_y) / (max_y - min_y)
def denormalization_with_self(y_norm, y_normorig):
    '''

    :param y_norm: the list of vectors (num, feature) to be denormalized
    :param y_normorig: the list of y originally used for normalization
    :return: denormalized y_norm
    '''
    y = check_array(y_normorig)
    min_y = np.min(y, axis=0)
    max_y = np.max(y, axis=0)
    y_denorm = y_norm * (max_y - min_y) + min_y
    return y_denorm

def normalization_with_nd(y):
    y = check_array(y)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(y)
    ndf = list(ndf)
    ndf_size = len(ndf)
    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    nd_front = y[ndf_extend, :]

    # normalization boundary
    min_nd_by_feature = np.amin(nd_front, axis=0)
    max_nd_by_feature = np.amax(nd_front, axis=0)

    # rule out exception nadir and ideal are too close
    # add more fronts to nd front
    if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
        print('nd front aligned problem, re-select nd front')
        ndf_index = ndf[0]
        for k in np.arange(1, ndf_size):
            ndf_index = np.append(ndf_index, ndf[k])
            nd_front = y[ndf_index, :]
            min_nd_by_feature = np.amin(nd_front, axis=0)
            max_nd_by_feature = np.amax(nd_front, axis=0)
            if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
                continue
            else:
                break
    norm_y = (y - min_nd_by_feature) / (max_nd_by_feature - min_nd_by_feature)
    return norm_y

def denormalization_with_nd(y_norm, y):
    y = check_array(y)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(y)
    ndf = list(ndf)
    ndf_size = len(ndf)
    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    nd_front = y[ndf_extend, :]
    # normalization boundary
    min_nd_by_feature = np.amin(nd_front, axis=0)
    max_nd_by_feature = np.amax(nd_front, axis=0)
    # rule out exception nadir and ideal are too close
    # add more fronts to nd front
    if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
        print('nd front aligned problem, re-select nd front')
        ndf_index = ndf[0]
        for k in np.arange(1, ndf_size):
            ndf_index = np.append(ndf_index, ndf[k])
            nd_front = y[ndf_index, :]
            min_nd_by_feature = np.amin(nd_front, axis=0)
            max_nd_by_feature = np.amax(nd_front, axis=0)
            if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
                continue
            else:
                break

    y_denorm = y_norm * (max_nd_by_feature - min_nd_by_feature) + min_nd_by_feature
    return y_denorm

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

def get_ndfrontx(train_x, train_y):
    '''
    find design variables of nd front
    :param train_x:
    :param train_y:
    :return: nd front points extracted from train_x
    '''
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    ndf_index = ndf[0]
    return train_x[ndf_index, :]

def lexsort_with_certain_row(f_matrix, target_row_index):
    '''
    problematic function, given lexsort, it does not matter how upper
    rows are shuffled, sort is according to the last row.
    sort matrix according to certain row in fact last row
    e.g. sort the last row, the rest rows move its elements accordingly
    however, the matrix except last row is also sorted row wise
    according to number of min values each row has
    '''

    # f_matrix should have the size of [n_obj * popsize]
    # determine min
    target_row = f_matrix[target_row_index, :].copy()
    f_matrix = np.delete(f_matrix, target_row_index, axis=0)  # delete axis is opposite to normal

    f_min = np.min(f_matrix, axis=1)
    f_min = np.atleast_2d(f_min).reshape(-1, 1)
    # according to np.lexsort, put row with largest min values last row
    # -- this following sort is useless, can just go straight in 3d
    f_min_count = np.count_nonzero(f_matrix == f_min, axis=1)
    f_min_accending_index = np.argsort(f_min_count)
    # adjust last_f_pop
    last_f_pop = f_matrix[f_min_accending_index, :]

    # add saved target
    last_f_pop = np.vstack((last_f_pop, target_row))

    # apply np.lexsort (works row direction)
    lexsort_index = np.lexsort(last_f_pop)
    # print(last_f_pop[:, lexsort_index])
    selected_x_index = lexsort_index[0]

    return selected_x_index

def additional_evaluation(x_krg, train_x, train_y, problem,
                          ):
    '''
    this method only deal with unconstraint mo
    it does closeness
    :return: add kriging estimated x to training data.
    '''
    n_var = problem.n_var
    n_obj = train_y.shape[1]

    for i in range(n_obj):
        x_i = np.atleast_2d(x_krg[i]).reshape(-1, n_var)
        y_i = problem.evaluate(x_i, return_values_of=['F'])
        train_x = np.vstack((train_x, x_i))
        train_y = np.vstack((train_y, y_i))
    train_y = close_adjustment(train_y)
    return train_x, train_y


def check_krg_ideal_points(krg, n_var, n_constr, n_obj, low, up, guide_x):
    '''This function uses  krging model to search for a better x
    krg(list): krging model
    n_var(int): number of design variable for kriging
    n_constr(int): number of constraints
    n_obj(int): number of objective function
    low(list):
    up(list)
    guide_x(row vector): starting point to insert to initial population
    '''

    last_x_pop = []
    last_f_pop = []


    x_pop_size = 100
    x_pop_gen = 100

    # identify ideal x and f for each objective
    for k_i, k in enumerate(krg):
        problem = single_krg_optim.single_krg_optim(k, n_var, n_constr, 1, low, up)
        single_bounds = np.vstack((low, up)).T.tolist()

        guide = np.atleast_2d(guide_x[k_i, :])
        _, _, pop_x, pop_f = optimizer_EI.optimizer_DE(problem, problem.n_constr, single_bounds,
                                                       guide, 0.8, 0.8, 100, 100, False, None, **{})

        # save the last population for lexicon sort
        last_x_pop = np.append(last_x_pop, pop_x)
        last_f_pop = np.append(last_f_pop, pop_f)  # var for test

    # long x
    last_x_pop = np.atleast_2d(last_x_pop).reshape(n_obj, -1)
    x_estimate = []
    # lex sort because
    # considering situation when f1 min has multiple same values
    # choose the one with bigger f2 value, so that nd can expand

    for i in range(n_obj):
        x_pop = last_x_pop[i, :]
        x_pop = x_pop.reshape(x_pop_size, -1)
        all_f = []
        # all_obj_f under current x pop
        for k in krg:
            f_k, _ = k.predict(x_pop)
            all_f = np.append(all_f, f_k)

        # reorganise all f in obj * popsize shape
        all_f = np.atleast_2d(all_f).reshape(n_obj, -1)
        # select an x according to lexsort
        x_index = lexsort_with_certain_row(all_f, i)
        # x_index = lexsort_specify_baserow(all_f, i)

        x_estimate = np.append(x_estimate, x_pop[x_index, :])

    x_estimate = np.atleast_2d(x_estimate).reshape(n_obj, -1)

    return x_estimate

def idealsearch_update(train_x, train_y, krg, target_problem):
    n_vals = train_x.shape[1]
    n_sur_cons = 0
    n_sur_objs = train_y.shape[1]

    # return current ideal x of two objectives
    best_index = np.argmin(train_y, axis=0)
    guide_x = np.atleast_2d(train_x[best_index, :])

    # run estimated new best x on each objective
    x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs,
                                   target_problem.xl, target_problem.xu, guide_x)
    train_x, train_y = additional_evaluation(x_out, train_x, train_y, target_problem)
    return train_x, train_y


def nd2csv(train_y, target_problem, seed_index, method_selection, search_ideal):
    # (5)save nd front under name \problem_method_i\nd_seed_1.csv
    path = os.getcwd()
    path = path + '\paper1_results'
    if not os.path.exists(path):
        os.mkdir(path)
    savefolder = path + '\\' + target_problem.name() + '_' + method_selection + '_' + str(int(search_ideal))
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    savename = savefolder + '\\nd_seed_' + str(seed_index) + '.csv'
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    ndindex = ndf[0]
    ndfront = train_y[ndindex, :]
    np.savetxt(savename, ndfront, delimiter=',')

    savename = savefolder + '\\trainy_seed_' + str(seed_index) + '.csv'
    np.savetxt(savename, train_y, delimiter=',')

def pfnd2csv(pf_nd, target_problem, seed_index, method_selection, search_ideal):
    path = os.getcwd()
    path = path + '\paper1_results'
    if not os.path.exists(path):
        os.mkdir(path)
    savefolder = path + '\\' + target_problem.name() + '_' + method_selection + '_' + str(int(search_ideal))
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    savename = savefolder + '\\hvconvg_seed_' + str(seed_index) + '.csv'
    pf_nd = pf_nd.reshape(-1, 2)
    np.savetxt(savename, pf_nd, delimiter=',')


def plot_initpop(train_y, target_problem, method_selection, search_ideal, seed):
    '''
    this function save png and eps plot of init population w.r.t. pareto front
    :param train_y:
    :param target_problem:
    :param seed
    :return: no return saved to target folder
            \problem_method_ideal\initpop_1.csv
    '''
    path = os.getcwd()
    path = path + '\paper1_results'
    if not os.path.exists(path):
        os.mkdir(path)
    savefolder = path + '\\' + target_problem.name() + '_' + method_selection + '_' + str(int(search_ideal))
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    savename1 = savefolder + '\\initpop_' + str(seed) + '.eps'
    savename2 = savefolder + '\\initpop_' + str(seed) + '.png'

    pf = get_paretofront(target_problem, 100)
    plt.scatter(pf[:, 0], pf[:, 1], c='red')
    plt.scatter(train_y[:, 0], train_y[:, 1], marker='X', c='blue')
    nd = get_ndfront(train_y)
    plt.scatter(nd[:, 0], nd[:, 1], marker='X', c='green')

    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.legend(['PF', 'Init population', 'Init nd front'])
    plt.title(target_problem.name())
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)

def get_paretofront(problem, n):
    from pymop.factory import get_uniform_weights
    n_obj = problem.n_obj
    if problem.name() == 'DTLZ1' or problem.name() == 'DTLZ2' or problem.name() == 'DTLZ3'\
            or problem.name() == 'DTLZ4':
        ref_dir = get_uniform_weights(n, n_obj)
        return problem.pareto_front(ref_dir)
    else:
        return problem.pareto_front(n_pareto_points=n)

def plot_process(ax, problem, train_y, norm_train_y, denormalize, idealsearch, model, train_x):
    ss = 16
    true_pf = get_paretofront(problem, 1000)
    # ---------- visual check
    ax.cla()
    ax.scatter(true_pf[:, 0], true_pf[:, 1], c='green', s=0.2)
    ax.scatter(train_y[:, 0], train_y[:, 1], c='blue', s=1)
    nd_front = get_ndfront(norm_train_y)
    nd_frontdn = denormalize(nd_front, train_y)
    ax.scatter(nd_frontdn[:, 0], nd_frontdn[:, 1], c='blue')

    # plot reference point
    ref = [1.1] * train_y.shape[1]
    ref_dn = denormalize(ref, train_y)
    ax.scatter(ref_dn[0], ref_dn[1], c='green', marker='D')

    bottom, top = ax.get_ylim()
    top = top + 3
    # bottom =  bottom - 0.4
    ax.set_ylim(bottom, top)
    plt.legend(['PF', 'archive A', 'nd front', 'ref point'], fontsize=ss, ncol=2, handleheight=2.4, labelspacing=0.005)
    # plt.legend(['PF',  'nd front', 'ref point'])
    # -----------visual check--

    if idealsearch:
        # search ideal then plot what is searched
        ax.scatter(train_y[-2:, 0], train_y[-2:, 1], c='red', marker='X', s=100)
        # if ideal search is conducted
        # also plots the estimated
        pred_y1, _ = model[0].predict(train_x[-2:, :])
        pred_y2, _ = model[1].predict(train_x[-2:, :])
        pred_y = np.hstack((pred_y1, pred_y2))
        pred_y = denormalize(pred_y, train_y)
        # ax.scatter(pred_y[:, 0], pred_y[:, 1], c='black', marker=7, s=100)
        bottom, top = ax.get_ylim()
        top = top + 3
        # bottom =  bottom - 0.4
        ax.set_ylim(bottom, top)
        # plt.legend(['PF', 'nd front', 'ref point', 'ideal search', 'estimates'])
        # plt.legend(['PF', 'archive A', 'nd front', 'ref point', 'extreme points', 'surrogate minima'], fontsize=ss-1, ncol=2, handleheight=2.4, labelspacing=0.005)
        plt.legend(['PF', 'archive A', 'nd front', 'ref point', 'extreme points', 'surrogate minima'], fontsize=ss,
                   ncol=2, handleheight=2.4, labelspacing=0.005)

    ax.set_title(problem.name(),fontsize=ss)
    ideal = np.min(nd_frontdn, axis=0)
    nadir = np.max(nd_frontdn, axis=0)
    line1 = [ideal[0], nadir[0], nadir[0], ideal[0], ideal[0]]
    line2 = [ideal[1], ideal[1], nadir[1], nadir[1], ideal[1]]
    line = Line2D(line1, line2, linestyle='--', c='orange')
    ax.add_line(line)

    # reference line horizon
    left, right = ax.get_xlim()
    line_hz1 = [left, ref_dn[0]]
    line_hz2 = [ref_dn[1], ref_dn[1]]
    line = Line2D(line_hz1, line_hz2, linestyle='--', c='black')
    ax.add_line(line)

    # reference line vertical
    bottom, top = ax.get_ylim()
    line_v1 = [ref_dn[0], ref_dn[0]]
    line_v2 = [bottom, ref_dn[1]]
    line = Line2D(line_v1, line_v2, linestyle='--', c='black')
    ax.add_line(line)

    ax.set_xlabel('f1', fontsize=ss)
    ax.set_ylabel('f2', fontsize=ss, rotation='horizontal')

    # bottom, top = ax.get_ylim()
    # top = top + 4
    # ax.set_ylim(bottom, top)
    plt.pause(5)


    # -----

    path = os.getcwd()
    savefolder = path + '\\paper1_results\\process_plot'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + '_process_nd_' + str(idealsearch) + '.eps'
    savename2 = savefolder + '\\' + problem.name() + '_process_nd_' + str(idealsearch) + '.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)
    plt.close()





def plot_process3d(ax, problem, train_y, norm_train_y, denormalize, idealsearch, model, train_x):
    true_pf = get_paretofront(problem, 1000)
    # ---------- visual check
    ax.cla()
    ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], c='green')
    # ax.scatter(train_y[:, 0], train_y[:, 1], train_y[:, 2], c='blue', s=0.2)
    nd_front = get_ndfront(norm_train_y)
    nd_frontdn = denormalize(nd_front, train_y)
    ax.scatter(nd_frontdn[:, 0], nd_frontdn[:, 1], nd_frontdn[:, 2], c='blue')

    # plot reference point
    ref = [1.1] * train_y.shape[1]
    ref_dn = denormalize(ref, train_y)
    ax.scatter(ref_dn[0], ref_dn[1], ref_dn[2], c='red', marker='D')

    # plt.legend(['PF', 'archive A', 'nd front', 'ref point'])
    plt.legend(['PF',  'nd front', 'ref point'])
    # -----------visual check--

    if idealsearch:
        # search ideal then plot what is searched
        ax.scatter(train_y[-3:, 0], train_y[-3:, 1], train_y[-3:, 2], c='red', marker='X', s=100)
        # if ideal search is conducted
        # also plots the estimated
        pred_y1, _ = model[0].predict(train_x[-3:, :])
        pred_y2, _ = model[1].predict(train_x[-3:, :])
        pred_y3, _ = model[2].predict(train_x[-3:, :])
        pred_y = np.hstack((pred_y1, pred_y2, pred_y3))
        pred_y = denormalize(pred_y, train_y)
        ax.scatter(pred_y[:, 0], pred_y[:, 1], pred_y[:, 2], c='black', marker=7, s=100)
        plt.legend(['PF','nd front', 'ref point', 'ideal search', 'estimates'])
        # plt.legend(['PF', 'archive A', 'nd front', 'ref point', 'ideal search', 'estimates'])

    plt.title(problem.name())

    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    plt.pause(1)

    # -----
    '''
    path = os.getcwd()
    savefolder = path + '\\paper1_results\\process_plot'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + '_process_ndr_step1.eps'
    savename2 = savefolder + '\\' + problem.name() + '_process_ndr_step1.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)
    '''


def hv_converge(target_problem, train_y):
    '''
    every iteration, this function takes pf and nd_front, and return two values of [hv_pf, hv_nd]
    :param target_problem:  used to generate pareto front
    :param train_y:  used to generate nd front
    :return: hv_pf, hv_nd
    '''
    pf = get_paretofront(target_problem, 100)
    nd = get_ndfront(train_y)
    pf_endmax = np.max(pf, axis=0)
    pf_endmin = np.min(pf, axis=0)
    pf = (pf - pf_endmin) / (pf_endmax - pf_endmin)
    nd = (nd - pf_endmin) / (pf_endmax - pf_endmin)
    ref = [1.1] * train_y.shape[1]

    pf_hv = gethv(pf, ref)
    nd_hv = gethv(nd, ref)
    return pf_hv, nd_hv

def gethv(front, ref):
    # front needs to be processed first to eliminate points beyond ref
    n = front.shape[0]
    n_obj = front.shape[1]
    newfront = []
    for i in range(n):
        if np.any(front[i, :] >= ref):
            continue
        else:
            newfront = np.append(newfront, front[i, :])
    newfront = np.atleast_2d(newfront).reshape(-1, n_obj)
    if len(newfront) > 0:
        hv_class = pg.hypervolume(newfront)
        return hv_class.compute(ref)
    else:
        return 0.0


def paper1_mainscript(seed_index, target_problem, method_selection, search_ideal, max_eval, num_pop, num_gen, visual):
    '''
    :param seed_index:
    :param target_problem:
    :param method_selection: function name string, normalization scheme
    :param ideal_search: whether to use kriging to search for ideal point
    :return:
    '''
    # steps
    # (1) init training data with number of initial_samples
    # (2) normalization on f
    # (3) train krg
    # (4) enter iteration, propose next x till number of iteration is met
    # (5) save nd front under name \problem_method_i\nd_seed_1.csv
    # (6) save hv converge  under name \problem_method_i\hvconvg_seed_1.csv

    enable_crossvalidation = False
    mp.freeze_support()
    np.random.seed(seed_index)

    target_problem = eval(target_problem)
    print('Problem %s, seed %d' % (target_problem.name(), seed_index))
    hv_ref = [1.1, 1.1]

    if visual:
        plt.ion()
        figure, ax = plt.subplots()

    # collect problem parameters: number of objs, number of constraints
    n_vals = target_problem.n_var
    if 'WFG' in target_problem.name():
        number_of_initial_samples = 200
    else:
        number_of_initial_samples = 11 * n_vals - 1

    n_iter = max_eval - number_of_initial_samples  # stopping criterion set

    pf_nd = []  # analysis parameter, due to search_ideal, size is un-determined

    # (1) init training data with number of initial_samples
    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem, seed_index)
    # plot_initpop(train_y, target_problem, method_selection, search_ideal, seed_index)
    # (2) normalization scheme
    norm_scheme = eval(method_selection)
    norm_train_y = norm_scheme(train_y)
    denormalize_funcname = 'de' + method_selection
    denormalize = eval(denormalize_funcname)

    # (3) train krg
    krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
    krg1 = copy.deepcopy(krg)

    # (4-0) before enter propose x phase, conduct once krg search on ideal
    if search_ideal:
        train_x, train_y = idealsearch_update(train_x, train_y, krg, target_problem)
        norm_train_y = norm_scheme(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        if visual:
            plot_process(ax, target_problem, train_y, norm_train_y, denormalize, True, krg1, train_x)
            return



    # (4) enter iteration, propose next x till number of iteration is met
    ego_eval = EI.ego_fit(target_problem.n_var, target_problem.n_obj, target_problem.n_constr, target_problem.xu, target_problem.xl,target_problem.name())

    for iteration in range(n_iter):
        print('iteration %d' % iteration)
        # (4-1) de search for proposing next x point
        # visual check
        if visual:
            plot_process(ax, target_problem, train_y, norm_train_y, denormalize, False, krg, train_x)
            # return
        # use my own DE faster
        nd_front = get_ndfront(norm_train_y)
        ego_evalpara = {'krg': krg, 'nd_front': nd_front, 'ref': hv_ref,  # ego search parameters
                        'denorm': denormalize, 'normdata': train_y,       # ego search plot parameters
                        'pred_model': krg, 'real_prob': target_problem,   # ego search plot parameter
                        'ideal_search': search_ideal, 'seed': seed_index, 'method': method_selection}  # plot save params
        bounds = np.vstack((target_problem.xl, target_problem.xu)).T.tolist()
        insertpop = get_ndfrontx(train_x, norm_train_y)

        visualplot = True
        ax = None
        next_x, _, _, _ = optimizer_EI.optimizer_DE(ego_eval, ego_eval.n_constr, bounds,
                                                    insertpop, 0.8, 0.8, num_pop, num_gen,
                                                    visualplot, ax, **ego_evalpara)
        # propose next_x location
        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)
        next_y = target_problem.evaluate(next_x, return_values_of=['F'])

        #---visual check
        if visual:
            process_visualcheck(ax, next_x, next_y, krg, denormalize, train_y)

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))

        # analysis parameter, always follow np.vstack
        pf_hv, nd_hv = hv_converge(target_problem, train_y)
        pf_nd = np.append(pf_nd, pf_hv)
        pf_nd = np.append(pf_nd, nd_hv)

        # count evaluation and break
        if train_y.shape[0] >= max_eval:
            train_x = train_x[0:max_eval, :]
            train_y = train_y[0:max_eval, :]
            break



        # (4-2) according to configuration determine whether to estimate new point
        if search_ideal:
            if confirm_search(next_y, train_y[0:-1, :]):
                print('ideal search')
                train_x, train_y = idealsearch_update(train_x, train_y, krg, target_problem)
                # analysis parameter, always follow np.vstack
                pf_hv, nd_hv = hv_converge(target_problem, train_y)
                pf_nd = np.append(pf_nd, pf_hv)
                pf_nd = np.append(pf_nd, nd_hv)
                if visual:
                    plot_process(ax, target_problem, train_y, norm_train_y, denormalize, True, krg, train_x)

        # retrain krg, normalization needed
        norm_train_y = norm_scheme(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)



    # (5) save nd front under name \problem_method_i\nd_seed_1.csv
    # (6) save hv converge  under name \problem_method_i\hvconvg_seed_1.csv
    nd2csv(train_y, target_problem, seed_index, method_selection, search_ideal)
    pfnd2csv(pf_nd, target_problem, seed_index, method_selection, search_ideal)

def process_visualcheck(ax, next_x, next_y, krg, denormalize, train_y):
    '''
    process visual next
    plot predicted next point f and real next point f on given ax
    '''
    ax.scatter(next_y[:, 0], next_y[:, 1], c='orange', marker='X')
    pred_y1, _ = krg[0].predict(next_x)
    pred_y2, _ = krg[1].predict(next_x)
    pred_y = denormalize(np.hstack((pred_y1, pred_y2)), train_y)
    ax.scatter(pred_y[:, 0], pred_y[:, 1], marker=7, c='black')
    plt.pause(2)

def paper1_mainscript3d(seed_index, target_problem, method_selection, search_ideal, max_eval, num_pop, num_gen):
    '''
        :param seed_index:
        :param target_problem:
        :param method_selection: function name string, normalization scheme
        :param ideal_search: whether to use kriging to search for ideal point
        :return:
        '''
    # steps
    # (1) init training data with number of initial_samples
    # (2) normalization on f
    # (3) train krg
    # (4) enter iteration, propose next x till number of iteration is met
    # (5) save nd front under name \problem_method_i\nd_seed_1.csv
    # (6) save hv converge  under name \problem_method_i\hvconvg_seed_1.csv

    enable_crossvalidation = False
    mp.freeze_support()
    np.random.seed(seed_index)

    target_problem = eval(target_problem)
    print('Problem %s, seed %d' % (target_problem.name(), seed_index))
    hv_ref = [1.1, 1.1, 1.1]

    # step(0) get ready to plot
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # collect problem parameters: number of objs, number of constraints
    n_vals = target_problem.n_var
    if 'WFG' in target_problem.name():
        number_of_initial_samples = 200
    else:
        number_of_initial_samples = 11 * n_vals - 1

    n_iter = max_eval - number_of_initial_samples  # stopping criterion set
    pf_nd = []  # analysis parameter, due to search_ideal, size is un-determined

    # (1) init training data with number of initial_samples
    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem, seed_index)
    # plot_initpop(train_y, target_problem, method_selection, search_ideal, seed_index)

    # (2) normalization scheme
    norm_scheme = eval(method_selection)
    norm_train_y = norm_scheme(train_y)
    denormalize_funcname = 'de' + method_selection
    denormalize = eval(denormalize_funcname)

    # (3) train krg
    krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    # (4-0) before enter propose x phase, conduct once krg search on ideal
    if search_ideal:
        train_x, train_y = idealsearch_update(train_x, train_y, krg, target_problem)
        norm_train_y = norm_scheme(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        # plot_process3d(ax, target_problem, train_y, norm_train_y, denormalize, True, krg, train_x)
        # return

    # (4) enter iteration, propose next x till number of iteration is met
    ego_eval = EI.ego_fit(target_problem.n_var, target_problem.n_obj, target_problem.n_constr, target_problem.xu,
                          target_problem.xl, target_problem.name())

    for iteration in range(n_iter):
        print('iteration %d' % iteration)
        # (4-1) de search for proposing next x point
        # visual check

        # plot_process3d(ax, target_problem, train_y, norm_train_y, denormalize, False, krg, train_x)

        # use my own DE faster
        nd_front = get_ndfront(norm_train_y)
        ego_evalpara = {'krg': krg, 'nd_front': nd_front, 'ref': hv_ref,  # ego search parameters
                        'denorm': denormalize, 'normdata': train_y,  # ego search plot parameters
                        'pred_model': krg, 'real_prob': target_problem,  # ego search plot parameter
                        'ideal_search': search_ideal, 'seed': seed_index,
                        'method': method_selection}  # plot save params
        bounds = np.vstack((target_problem.xl, target_problem.xu)).T.tolist()
        insertpop = get_ndfrontx(train_x, norm_train_y)

        visualplot = False
        ax = None

        next_x, _, _, _ = optimizer_EI.optimizer_DE(ego_eval, ego_eval.n_constr, bounds,
                                                    insertpop, 0.8, 0.8, num_pop, num_gen,
                                                    visualplot, ax, **ego_evalpara)
        # propose next_x location
        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)
        next_y = target_problem.evaluate(next_x, return_values_of=['F'])

        # --------visual check
        '''
        ax.scatter(next_y[:, 0], next_y[:, 1],next_y[:, 2], c='orange', marker='X')
        pred_y1, _ = krg[0].predict(next_x)
        pred_y2, _ = krg[1].predict(next_x)
        pred_y3, _ = krg[2].predict(next_x)
        pred_y = denormalize(np.hstack((pred_y1, pred_y2, pred_y2)), train_y)
        ax.scatter(pred_y[:, 0], pred_y[:, 1], pred_y[:, 2], marker=7, c='black')
        plt.pause(1)
        '''
        # ------------visual check

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))

        # analysis parameter, always follow np.vstack
        pf_hv, nd_hv = hv_converge(target_problem, train_y)
        pf_nd = np.append(pf_nd, pf_hv)
        pf_nd = np.append(pf_nd, nd_hv)

        # count evaluation and break
        if train_y.shape[0] >= max_eval:
            train_x = train_x[0:max_eval, :]
            train_y = train_y[0:max_eval, :]

        # (4-2) according to configuration determine whether to estimate new point
        if search_ideal:
            if confirm_search(next_y, train_y[0:-1, :]):
                print('ideal search')
                train_x, train_y = idealsearch_update(train_x, train_y, krg, target_problem)
                # analysis parameter, always follow np.vstack
                pf_hv, nd_hv = hv_converge(target_problem, train_y)
                pf_nd = np.append(pf_nd, pf_hv)
                pf_nd = np.append(pf_nd, nd_hv)
                # plot_process3d(ax, target_problem, train_y, norm_train_y, denormalize, True, krg, train_x)

        # retrain krg, normalization needed
        norm_train_y = norm_scheme(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    # (5) save nd front under name \problem_method_i\nd_seed_1.csv
    # (6) save hv converge  under name \problem_method_i\hvconvg_seed_1.csv
    nd2csv(train_y, target_problem, seed_index, method_selection, search_ideal)
    pfnd2csv(pf_nd, target_problem, seed_index, method_selection, search_ideal)


def single_run():
    import json
    # problems_json = 'p/zdt_problems_hvnd.json'
    problems_json = 'p/zdt_problems_hvndr.json'
    # problems_json = 'p/zdt_problems_hv.json'

    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['MO_target_problems']
    method_selection = hyp['method_selection']
    search_ideal = hyp['search_ideal']
    max_eval = hyp['max_eval']
    num_pop = hyp['num_pop']
    num_gen = hyp['num_gen']

    target_problem = target_problems[0]
    target_problems = "ZDT1(n_var=6)"
    method_selection = "normalization_with_nd"
    seed_index = 8
    paper1_mainscript(seed_index, target_problem, method_selection, search_ideal, max_eval, num_pop, num_gen, visual)
    return None

def para_run():
    import json
    problems_json = ['p/dtlz_problems_hv.json',
                     'p/dtlz_problems_hvnd.json',
                     'p/dtlz_problems_hvndr.json',
                      'p/zdt_problems_hv.json',
                      'p/zdt_problems_hvnd.json',
                      'p/zdt_problems_hvndr.json',
                     ]
    args = []
    seedmax = 29
    for problem_setting in problems_json:
        with open(problem_setting, 'r') as data_file:
            hyp = json.load(data_file)
        target_problems = hyp['MO_target_problems']
        method_selection = hyp['method_selection']
        search_ideal = hyp['search_ideal']
        max_eval = hyp['max_eval']
        num_pop = hyp['num_pop']
        num_gen = hyp['num_gen']
        for problem in target_problems:
            for seed in range(seedmax):
                args.append((seed, problem, method_selection, search_ideal, max_eval, num_pop, num_gen, False))

    num_workers = 14
    pool = mp.Pool(processes=num_workers)
    pool.starmap(paper1_mainscript, ([arg for arg in args]))

    return None

def plot_run():
    import json
    problems_json = 'p/resconvert.json'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['MO_target_problems']

    method_selection = ['normalization_with_self', 'normalization_with_nd', 'normalization_with_nd']
    search_ideal = 0
    max_eval = 250
    num_pop = 100
    num_gen = 100
    seed_index = 1
    i = 13
    # for i in range(4, 9):
    paper1_mainscript(seed_index, target_problems[i], method_selection[1], search_ideal, max_eval, num_pop, num_gen)

def p3d_run():
    target_problems = ["DTLZ3(n_var=6, n_obj=3)"]
    method_selection = ['normalization_with_self', 'normalization_with_nd', 'normalization_with_nd']
    search_ideal = 1
    max_eval = 100
    num_pop = 100
    num_gen = 100
    seed_index = 1
    paper1_mainscript3d(seed_index, target_problems[0], method_selection[1], search_ideal, max_eval, num_pop, num_gen)


def p3d_pararun():
    target_problems = ["DTLZ1(n_var=6, n_obj=3)", "DTLZ2(n_var=6, n_obj=3)", "DTLZ3(n_var=6, n_obj=3)"]
    method_selections = ['normalization_with_self', 'normalization_with_nd']
    search_ideal = 0
    max_eval = 250
    num_pop = 100
    num_gen = 100
    seedmax = 29


    args = []

    for problem in target_problems:
        for method_selection in method_selections:
            for seed in range(seedmax):
                args.append((seed, problem, method_selection, search_ideal, max_eval, num_pop, num_gen))

    search_ideal = 1
    for problem in target_problems:
        for seed in range(seedmax):
            args.append((seed, problem, method_selections[1], search_ideal, max_eval, num_pop, num_gen))

    num_workers = 14
    pool = mp.Pool(processes=num_workers)
    pool.starmap(paper1_mainscript3d, ([arg for arg in args]))

def vary_egofit():
    print(0)


if __name__ == "__main__":
    # p3d_pararun()
    # p3d_run()
    plot_run()

    # single_run()
    # para_run()
