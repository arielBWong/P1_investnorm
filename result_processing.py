import numpy as np
import optimizer
from joblib import dump, load
import os
import pygmo as pg


def reverse_zscore(data, m, s):
    return data * s + m



def compare_somg():
    # single objective
    # multiple constraints

    diff = 0
    # problem_list = ['Gomez3', 'new_branin_5', 'Mystery', 'ReverseMystery', 'SHCBc', 'Haupt_schewefel', 'HS100', 'GPc']
    problem_list = ['HS100']
    problem_diff = {}
    for problem in problem_list:
        output_folder_name = 'outputs\\' + problem
        if os.path.exists(output_folder_name):
            # f_opt_name = output_folder_name + '\\' + problem + '.txt'
            # f_opt = genfromtxt(f_opt_name)
            diff = 0
            count = 0
            for output_index in range(20):
                output_f_name = output_folder_name + '\\' + 'best_f_seed_' + str(output_index) + '.joblib'
                output_x_name = output_folder_name + "\\" + 'best_x_seed_' + str(output_index) + '.joblib'
                best_f = load(output_f_name)
                best_x = load(output_x_name)
                print(best_f)
                print(best_x)
                # if os.path.exists(output_f_name):
                # best_f = load(output_f_name)
                # diff = np.abs(best_f - f_opt)
                # diff = diff + best_f
                # count = count + 1
        # print(problem)
        # print('f difference')
        # print(diff/count)
        # print(count)
        # problem_diff[problem] = diff/count

    # import json
    # with open('f_diff.json', 'w') as file:
    # file.write(json.dumps(problem_diff))


def ego_outputs_read(prob):
    output_folder_name = 'outputs\\' + prob
    output_f_name = output_folder_name + '\\' + 'best_f_seed_' + str(100) + '.joblib'
    best_f = load(output_f_name)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(best_f)
    ndf = list(ndf)
    f_pareto = best_f[ndf[0], :]
    test_f = np.sum(f_pareto, axis=1)
    return f_pareto


def nsga2_outputs_read(prob):
    nsga_problem_save = 'NSGA2\\' + prob + '\\' + 'pareto_f.joblib'
    f_pareto2 = load(nsga_problem_save)
    return f_pareto2


def compare_save_ego2nsga(problem_list):
    save_compare = np.atleast_2d([0, 0])
    for p in problem_list:
        problem = p
        f_pareto = ego_outputs_read(problem)
        f_pareto2 = nsga2_outputs_read(problem)

        point_list = np.vstack((f_pareto, f_pareto2))
        point_nadir = np.max(point_list, axis=0)
        point_reference = point_nadir * 1.1

        hv_ego = pg.hypervolume(f_pareto)
        hv_nsga = pg.hypervolume(f_pareto2)

        hv_value_ego = hv_ego.compute(point_reference)
        hv_value_nsga = hv_nsga.compute(point_reference)

        new_compare = np.atleast_2d([hv_value_ego, hv_value_nsga])
        save_compare = np.vstack((save_compare, new_compare))

    save_compare = np.delete(save_compare, 0, 0).reshape(-1, 2)
    print(save_compare)
    with open('mo_compare.txt', 'w') as f:
        for i, p in enumerate(problem_list):
            f.write(p)
            f.write('\t')
            f.write(str(save_compare[i, 0]))
            f.write('\t')
            f.write(str(save_compare[i, 1]))
            f.write('\n')


if __name__ == "__main__":
    problem_list = ['DTLZ2']
    compare_save_ego2nsga(problem_list)




















    '''
    target_problem = branin.new_branin_5()
    number_of_initial_samples = 1000
    n_vals = target_problem.n_var

    sample_x = pyDOE.lhs(n_vals, number_of_initial_samples)
    sample_x = target_problem.hyper_cube_sampling_convert(sample_x)
    sample_y, sample_g = target_problem.evaluate(sample_x)

    mse_f_collect = 0
    mse_g_collect = 0
    for output_index in range(1):
        output_file_name = 'sample_x_seed_ ' + str(output_index) + '.joblib'
        train_x = load(output_file_name)

        out = {}
        train_y, cons_y = target_problem._evaluate(train_x, out)


        # normalize
        mean_cons_y, std_cons_y, norm_cons_y = norm_data(cons_y)
        mean_train_y, std_train_y, norm_train_y = norm_data(train_y)
        mean_train_x, std_train_x, norm_train_x = norm_data(train_x)

        # cross-validation:


        gpr, gpr_g = cross_val_hyperp. cross_val_gpr(norm_train_x, norm_train_y, norm_cons_y)
        n_obj = target_problem.n_obj
        n_cons = target_problem.n_constr

        # on prediction
        norm_sample_x = (sample_x - mean_train_x)/std_train_x
        norm_sample_y = (sample_y - mean_train_y)/std_train_y
        norm_sample_g = (sample_g - mean_cons_y)/std_cons_y

        mse_f = 0
        for j in range(n_obj):
            pred_y_norm = gpr[j].predict(norm_sample_x)
            pred_y = reverse_zscore(pred_y_norm, mean_train_y, std_train_y)
            mse_f = mse_f + mean_squared_error(pred_y, sample_y)

        mse_g = 0
        for j in range(n_cons):
            pred_g_norm = gpr_g[j].predict(norm_sample_x)
            pred_g = reverse_zscore(pred_g_norm, mean_cons_y, std_cons_y)
            mse_g = mse_g + mean_squared_error(pred_g, sample_g)

        mse_f_collect = mse_f_collect + mse_f
        mse_g_collect = mse_g_collect + mse_g


    print('mse of average f')
    print(mse_f_collect/20.0)

    print('mse of average g')
    print(mse_g_collect/20.0)
    '''







