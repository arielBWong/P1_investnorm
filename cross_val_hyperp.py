import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error
import multiprocessing as mp
from pymop.factory import get_problem_from_func

def external_optimizer(obj_func, initial_theta, bounds):



    upper_bound = np.ones(n_variables)
    lower_bound = np.ones(n_variables) * -1

    hyper_p_problem = get_problem_from_func(obj_func,
                                            lower_bound,
                                            upper_bound,
                                            n_var=n_variables,
                                            func_args=evalparas)



    return theta_opt, func_min




def cross_val_mse_para(train_x, train_y, val_x, val_y):
    val_x = check_array(val_x)
    val_y = check_array(val_y)

    train_x = check_array(train_x)
    train_y = check_array(train_y)

    # fit GPR
    # kernal initialization should also use external configuration
    kernel = RBF(1, (np.exp(-1), np.exp(3)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)

    gpr.fit(train_x, train_y)

    pred_y = gpr.predict(val_x)
    mse = mean_squared_error(val_y, pred_y)

    return mse

def recreate_gpr(fold_id, k_fold, fold_size, shuffle_index, train_x, train_y):
    # input train_x and train_y has been shuffled already
    # use this fold index to re-create the best gpr

    n_samples = train_x.shape[0]
    if fold_id != k_fold - 1:
        sep_front = fold_id * fold_size
        sep_back = (fold_id + 1) * fold_size
    else:
        sep_front = fold_id * fold_size
        sep_back = n_samples - 1

    temp_x = train_x
    temp_y = train_y

    # recover the training data
    train_fold_x = np.delete(temp_x, range(sep_front, sep_back), axis=0)
    train_fold_y = np.delete(temp_y, range(sep_front, sep_back), axis=0)

    # fit GPR
    # kernal initialization should also use external configuration
    kernel = RBF(1, (np.exp(-1), np.exp(3)))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)
    gpr.fit(train_fold_x, train_fold_y)

    return gpr


def n_fold_cross_val_para(train_x, train_y, cons_y):

    # set up pool
    # number of processors probably need to be configurable
    pool = mp.Pool(processes=1)

    # the caller of this method has checked np.atleast_2d on variables
    # so no more check array needed
    n_samples = train_x.shape[0]
    n_sur_objs = train_y.shape[1]
    n_sur_cons = cons_y.shape[1]

    # this n-fold probably needs some configuration
    n = 5

    # in case there is zero fold_size outcome
    if n > n_samples:
        fold_size = 1
        # deal with situations (1 variable problem) where n_samples are not enough
        # change n fold cross-validation to leave-one-out cross validation
        n = n_samples
    else:
        # what is left over is included in the last folder
        fold_size = int(n_samples / n)

    # do I have to shuffle every time I extract a fold?
    # this is left for future change
    # yes first, let us shuffle the sample data
    index_samples = np.arange(n_samples)
    np.random.shuffle(index_samples)

    train_x = train_x[index_samples, :]
    train_y = train_y[index_samples, :]
    cons_y = cons_y[index_samples, :]

    mse_list = []
    mse_g_list = []
    results = []
    results_map = []
    results_g = []
    results_g_map = []


    for i in range(n):

        temp_x = train_x
        temp_y = train_y
        temp_g = cons_y

        # decide the index range that is used as validation set
        if i != n - 1:
            sep_front = i * fold_size
            sep_back = (i + 1) * fold_size
        else:
            sep_front = i * fold_size
            sep_back = n_samples

        # select validation set
        val_fold_x = train_x[sep_front: sep_back, :]
        val_fold_y = train_y[sep_front: sep_back, :]
        val_fold_g = cons_y[sep_front: sep_back, :]

        # select train set
        train_fold_x = np.delete(temp_x, range(sep_front, sep_back), axis=0)
        train_fold_y = np.delete(temp_y, range(sep_front, sep_back), axis=0)
        train_fold_g = np.delete(temp_g, range(sep_front, sep_back), axis=0)

        # generate jobs for pool
        # results.append(pool.apply_async(cross_val_mse_para, (train_fold_x, train_fold_y, val_fold_x, val_fold_y)))
        # results_g.append(pool.apply_async(cross_val_mse_para, (train_fold_x, train_fold_g, val_fold_x, val_fold_g)))

        obj_data_split = []
        for j in range(n_sur_objs):
            one_obj_y = np.atleast_2d(train_fold_y[:, j]).reshape(-1, 1)
            one_obj_y_val = np.atleast_2d(val_fold_y[:, j]).reshape(-1, 1)
            obj_data_split.append((train_fold_x, one_obj_y, val_fold_x, one_obj_y_val))
        results_map.append(pool.starmap(cross_val_mse_para,
                                        ([para_tuple for para_tuple in obj_data_split])))

        # train for constraints
        # later  convert results_g back to n fold * n_sur_cons matrix
        cons_data_split = []
        for j in range(n_sur_cons):
            one_cons_g = np.atleast_2d(train_fold_g[:, j]).reshape(-1, 1)
            one_cons_g_val = np.atleast_2d(val_fold_g[:, j]).reshape(-1, 1)
            cons_data_split.append((train_fold_x, one_cons_g, val_fold_x, one_cons_g_val))
        results_g_map.append(pool.starmap(cross_val_mse_para,
                                          ([para_tuple for para_tuple in cons_data_split])))

    pool.close()
    pool.join()

    # recreate n * n_sur_objs matrix for multiple-objective compatible
    results_obj_map = np.array(results_map).reshape(n, n_sur_objs)
    mse_min_index = np.argmin(results_obj_map, 0)

    results_g_map = np.array(results_g_map).reshape(n, n_sur_cons)
    mse_min_g_index = np.argmin(results_g_map, 0)

    gpr = []
    for i in range(n_sur_objs):
        gpr.append(recreate_gpr(mse_min_index[i], n, fold_size, index_samples, train_x, train_y))
    # gpr = recreate_gpr(min_fold_index, n, fold_size, index_samples, train_x, train_y)

    gpr_g = []
    for i in range(n_sur_cons):
        gpr_g.append(recreate_gpr(mse_min_g_index[i], n, fold_size, index_samples, train_x, cons_y))
    # gpr_g = recreate_gpr(min_fold_g, n, fold_size, index_samples, train_x, cons_y)

    return gpr, gpr_g


def cross_val_gpr(train_x, train_y, cons_y):

    # inputs are normalized variables
    train_x = check_array(train_x)
    train_y = check_array(train_y)
    cons_y = check_array(cons_y)
    # gpr = n_fold_cross_val(train_x, train_y)
    gpr, gpr_g = n_fold_cross_val_para(train_x, train_y, cons_y)
    return gpr, gpr_g

