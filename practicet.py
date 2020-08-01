import numpy as np
import scipy
import EGO_krg
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
    DTLZ1, DTLZ2, \
    BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK


def test_lexsort_specify_baserow():
    a = np.atleast_2d([1, 5, 1, 4, 3, 4, 4])
    b = np.atleast_2d([0, 4, 0, 4, 0, 2, 1])

    c = np.vstack((b, a))

    # n = EGO_krg.lexsort_specify_baserow(c, 0)
    n = EGO_krg.lexsort_with_certain_row(c, 0)
    print(n)


def test_confirm_search():
    y = np.array([[1, 2], [-1, 1], [2, 0]])
    x = np.array([3, 0])
    s = EGO_krg.confirm_search(x, y[:, :])
    print(s)


def test_normalization_with_nothing():
    a = np.atleast_2d([[1, 2], [3, 4]])
    func = 'EGO_krg.normalization_with_self'
    b = eval(func)
    print(b(a))


def test_scipyde():
    # -----------------------
    # scipy de is too slow
    # def obj(x):
    #     fitness = EI_problem.ego_believer(x, krg, nd_front, hv_ref)
    #     fitness = fitness.flatten()
    #     return fitness
    # bounds = Bounds(lb=target_problem.xl, ub=target_problem.xu)
    # result = differential_evolution(obj, bounds, maxiter=num_gen, popsize=num_pop, mutation=0.8, recombination=0.8, seed=seed_index )
    # next_x = result.x
    # -----------------
    return None


if __name__ == "__main__":
    print(0)

    y = np.array([[1, 2], [-1, 1], [2, 0]])
    print(y[0:2, :])

