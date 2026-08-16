# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:43:20 2025

@author: emilia
"""

#!pip install composites
#!pip install bfscplate2d
#!pip install pymoo
import sys
sys.path.append('..')

import numpy as np
import scipy.optimize as opt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from composites import laminated_plate
from composites.utils import n_double_plate, read_laminaprop

from bfscplate2d import (BFSCPlate2D, update_KC0, update_KG, update_KG_cte_N,
        DOF, KC0_SPARSE_SIZE, KG_SPARSE_SIZE, DOUBLE, INT)
from bfscplate2d.quadrature import get_points_weights


# Table 01 Haftka's 1993 paper, first row of that table
a_value = 20. # [in]
b_value = 5. # [in]
load_Nxx = 1. # [lb]
load_Nyy = 0 # [lb]
tmp = [90]*2 + [+45,-45]*2 + [90]*2 + [+45, -45] + [90]*2 + [+45, -45]*6
ref_stack = tmp + tmp[::-1]

assert len(ref_stack) == 48

# material properties
E11 = 127.55e9 # [Pa]
E22 = 13.03e9
nu12 = 0.3
G12 = 6.41e9
G13 = 6.41e9
G23 = 6.41e9
laminaprop = (E11, E22, nu12, G12, G13, G23)
ply_thickness = 0.005*25.4/1000 # [m]
h = ply_thickness*len(ref_stack)

ref_prop = laminated_plate(stack=ref_stack, plyt=ply_thickness, laminaprop=laminaprop)
mat = read_laminaprop(laminaprop)
tr = mat.q11 + mat.q22 + 2*mat.q66
A11s_LQL = ref_prop.A11/tr/h
A22s_LQL = ref_prop.A22/tr/h
A66s_LQL = ref_prop.A66/tr/h

assert np.isclose(ref_prop.h, h)


def calc_failure_load_Haftka(prop, lbd0=100):
    def strain_MS(lbd):
        Nxx = -load_Nxx*4.448222/(25.4/1000) #[N/m]
        Nyy = -load_Nyy*4.448222/(25.4/1000) #[N/m]
        Nxx = lbd*Nxx*1.5
        Nyy = lbd*Nyy*1.5
        vecN = np.asarray([Nxx, Nyy])
        A11 = prop.ABD[0, 0]
        A12 = prop.ABD[0, 1]
        A22 = prop.ABD[1, 1]
        exx, eyy = np.linalg.inv(np.array([[A11, A12], [A12, A22]])) @ vecN
        # NOTE allowable strains from Haftka's 1993 paper
        epsilon_1_allowable = 0.008
        epsilon_2_allowable = 0.029
        gamma_12_allowable = 0.015
        MS = 1e15
        for thetadeg in prop.stack:
            cost = np.cos(np.deg2rad(thetadeg))
            sint = np.sin(np.deg2rad(thetadeg))
            epsilon_i_1 = cost**2*exx + sint**2*eyy
            epsilon_i_2 = sint**2*exx + cost**2*eyy
            gamma_i_12 = sint**2*(eyy - exx)
            MSnew = min(
                    epsilon_1_allowable/abs(epsilon_i_1) - 1,
                    epsilon_2_allowable/abs(epsilon_i_2) - 1,
                    gamma_12_allowable/abs(gamma_i_12) - 1
                    )
            MS = min(MS, MSnew)
        return MS
    positiveMS = opt.NonlinearConstraint(strain_MS, 0., np.inf, jac='2-point')
    res = opt.minimize(strain_MS, lbd0, tol=1e-6, bounds=((100, None),),
            constraints=[positiveMS], jac='2-point')
    assert res.success
    return res.x[0]


def calc_buckling(prop):
    # number of nodes
    nx = 15 # along x
    ny = 15 # along y

    # getting integration points
    points, weights = get_points_weights(nint=4)

    # geometry
    a = a_value*25.4/1000 # [m] along x
    b = b_value*25.4/1000 # [m] along y

    # creating mesh
    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(x, y)

    # node coordinates and position in the global matrix
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    # identifying nodal connectivity for plate elements
    # similar than Nastran's CQUAD4
    #
    #   ^ y
    #   |
    #
    #  4 ________ 3
    #   |       |
    #   |       |   --> x
    #   |       |
    #   |_______|
    #  1         2


    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    num_elements = len(n1s)

    N = DOF*nx*ny
    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_KG = 0

    plates = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSCPlate2D()
        plate.c1 = DOF*nid_pos[n1]
        plate.c2 = DOF*nid_pos[n2]
        plate.c3 = DOF*nid_pos[n3]
        plate.c4 = DOF*nid_pos[n4]
        plate.ABD = prop.ABD
        plate.lex = a/(nx - 1)
        plate.ley = b/(ny - 1)
        plate.init_k_KC0 = init_k_KC0
        plate.init_k_KG = init_k_KG
        update_KC0(plate, points, weights, Kr, Kc, Kv)
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        plates.append(plate)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    # simply supported

    # locating nodes
    bk = np.zeros(KC0.shape[0], dtype=bool) # constrained DOFs, can be used to prescribe displacements

    x = ncoords[:, 0]
    y = ncoords[:, 1]

    # constraining w at all edges
    check = (np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0.) | np.isclose(y, b))
    bk[6::DOF] = check
    # constraining all in-plane motion
    bk[0::DOF] = True
    bk[1::DOF] = True
    bk[2::DOF] = True
    bk[3::DOF] = True
    bk[4::DOF] = True
    bk[5::DOF] = True

    # unconstrained nodes
    bu = ~bk # logical_not

    Kuu = KC0[bu, :][:, bu]

    Nxx = -load_Nxx*4.448222/(25.4/1000) #[N/m]
    Nyy = -load_Nyy*4.448222/(25.4/1000) #[N/m]
    Nxy = 0
    Mxx = 0
    Myy = 0
    Mxy = 0
    for plate in plates:
        update_KG_cte_N(Nxx, Nyy, Nxy, plate, points, weights, KGr, KGc, KGv)

    # eigenvalue solver
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    # solving modified generalized eigenvalue problem
    # Original: (KC0 + lambda*KG)*v = 0
    # Modified: (-1/lambda)*KC0*v = KG*v  #NOTE here we find (-1/lambda)
    num_eigenvalues = 1
    eigvals, eigvecsu = eigsh(A=KGuu, k=num_eigenvalues, which='SM', M=Kuu,
            tol=1e-6, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    eigvecs = np.zeros((KC0.shape[0], num_eigenvalues), dtype=float)
    eigvecs[bu, :] = eigvecsu

    return eigvals[0]


ref_lambda_cs = calc_failure_load_Haftka(ref_prop)
print('ref_lambda_cs', ref_lambda_cs)
#assert np.isclose(ref_lambda_cs, 10394.81, rtol=1e-3)

ref_lambda_cb = calc_buckling(ref_prop)
print('ref_lambda_cb', ref_lambda_cb)
#assert np.isclose(ref_lambda_cb, 9998.19, rtol=1e-2)

ref_prop.make_orthotropic()
ref_prop.make_symmetric()
ref_lambda_cb = calc_buckling(ref_prop)
print('ref_lambda_cb forced A16=A26=D16=D26=0', ref_lambda_cb)
ref_lambda_cs = calc_failure_load_Haftka(ref_prop)
print('ref_lambda_cs forced A16=A26=D16=D26=0', ref_lambda_cs)

"""
From Eq. 10 in
    Shrivastava, S., Sharma, N., Tsai, S. W., and Mohite, P. M., 2020, “D and
    DD-Drop Layup Optimization of Aircraft Wing Panels under Multi-Load Case Design
    Environment,” Compos. Struct., 248(January), p. 112518.
"""
hist = []


def objective_free(x): #NOTE to be minimized
    # NOTE x has 12 layers to build at the end a laminate with 48 layers
    x = np.round(x)
    # NOTE balancing
    x = np.vstack((x, -x)).T.flatten()
    x = x.tolist()
    # NOTE making symmetric
    stack = x + x[::-1]
    prop = laminated_plate(stack=stack, plyt=ply_thickness,
                           laminaprop=laminaprop)
    lambda_cb = calc_buckling(prop)
    lambda_cs = calc_failure_load_Haftka(prop)
    p = 0.08
    obj = (1-p)*min(lambda_cs, lambda_cb)
    hist.append(stack + [lambda_cb, lambda_cs, obj])
    # NOTE uncomment here to make the objective driven by buckling
    #return 1/lambda_cb
    # NOTE uncomment here to make the objective driven by failure criterion
    # return 1/lambda_cs
    # NOTE uncomment here to make the objective driven by Haftka's objective function
    return 1/obj


def objective(x): #NOTE to be minimized
    a1_deg, a2_deg = x
    a1_deg = round(a1_deg, 0)
    a2_deg = round(a2_deg, 0)
    prop_DD = n_double_plate(h, [a1_deg, a2_deg], laminaprop)
    prop_DD.stack = [a1_deg, a2_deg]
    lambda_cb = calc_buckling(prop_DD)
    lambda_cs = calc_failure_load_Haftka(prop_DD)
    p = 0.08
    ref_obj = (1-p)*min(ref_lambda_cs, ref_lambda_cb)
    obj = (1-p)*min(lambda_cs, lambda_cb)
    hist.append([a1_deg, a2_deg, lambda_cb, lambda_cs, obj])
    # NOTE uncomment here to make the objective driven by buckling
    #return 1/lambda_cb
    # NOTE uncomment here to make the objective driven by failure criterion
    # return 1/lambda_cs
    # NOTE uncomment here to make the objective driven by Haftka's objective function
    return 1/obj


from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair


n_plies = 48
n_var = n_plies//4 # balanced and symmetric

# define permutation problem
class Problem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var = n_var,
                         n_obj = 1,
                         xl    = [0]*n_var,
                         xu    = [90]*n_var,
                         vtype = int)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = objective_free(x) # objective function


problem = Problem()

method = algorithm = GA(
    pop_size=50,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True)

res = minimize(problem,
               method,
               termination=('n_gen', 100),
               seed=1,
               save_history=True
               )

hist = np.asarray(hist)

print(res)
# NOTE x has 12 layers to build at the end a laminate with 48 layers
X = np.round(res.X)
# NOTE balancing
X = np.vstack((X, -X)).T.flatten()
X = X.tolist()
# NOTE making symmetric
stack = X + X[::-1]
prop_opt = laminated_plate(stack=stack, plyt=ply_thickness,
                           laminaprop=laminaprop)
opt_lambda_cb = calc_buckling(prop_opt)
opt_lambda_cs = calc_failure_load_Haftka(prop_opt)
print('opt_lambda_cb', opt_lambda_cb)
print('opt_lambda_cs', opt_lambda_cs)
