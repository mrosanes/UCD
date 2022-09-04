
from object.object_main_func import flux_densities_1D

flux_densities_1D(
    L=40, n=41, beta=65, inclination=25, Robj_Rsun_scale=4, Bp=7700, Pr=1,
    D_pc=373, f=5e9, Ra=15, l_middlemag=7, δ=2, r_ne=0.002, v_inf=600,
    rotation_angle_step=30, inner_contrib=True, n_p0=3e9, T_p0=1e5,
    use_symmetry=True, plot3d=False, mk_new_qapp=True)

