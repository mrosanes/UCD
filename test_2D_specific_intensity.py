
from object.object_main_func import specific_intensities_2D

specific_intensities_2D(
        L=30, n=80, beta=0, rotation_angle=0, inclination=90,
        Robj_Rsun_scale=4, Bp=7700, Pr=1, D_pc=373, f=5e9, Ra=16,
        l_middlemag=4, δ=2, r_ne=0.002, v_inf=600, inner_contrib=True,
        n_p0=1e8, T_p0=1e6, plot3d=False)

