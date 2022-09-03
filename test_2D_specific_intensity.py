
from object.object_main_func import specific_intensities_2D

# MCP Star HD37479: rotation_angle = -self.phi_r + 72*np.pi/180.0
# MCP Star HD37017: -self.phi_r + np.pi/180.0
specific_intensities_2D(
        L=20, n=40, beta=65, rotation_angle=324, inclination=25,
        Robj_Rsun_scale=4, Bp=7700, Pr=1, D_pc=373, f=5e9, Ra=14,
        l_middlemag=6, δ=2, r_ne=0.002, v_inf=600, inner_contrib=False,
        n_p0=1e8, T_p0=1e6, plot3d=False)

