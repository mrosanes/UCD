import time
from ucd import UCD


def main():
    """
    Notes:
    - Create a UCD with a low grid sampling "n" per edge (eg: <13),
      to be able to plot the vectors without losing a relatively good
      visibility of the vectors
    - Create a UCD with a higher grid sampling "n" per edge (eg: >31 <101)
      in order to have a middle-magnetosphere with a better sampling
      resolution, but without going to too long computation times (for
      101 points per cube edge, ~2min for finding the points of the
      middle-magnetosphere)
    """
    start_time = time.time()
    ucd = UCD(n=15, beta=1, plot3d=True)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_plot, points_LoS_in_B = ucd.LoS_cube()
    # Compute and Plot the UCD (or other (sub)stellar object) dipole
    # magnetic vector field and plot it together with the rotation and the
    # magnetic axes, and the different coordinate systems
    ucd.ucd_compute_and_plot(points_LoS_in_B, points_LoS_plot)
    voxels_middlemag = ucd.find_middle_magnetosphere()
    ucd.plot_middlemag_in_slices(voxels_middlemag, marker_size=2)
    ucd.LoS_voxel_rays()
    ucd.compute_specific_intensity_LoS()
    print("Total Specific Intensity in the plane perpendicular to the LoS")
    print("{:.3g}".format(ucd.total_specific_intensity_LoS))
    end_time = time.time()
    print("UCD computations took %d seconds\n"
          "for %d elements per cube edge" % (end_time - start_time, ucd.n))


if __name__ == "__main__":
    main()

