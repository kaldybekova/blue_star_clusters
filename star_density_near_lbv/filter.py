import numpy as np

star_table = np.loadtxt("table6_cut_wide.dat")

blue_stars = star_table[(star_table[:, 8] < -0.20) & (star_table[:, 10] < -1.00) & (star_table[:, 6] < 20.0)]
blue_stars_coord = []

for star in blue_stars:
    coord_alpha = 15 * star[0] + 15 * star[1] / 60 + 15 * star[2] / 3600
    coord_delta = star[3] + star[4] / 60 + star[5] / 3600
    blue_stars_coord.append([coord_alpha, coord_delta])

np.savetxt("blue_stars_coord_bright20", np.array(blue_stars_coord))
