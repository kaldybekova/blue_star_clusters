import numpy as np

star_table = np.loadtxt("table6_cut_wide.dat")

b_v = star_table[:, 8]
u_b = star_table[:, 10]
v_mag = star_table[:, 6]

e_b_v = 0.13
distance_to_galaxy_pc = 930 * 10**3

abs_mag_upper = -5
v_mag_upper = abs_mag - 5 + 5 * np.log10(distance_to_galaxy_pc) 

blue_stars = star_table[(b_v < -0.20) & (u_b < -1.00) & (v_mag < v_mag_upper)]
blue_stars_coord = []

for star in blue_stars:
    coord_alpha = 15 * star[0] + 15 * star[1] / 60 + 15 * star[2] / 3600
    coord_delta = star[3] + star[4] / 60 + star[5] / 3600
    blue_stars_coord.append([coord_alpha, coord_delta])

np.savetxt("blue_stars_coord_bright20", np.array(blue_stars_coord))
