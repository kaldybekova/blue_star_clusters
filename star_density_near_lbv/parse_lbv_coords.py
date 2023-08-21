import numpy as np

def parse_alpha(alpha_str):
    alpha_hours = float(alpha_str[:2])
    alpha_min = float(alpha_str[2:4])
    alpha_sec = float(alpha_str[4:])
    alpha_coord = 15 * alpha_hours + 15 * alpha_min / 60 + 15 * alpha_sec / 3600
    return alpha_coord

def parse_delta(delta_str):
    delta_deg = float(delta_str[:2])
    delta_min = float(delta_str[2:4])
    delta_sec = float(delta_str[4:])
    delta_coord = delta_deg + delta_min / 60 + delta_sec / 3600
    return delta_coord

lbv_converted_coords = []

with open("m33_lbv_list.txt", "r") as lbv_file:
    lbv_file_lines = lbv_file.readlines()
    for lbv_line in lbv_file_lines:
        lbv_coord = lbv_line.split()[0].split("+")
        alpha_str = lbv_coord[0][1:]
        delta_str = lbv_coord[1]
        alpha = parse_alpha(alpha_str)
        delta = parse_delta(delta_str)
        lbv_converted_coords.append([alpha, delta])

np.savetxt("lbv_coords", np.array(lbv_converted_coords))
