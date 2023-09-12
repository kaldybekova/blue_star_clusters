import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import pandas as pd
from scipy.interpolate import CubicSpline
from functools import lru_cache
import time

def star_count_in_cluster(min_star_count, max_star_count):
    return round(1 / (np.random.uniform() * (max_star_count**-1 - min_star_count**-1) + min_star_count**-1))

# масса всех звёзд в скоплении
def star_mass_in_cluster(star_count, min_star_mass, max_star_mass): # число звёзд в скоплении, мин масса, макс масса
    return (1 / (np.random.uniform(0, 1, star_count) * (max_star_mass**-1.35 - min_star_mass**-1.35) + min_star_mass**-1.35))**0.74

@lru_cache()
def spline('mass-time.csv'): #функция строит сплайн
    table = pd.read_csv('mass-time.csv')
    xdata = table['MASS']
    ydata = table['TIME']
    cubic_spline = CubicSpline(xdata, ydata) #построить сплайн
    return cubic_spline #возвращает все точки сплайна

cs = spline()
    
def cluster_sample(star_masses, age):
    return [mass for mass in star_masses if cs(mass) > age]

# распредление скоплений
def cluster_distrubution(cluster_star_number, age_myr, velocity_dispersion_kms): # число сколпений, возраст, дисперсия скорости
    sigma = velocity_dispersion_kms * age_myr * 3.154 * 10**7 * 10**6 / (3.086 * 10**13)
    star_distances = np.abs(np.random.normal(scale=sigma, size=cluster_star_number))
    star_angles = 2 * np.pi * np.random.uniform(0, 1, cluster_star_number)
    cluster_stars_x = star_distances * np.cos(star_angles) 
    cluster_stars_y = star_distances * np.sin(star_angles)
    return cluster_stars_x, cluster_stars_y

# поиск центра скоплений
def cluster_centers(min_cluster_separation, half_box_size_cluster_pc, cluster_number):
    cluster_centers_coord = 2 * half_box_size_cluster_pc * np.random.random(size=(cluster_number, 2)) - half_box_size_cluster_pc
    cluster_centers_pdist = pdist(cluster_centers_coord)

    while np.any(cluster_centers_pdist < min_cluster_separation):
        cluster_centers_coord = 2 * half_box_size_cluster_pc * np.random.random(size=(cluster_number, 2)) - half_box_size_cluster_pc
        cluster_centers_pdist = pdist(cluster_centers_coord)
       
    return cluster_centers_coord

# Density setup
half_box_size_pc = 150
star_mean_density_per_square_pc = 3 * 10**-4 # средняя плотность звёзд на квадратный пк
star_mean_count = 4 * half_box_size_pc**2 * star_mean_density_per_square_pc # среднее число звёзд

half_box_size_cluster_pc = 50
cluster_mean_density_per_square_pc = 10**-4 # средняя плотность скоплений на квадратный пк
cluster_mean_count = 4 * half_box_size_cluster_pc**2 * cluster_mean_density_per_square_pc # среднее число скоплений
min_cluster_separation = 50 # в каких единицах?


min_star_count = 30
max_star_count = 50

min_cluster_age = 0
max_cluster_age = 11

min_star_mass = 16
max_star_mass = 100

velocity_dispersion = 13.5 # в км/с?

raw_final_data_frame = []

# Background modeling

# среднее число звёзд разбросать по пуассону
model_stars_number = np.random.poisson(star_mean_count)

background_stars_x = 2 * half_box_size_pc * np.random.uniform(0, 1, model_stars_number) - half_box_size_pc
background_stars_y = 2 * half_box_size_pc * np.random.uniform(0, 1, model_stars_number) - half_box_size_pc

# 0 is ID for background
raw_final_data_frame = list(zip([0] * len(background_stars_x), background_stars_x, background_stars_y))

# Cluster modeling

#model_cluster_number = np.random.poisson(cluster_number)
model_cluster_number = 1
all_clusters_center_coord = cluster_centers(min_cluster_separation, half_box_size_cluster_pc, model_cluster_number)
cluster_id = 1
raw_final_cluster_center = []

# Age modeling

cluster_age = max_cluster_age * np.random.random() - min_cluster_age
print(cluster_age)

plt.figure() #temp

for cluster_coord in all_clusters_center_coord:
    cluster_center_x = cluster_coord[0]
    cluster_center_y = cluster_coord[1]

    star_number = star_count_in_cluster(min_star_count, max_star_count)
    
    # масса всех звёзд в скоплении
    star_masses = star_mass_in_cluster(star_number, min_star_mass, max_star_mass)
#     print(star_masses)
    
    start = time.time()    
    star_sample = cluster_sample(star_masses, cluster_age)
    end = time.time()    
    print('star_sample', '\n', star_sample)
    print(f'test1 = {end - start}\n')
    
    
    start = time.time()    
    star_sample = cluster_sample(star_masses, cluster_age)
    end = time.time()    
    print('star_sample', '\n', star_sample)
    print(f'test1 = {end - start}\n')
    
    
    cluster_stars_x, cluster_stars_y = cluster_distrubution(len(star_sample), cluster_age, velocity_dispersion)
    cluster_stars_x += cluster_center_x
    cluster_stars_y += cluster_center_y

    raw_final_data_frame.extend(list(zip([cluster_id] * len(cluster_stars_x), cluster_stars_x, cluster_stars_y)))
    raw_final_cluster_center.append([cluster_id, cluster_center_x, cluster_center_y])
    cluster_id += 1

    plt.scatter(cluster_center_x, cluster_center_y, color='k', s=100, marker='*', zorder=3)
    plt.scatter(cluster_stars_x, cluster_stars_y, alpha=0.5)

plt.scatter(background_stars_x, background_stars_y, color='grey', alpha=0.5)
plt.xlim([-150, 150])
plt.ylim([-150, 150])
plt.show()

header_info = "cluster_age_myr=%.2f, background_star_density=%.2e\nstar_cluster_density=%.2e, model_cluster_number=%d" % (cluster_age, star_mean_density_per_square_pc, cluster_mean_density_per_square_pc, model_cluster_number)
np.savetxt("cluster_centers_and_modeling_info", raw_final_cluster_center, fmt="%.4f %.4f %d", header=header_info)

cluster_data = pd.DataFrame(raw_final_data_frame, columns=['cluster_id', 'x','y'])
cluster_data.to_csv("cluster_data")

print(star_masses)
