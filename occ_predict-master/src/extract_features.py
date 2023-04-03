"""
Extracts architecture-based features from a set of geoms.

Saves the features in a new folder.

"""

# Imports
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as T

from skimage import morphology
from skimage import segmentation
from matplotlib.path import Path
from scipy.spatial import Delaunay
from scipy.spatial import distance
from scipy.spatial import ConvexHull
from scipy import stats

# Set up input argument parser
# Set up the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--geoms_dir', default=os.path.join('data', 'geoms'),
    help="Directory containing the extracted geometry")
parser.add_argument('--features_dir', default=os.path.join('data', 'features'),
    help="Directory to save the extracted features")

# Descriptive statistics feature calculation
def descriptive_stats(x):
    minimum = np.amin(x)
    maximum = np.amax(x)
    mean = np.mean(x)
    variance = np.var(x)
    standard_deviation = np.std(x)
    skewness = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    moment_5 = stats.moment(x, moment = 5)
    moment_6 = stats.moment(x, moment = 6)
    moment_7 = stats.moment(x, moment = 7)
    moment_8 = stats.moment(x, moment = 8)
    moment_9 = stats.moment(x, moment = 9)
    moment_10 = stats.moment(x, moment = 10)
    moment_11 = stats.moment(x, moment = 11)
    geometric_mean = stats.gmean(x)
    harmonic_mean = stats.hmean(x)
    features = [minimum, maximum, mean, variance, standard_deviation,\
                skewness, kurtosis, moment_5, moment_6, moment_7,\
                moment_8, moment_9, moment_10, moment_11, geometric_mean, harmonic_mean]
    return(features)

def assign_wave_index(tum_bin, sat_bounds):
    tum_counter = 0
    sat_wave_number = np.zeros((len(sat_bounds),1))
    max_dilations = 1500

    while True:
        # Dilate the tumor
        tum_bin = morphology.binary_dilation(tum_bin)

        # Increment the counter
        tum_counter += 1

        # Check to see if any satellites are "hit" by the (dilated) tumor
        # Get dilated tumor boundary points
        img_tum_boundary = segmentation.find_boundaries(tum_bin)
        boundary_tum = np.nonzero(img_tum_boundary)

        # Split apart boundary coordinates
        boundary_tum_x = boundary_tum[0]
        boundary_tum_y = boundary_tum[1]
        tum_bin_points = np.array([boundary_tum_x, boundary_tum_y]).T
            
        # Get satellite wave number   
        for sat_idx, sat_bound in enumerate(sat_bounds):
            
            sat_bound = np.array([sat_bound[0], sat_bound[1]]).T    
            tum_poly = Path(tum_bin_points)
            sat_hit = tum_poly.contains_points(sat_bound)
            if np.any(sat_hit == True) and sat_wave_number[sat_idx] == 0:
                sat_wave_number[sat_idx] = tum_counter

        # Check to see if every satellite has been hit
        if np.all(sat_wave_number > 0):
            return sat_wave_number

        # Make sure we aren't in an infinite loop
        if tum_counter > max_dilations:
            print(f"Not all sats have been assigned an index after {max_dilations} iterations. Exiting.")
            return sat_wave_number

if __name__=="__main__":

    # Parse input arguments
    args = parser.parse_args()
    geoms_dir = args.geoms_dir
    features_dir = args.features_dir

    # Create the output directory if it does not exist
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # Locate the geoms and set up directories
    geom_files = glob.glob(os.path.join(geoms_dir, '*.npz'))
    assert len(geom_files)>0, f'Could not find any geoms in directory {geoms_dir}!'

    # Cycle through the geoms
    for geom_file in geom_files:
        # Grab the name of the geom file
        geom_name = os.path.basename(geom_file)
        feature_name = geom_name.replace('.npz', '_features.npz')
        feature_path = os.path.join(features_dir, feature_name)

        print(f'Analyzing {geom_name}')

        # Check to see if the features exist
        if os.path.exists(feature_path):
            print(f'\tFeature {feature_name} exists at {feature_path}, skipping')
            continue

        # Load the geom
        with np.load(geom_file, allow_pickle=True) as f:
            sat_bounds = f['sat_bounds']
            sat_centroids = f['sat_centroids']
            tum_bounds = f['tum_bounds']

        if len(sat_bounds) == 0:
            print('\tSat bounds is empty!')
            continue
        if len(sat_centroids) == 0:
            print('\tSat centroids are empty!')
            continue
        if len(sat_centroids) <3:
            print('\tNot enough satellites!')
            continue
        if len(tum_bounds) == 0:
            print('\ttumor boundaries are empty!')
            continue

        # Pre-process the data
        print('\tNumber of satellites: {}'.format(len(sat_centroids)))
        print('\tMinimum satellite centroid: {}'.format(np.min(sat_centroids)))
        print('\tMaximum satellite centroid: {}'.format(np.max(sat_centroids)))
        print('\tNumber of tumor boundary points: {}'.format(len(tum_bounds)))
        # DEBUG: Create a function to map the boundaries onto the labelmaps / original slide data

        # Extract Features
        # Get the list of triangles that contain tumor boundary points
        if geom_name == '004a.npz':
            ## Delaunay
            tri = Delaunay(sat_centroids)
            # Grab the simplices
            
            eliminate_triangles = tri.find_simplex(tum_bounds)
            print(f'Length of simplexes that cross the tumor boundary: {len(eliminate_triangles)}')
    
            # Grab the unique simplices that are greater than -1
            eliminate_triangles = np.unique(eliminate_triangles[eliminate_triangles>0])
            print(f'Unique, non-negative simplex coordinates: {eliminate_triangles}')
    
            # Extract the triangles that are listed in "eliminate_triangles"
            tri_simplices = tri.simplices.copy()
            tri_simplices = np.delete(tri_simplices, eliminate_triangles, axis=0)
    
            #idx_eliminate = []
            # for i,triangle in enumerate(tri.simplices):
            plt.triplot(sat_centroids[:,0], sat_centroids[:,1], tri_simplices)
            plt.plot(sat_centroids[:,0], sat_centroids[:,1], 'o')
            plt.scatter(tum_bounds[:,0], tum_bounds[:,1])
            plt.show()
            
            # # Area and Length of sides of each triangle in Delaunay
            t = T.Triangulation(sat_centroids[:,0], sat_centroids[:,1], tri_simplices)
            triangle_lengths = []
    
            for edge in t.edges:
                x1 = sat_centroids[edge[0], 0]
                x2 = sat_centroids[edge[1], 0]
                y1 = sat_centroids[edge[0], 1]
                y2 = sat_centroids[edge[1], 1]
                triangle_lengths.append( np.sqrt((x2-x1)**2 + (y2-y1)**2 ) )
    
            triangle_length_features = descriptive_stats(triangle_lengths)
            # triangle_length_feats.append(np.reshape(np.array(triangle_length_features), [1,16]))
            
            triangle_areas = []
            for simplex in tri_simplices:
                # Pull out the points for this triangle
                p1 = sat_centroids[simplex[0], :]
                p2 = sat_centroids[simplex[1], :]
                p3 = sat_centroids[simplex[2], :]
                
                # Calculate edge lengths for this triangle
                e12 = np.sqrt( (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 )
                e13 = np.sqrt( (p3[0]-p1[0])**2 + (p3[1]-p1[1])**2 )
                e23 = np.sqrt( (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2 )
                
                # Calculate area for this triangle
                s = (e12 + e13 + e23) / 2
                a = np.sqrt( s * (s-e12) * (s-e13) * (s-e23))
                triangle_areas.append(a)
                
            triangle_area_features = descriptive_stats(triangle_areas)
            # triangle_area_feats.append(np.reshape(np.array(triangle_area_features), [1,16]))


#------------------ Statistics / Distance---------------------------------------------------------------------------------------------
            sat_dis = []
            sat_distance = []
            for sat_bound in sat_bounds:
                sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                sat_dis = distance.cdist(sat_bound, tum_bounds, 'euclidean')
                sat_distance.append(sat_dis)
            
            ## Minimum distance between each satellite and main tumor
            ## sat_min - minimum distance between each sat_bounds value and main tumor
            ## sat_dist_minimum - minimum distance between each satellite and main tumor
            # sat_min = []
            sat_dist_minimum = []
            min_sat_bounds = []
            min_tum_bounds = []
            for idx, sat_dist in enumerate(sat_distance):
                # min_dist = np.amin(sat_dist, axis = 1)
                # sat_min.append(min_dist)
                # sat_dist_minimum.append(np.amin(min_dist))
                min_dist = np.min(sat_dist)
                min_dist_idx = np.where(sat_dist == min_dist)
                min_dist_idx = np.array([min_dist_idx[0][0], min_dist_idx[1][0]])
                sat_dist_minimum.append(np.min(sat_dist))
                sat_bound = np.array([sat_bounds[idx][0], sat_bounds[idx][1]]).T
                min_sat_bounds.append([(sat_bound[min_dist_idx[0]][0], sat_bound[min_dist_idx[0]][1])])
                min_tum_bounds.append([(tum_bounds[min_dist_idx[1]][0], tum_bounds[min_dist_idx[1]][1])])
            sat_dist_features = descriptive_stats(sat_dist_minimum)
            # sat_dist_feats.append(np.reshape(np.array(sat_dist_features), [1,16]))   
                
            plt.figure(4)
            min_sat_bounds = np.reshape(min_sat_bounds, [len(sat_distance),2])
            min_tum_bounds = np.reshape(min_tum_bounds, [len(sat_distance),2])
            plt.plot([min_sat_bounds[:,0], min_tum_bounds[:,0]], [min_sat_bounds[:,1], min_tum_bounds[:,1]], 'k', linewidth = 3.0)
            plt.scatter(tum_bounds[:,0], tum_bounds[:,1])
            for sat_bound in sat_bounds:
                sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                plt.scatter(sat_bound[:,0], sat_bound[:,1])
            plt.show()
            
#------------------ Convex Hull-------------------------------------------------------------------------------------------------------
            sat_boundaries = []
            sat_convexHull = []
            for sat_bound in sat_bounds:
                sat_boundary = np.array([sat_bound[0], sat_bound[1]]).T
                sat_convexHull.append(ConvexHull(sat_boundary))
                if len(sat_boundaries) == 0:
                    sat_boundaries = sat_boundary
                    continue
                sat_boundaries = np.concatenate((sat_boundaries, sat_boundary), axis = 0)
                
            sat_hull = ConvexHull(sat_boundaries)
            tum_hull = ConvexHull(tum_bounds)
            plt.figure(2)
            for simplex in sat_hull.simplices:
                plt.plot(sat_boundaries[simplex,0], sat_boundaries[simplex,1], c='k', alpha=0.5)
            # for simplex in tum_hull.simplices:
            #     plt.plot(tum_bounds[simplex,0], tum_bounds[simplex,1], c='k', alpha=0.5)
            plt.scatter(tum_bounds[:,0], tum_bounds[:,1])
            for sat_bound in sat_bounds:
                sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                plt.scatter(sat_bound[:,0], sat_bound[:,1])
            plt.show()
    
            convex_area = sat_hull.area
            # convex_area = sat_hull.area - tum_hull.area
            sat_convex_ratio = []
            for sat_area in sat_convexHull:
                sat_convex_ratio.append(sat_area.area/convex_area)
            convex_ratio_features = descriptive_stats(sat_convex_ratio)

#-------------------- Wave Features------------------------------------------------------------------------------
        #Creating satellite wave numbers
            sat_wave_number = assign_wave_index(tum_bin, sat_bounds)
        #Creating graphs
            sat_wave_indices = np.reshape(sat_wave_number, [len(sat_wave_number),])
             #Plot centroids and wave indices
            min_col, min_row = np.min(sat_centroids, axis = 0)
            max_col, max_row = np.max(sat_centroids, axis = 0)
            width = max_col - min_col
            height = max_row - min_row
            fig, ax = plt.subplots() 
            for idx, sat_centroid in enumerate(sat_centroids):
                ax.text(sat_centroid[0], sat_centroid[1], str(idx), style ='normal', fontweight = 'bold', 
                    fontsize =20, color ="black") 
              
                # ax.set(xlim =(0, width+1000), ylim =(0, height+1000))   
            for sat_bound in sat_bounds:
                sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                plt.scatter(sat_bound[:,0], sat_bound[:,1])
            plt.show()
            
            fig, ax = plt.subplots() 
            for idx, sat_centroid in enumerate(sat_centroids):
                ax.text(sat_centroid[0], sat_centroid[1], str(np.ceil(sat_wave_indices[idx])), 
                    style ='normal', fontweight = 'bold', 
                    fontsize =20, color ="black") 
              
                # ax.set(xlim =(0, width+1000), ylim =(0, height+1000))   
            for sat_bound in sat_bounds:
                sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                plt.scatter(sat_bound[:,0], sat_bound[:,1])
            plt.scatter(tum_bounds[:,0], tum_bounds[:,1], linewidth = 0.01, edgecolors = 'b')
            plt.show()
            

            #Plot graph            
            new_sat_centroids = sat_centroids
            used_indices = np.zeros(len(sat_wave_indices))
            fig = plt.figure()
            distances_list = []
            while (len(sat_wave_indices) != 0):

                flag = 0
                max_sat_number = np.max(sat_wave_indices)
                max_sat_idx = np.argmax(sat_wave_indices)
                used_indices[max_sat_idx] += 1
                while (flag == 0):
                    initialTOtargets_min_distance = []
                    target_matrix = []
                    if used_indices[max_sat_idx] > 1:
                        sat_wave_indices = np.delete(sat_wave_indices, max_sat_idx)
                        new_sat_centroids = np.delete(new_sat_centroids, max_sat_idx, 0)
                        used_indices = np.delete(used_indices, max_sat_idx)
                        flag = 1
                    else:
                        target_satellites = sat_wave_indices < max_sat_number
                        sat_initial_bounds = [(new_sat_centroids[max_sat_idx][0], new_sat_centroids[max_sat_idx][1])]
                    
                        tum_sat_distance = distance.cdist(sat_initial_bounds, tum_bounds, metric = 'euclidean')
                        min_tum_distance = np.min(tum_sat_distance)
                        min_tum_idx = np.argmin(tum_sat_distance)
                        for ind, outcome in enumerate(target_satellites):
                            if outcome == True:
                                sat_target_bounds = [(new_sat_centroids[ind][0], new_sat_centroids[ind][1])]
                                target_distance = distance.cdist(sat_initial_bounds, sat_target_bounds, metric = 'euclidean')
                                target_matrix.append(target_distance)
                                initialTOtargets_min_distance.append(np.min(target_distance))
                
                                # if (len(initialTOtargets_min_distance) == len(np.array(np.nonzero(target_satellites)).T)):
                                if (len(initialTOtargets_min_distance) == len(target_satellites)):
                                    distance_min = np.min(initialTOtargets_min_distance)
                                    distance_idx = np.argmin(initialTOtargets_min_distance)
                                    sat_target_number = sat_wave_indices[distance_idx]
                                    
                                    if distance_min < min_tum_distance:
                                        distances_list.append([(max_sat_number, sat_target_number, distance_min)])
                                        new_target_bounds = [(new_sat_centroids[distance_idx][0], new_sat_centroids[distance_idx][1])]
                                        # fig = plt.figure()
                                        for sat_bound in sat_bounds:
                                            sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                            plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                        plt.scatter(tum_bounds[:,0], tum_bounds[:,1], edgecolors = 'b')
                                        x = sat_initial_bounds
                                        y = new_target_bounds
                                        plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                        plt.show()
                                        if np.any((sat_wave_indices > max_sat_number))==True:
                                            sat_wave_indices = sat_wave_indices
                                            max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                            max_sat_idx = distance_idx
                                            used_indices[distance_idx] += 1
                                            target_satellites = []
                                        else:
                                            sat_wave_indices = np.delete(sat_wave_indices, max_sat_idx)
                                            new_sat_centroids = np.delete(new_sat_centroids, max_sat_idx, 0)
                                            used_indices = np.delete(used_indices, max_sat_idx)
                                            if distance_idx > max_sat_idx:
                                                max_sat_number = np.float64(sat_wave_indices[distance_idx - 1])
                                                max_sat_idx = distance_idx - 1
                                                used_indices[distance_idx - 1] += 1
                                                target_satellites = []
                                                # sat_wave_indices = new_wave_indices
                                            else:
                                                max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                                max_sat_idx = distance_idx
                                                used_indices[distance_idx] += 1
                                                target_satellites = []
                                                # sat_wave_indices = new_wave_indices
                                    else:
                                        distances_list.append([(max_sat_number, 0, min_tum_distance)])
                                        new_tum_bounds = [(tum_bounds[min_tum_idx][0], tum_bounds[min_tum_idx][1])]
                                        
                                        # fig, ax = plt.subplots()
                                        # fig = plt.figure()
                                        for sat_bound in sat_bounds:
                                            sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                            plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                        plt.scatter(tum_bounds[:,0], tum_bounds[:,1],edgecolors ='b')
                                        x = sat_initial_bounds
                                        y = tum_bounds[min_tum_idx]
                                        y = [(y[0], y[1])]
                                        plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                        plt.show()
                                        flag = 1
                            else:
                                target_distance = np.inf
                                target_matrix.append(target_distance)
                                initialTOtargets_min_distance.append(target_distance)
                                
                                # if (len(initialTOtargets_min_distance) == len(np.array(np.nonzero(target_satellites)).T)):
                                if (len(initialTOtargets_min_distance) == len(target_satellites)):
                                    distance_min = np.min(initialTOtargets_min_distance)
                                    distance_idx = np.argmin(initialTOtargets_min_distance)
                                    sat_target_number = sat_wave_indices[distance_idx]
                                    if distance_min < min_tum_distance:
                                        distances_list.append([(max_sat_number, sat_target_number, distance_min)])
                                        new_target_bounds = [(new_sat_centroids[distance_idx][0], new_sat_centroids[distance_idx][1])]
                                        
                                        # fig, ax = plt.subplots()
                                        # fig = plt.figure()
                                        for sat_bound in sat_bounds:
                                            sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                            plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                        plt.scatter(tum_bounds[:,0], tum_bounds[:,1],edgecolors = 'b')
                                        x = sat_initial_bounds
                                        y = new_target_bounds
                                        plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                        plt.show()
                                        if np.any((sat_wave_indices > max_sat_number))==True:
                                            sat_wave_indices = sat_wave_indices
                                            max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                            max_sat_idx = distance_idx
                                            used_indices[distance_idx] += 1
                                            target_satellites = []
                                        else:
                                            sat_wave_indices = np.delete(sat_wave_indices, max_sat_idx)
                                            new_sat_centroids = np.delete(new_sat_centroids, max_sat_idx, 0)
                                            used_indices = np.delete(used_indices, max_sat_idx)
                                            if distance_idx > max_sat_idx:
                                                max_sat_number = np.float64(sat_wave_indices[distance_idx - 1])
                                                max_sat_idx = distance_idx - 1
                                                used_indices[distance_idx - 1] += 1
                                                target_satellites = []
                                                # sat_wave_indices = new_wave_indices
                                            else:
                                                max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                                max_sat_idx = distance_idx
                                                used_indices[distance_idx] += 1
                                                target_satellites = []
                                                # sat_wave_indices = new_wave_indices
                                    else:
                                        distances_list.append([(max_sat_number, 0, min_tum_distance)])
                                        new_tum_bounds = [(tum_bounds[min_tum_idx][0], tum_bounds[min_tum_idx][1])]
                                        
                                        # fig, ax = plt.subplots()
                                        # fig = plt.figure()
                                        for sat_bound in sat_bounds:
                                            sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                            plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                        plt.scatter(tum_bounds[:,0], tum_bounds[:,1],edgecolors ='b')
                                        x = sat_initial_bounds
                                        y = tum_bounds[min_tum_idx]
                                        y = [(y[0], y[1])]
                                        plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                        plt.show()
                                        flag = 1
        #Calculating cumulative distance from each satellite
            sat_tum_distances = []
            sat_wave_index = np.reshape(sat_wave_number, [len(sat_wave_number),])
            distances = np.array(distances_list)
            dis = np.reshape(distances, [distances.shape[0],3])
            for wave_num in sat_wave_index:
                number = 1
                sum_dist = 0
                idx, = np.where(dis[:,0] == wave_num)[0]
                while(number != 0):
                    number = dis[idx,1]
                    sum_dist = sum_dist + dis[idx,2]
                    if number > 0:
                        idx, = np.where(dis[:,0] == number)[0]
                    else:
                        number = 0    
                sat_tum_distances.append(sum_dist)
