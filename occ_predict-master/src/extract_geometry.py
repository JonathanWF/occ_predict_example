# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:15:28 2019

@author: Dhadma Balachandran
"""

import os
import glob
import argparse

from skimage import measure
from skimage import segmentation
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fix the complaints from PIL about image size
# See: https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = 1311157110

# Set up the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--labels_dir', default='data/labels',
    help="Directory containing .png labels for the dataset")
parser.add_argument('--geoms_dir', default='data/geoms',
    help="Directory to store the extracted geometry")
parser.add_argument('--resize_fraction', default=0.1,
    help="Resize fraction to create the coordinates")

def extract_geoms(label_img, resize_fraction):
    """Extract the satellite and tumor geometry from the label image.

    Requires a path string in "label_img" and will return three objects:
    - Set of satellite boundaries
    - Set of satellite centroids
    - Set of tumor boundaries
    """

    # Reading the actual Image and converting to 8bit
    img = Image.open(label_img)
    h,w = img.size

    # Resize the image to make analysis easier / faster
    new_size = (int(h*resize_fraction), int(w*resize_fraction))

    # Test to make sure that we don't have a problem with the new size, print
    # an error and continue if we do
    for x in new_size:
        if x<=0:
            tqdm.write(f"{label_img}: Original size ({h}, {w}) resize to {new_size}.")

            # Return a blank array for these coords
            return ([], [], [], [])
    img = img.resize(new_size, Image.NEAREST)
    img = np.array(img)
    img = np.uint8(img)

    # Separating the image into its channels
    img_red = img[:,:,0]
    img_green = img[:,:,1]
    img_blue = img[:,:,2]

    # Tumor is pure blue; satellites are pure green
    img_tumor = (img_red == 0) & (img_green == 0) & (img_blue == 255)
    img_satellites = (img_red == 0) & (img_green == 255) & (img_blue == 0)

    # Label the satellites and extract their properties to get centroids
    img_sat_labels = measure.label(img_satellites,return_num=True)
    satellite_props = measure.regionprops(img_sat_labels[0])

    # Build the sat centroids and sat area output
    sat_centroids = []
    sat_areas = []
    for satellite_prop in satellite_props:
        sat_centroids.append(satellite_prop.centroid)
        sat_areas.append(satellite_prop.area)

    # Grab the boundary of the tumor
    img_tum_boundary = segmentation.find_boundaries(img_tumor)
    boundary_tum = np.nonzero(img_tum_boundary)

    # Split apart boundary coordinates
    boundary_tum_x = boundary_tum[0]
    boundary_tum_y = boundary_tum[1]

    # Satellite boundaries
    img_sat_boundary =[]
    sat_bounds = []

    for label in range(1,img_sat_labels[1]+1):
        img_labels = (img_sat_labels[0] == label)
        img_sat_boundary.append(segmentation.find_boundaries(img_labels))
        sat_bounds.append(np.nonzero(img_sat_boundary[label-1]))

    # Construct the output boundary points
    tum_bounds = []
    for x, y in zip(boundary_tum_x, boundary_tum_y):
        tum_bounds.append((x,y))

    return (sat_bounds, sat_centroids, sat_areas, tum_bounds, img_tumor)

def basename_diffs(a, b):
    """Return the items in a whose 'root' (the part before the file extension)
    does not match the roots in b.

    Parameters:
    a (list): A list of strings that contain paths of interest.
    b (list): A list of strings of filenames used to filter a.

    Returns:
    list: A filtered copy of a that contains items whose roots do not match
    the roots of the items in b.
    """

    assert len(a)>0, f"Length of input list must exceed 0"

    # Grab the extension of the first item in a
    a_ext = os.path.splitext(a[0])[1]

    # Save the paths for a, confirm all paths start in the same folder
    a_heads = [os.path.split(x)[0] for x in a]
    assert len(np.unique(a_heads)) == 1, f"All input files must come from the same folder."
    a_head = a_heads[0]

    # Grab the filenames
    a_tails = [os.path.split(x)[1] for x in a]
    b_tails = [os.path.split(x)[1] for x in b]

    # Grab the roots of each filename (without the extension)
    a_stems = [os.path.splitext(x)[0] for x in a_tails]
    b_stems = [os.path.splitext(x)[0] for x in b_tails]

    # Get the difference between the two as a list
    a_diff = list(set(a_stems)-set(b_stems))

    # Return the new paths with only values that exist in a
    return [os.path.join(a_head, x+a_ext) for x in a_diff]

if __name__ == "__main__":
    # Parse input arguments
    args = parser.parse_args()
    labels_dir = args.labels_dir
    geoms_dir = args.geoms_dir
    resize_fraction = float(args.resize_fraction)

    # Check the input arguments
    assert os.path.exists(labels_dir), f'Labels directory ({labels_dir}) does not exist!'

    # Create the output directory if it doesn't exist
    if os.path.exists(geoms_dir):
        Warning('Path exists! Possible data overwrite!')
    else:
        os.makedirs(geoms_dir)

    # Grab a list of the labels in the label folder
    label_imgs = glob.glob(os.path.join(labels_dir, '*.png'))

    # Make sure that there is at least one image in the label folder
    assert len(label_imgs) > 0, 'Could not find any label images in the labels dir!'

    # Remove label images that are already extracted
    # Delete the files on disk if you want to re-extract
    geom_files = glob.glob(os.path.join(geoms_dir, '*.npz'))

    # Return the set difference of the filenames based on the root
    label_imgs = basename_diffs(label_imgs, geom_files)

    # Begin cycling through each label image and extracting geometry
    for label_img in tqdm(label_imgs, desc="Label Images"):
        # Run the extract_geoms for this label image
        sat_bounds, sat_centroids, sat_areas, tum_bounds, tum_bin = extract_geoms(label_img, resize_fraction)

        # Check that the data was extracted correctly; we should have at least
        # one main tumor and one satellite
        if len(sat_centroids) < 1 or len(tum_bounds) < 1:
            tqdm.write(f" {label_img}: Less than one satellite or tumor found. Please check.")
            tqdm.write(f"{sat_centroids}")
            tqdm.write(f"{tum_bounds}")
            continue

        # Save the extracted geometry
        save_path = os.path.join(geoms_dir, os.path.splitext(os.path.basename(label_img))[0]+'.npz')
        np.savez_compressed(save_path,
                            sat_bounds=sat_bounds,
                            sat_centroids=sat_centroids,
                            sat_areas=sat_areas,
                            tum_bounds=tum_bounds,
                            resize_fraction=resize_fraction,
                            tum_bin=tum_bin)
