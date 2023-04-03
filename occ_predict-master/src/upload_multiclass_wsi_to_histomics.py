"""Given a multiclass PNG labelmap image, this file will upload it to the corresponding WSI in HistomicsTK.

Requires a spreadsheet of GT codes and ROI-to-WSI matching, as well as a local
PNG file to upload.
"""

import json
import os
import argparse
import pickle

import girder_client
import large_image

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from imageio import imread

from skimage import measure
from skimage import segmentation
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

# Need to pull things like API key from a .env file
from dotenv import load_dotenv

#import utils

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
#    parse_slide_annotations_into_table,
    get_image_from_htk_response # Given a girder response, get np array image
)

from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours # Converts a contour table into a set of annotation docs for uploading
)

# Set up the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--label_path', help="Path to the .png label to upload.")
parser.add_argument('--env_path', default=".env", help="Path to an ENV file defining important data (default: /.env)")
parser.add_argument('--gtcodes_path', default=os.path.join('data', 'occ_histomics.json'), help="Path to the GT Codes file (default: data/gt_codes.csv)")
parser.add_argument('--roi_data_path', default=os.path.join('data', 'histomics_data.xlsx'), help="Path to the Excel spreadsheet containing roi-to-wsi conversions and offsets (default: data/histomics_data.xlsx)")

def get_collection_id(collection_name, conn):
    '''Given a connection, grab the id of the target collection.'''
    collection_id = None

    # List all collections and find the target one
    collection_list = conn.listCollection()

    for collection in collection_list:
        if collection['name'] == collection_name:
            collection_id = collection['_id']
    assert collection_id, f"Connected to server, but could not find collection named {collection_name}"

    return collection_id

def get_folder_id(folder_name, conn, collection_id):
    '''Given a folder name, connection, and collection ID number, return the folder ID number.'''

    folder_id = None

    # List all collections and find the target one
    folder_list = conn.listFolder(collection_id, parentFolderType='collection')
    for folder in folder_list:
        if folder['name'] == folder_name:
            folder_id = folder['_id']
    assert folder_id, f"Connected to server and found some folders, but could not find folder named {folder_name}"

    return folder_id

def get_sample_id(sample_name, conn, collection_name, folder_name):
    '''Given a connection, collection & folder combo, and sample name, return the sample ID number.

    We assume that there are no nested folders -- it goes collection /
    folder_list / item_list
    '''

    # Initialize the ids to check that we've found them
    collection_id = get_collection_id(collection_name, conn)
    folder_id = get_folder_id(folder_name, conn, collection_id)
    sample_id = None

    item_list = conn.listItem(folder_id)
    for item in item_list:
        if item['name'] == sample_name:
            sample_id = item['_id']

    return sample_id

def get_image_metadata(conn, sample_id):
    resp = conn.get(f'/item/{sample_id}/tiles')
    return resp

def verify_files(file_list):
    '''Given a list of files, verifies that they all exist.'''
    for f in file_list:
        assert os.path.exists(f), (f"Required file or folder {f} does not exist. Please check the provided location and try again.")
def get_connection(APIURL, APIKEY):
    # Authenticate with the server
    gc = girder_client.GirderClient(apiUrl=APIURL)
    gc.authenticate(apiKey=APIKEY)
    return gc

def load_contours_from_png(label_path, gtcodes_df, resize_fraction):
    '''Creates a dataframe of annotations ready to upload to HistomicsTK.
    Doesn't actually do the uploading.
    Requires a set of processed contours that have been saved next to the
    labelmaps; if this doesn't exist already, creates it from the PNG.
    This process takes awhile, which is why we check to see if we need to do it
    first.'''

    # We do the flagging first (this is fast) so we only load the labelmap once
    # (slow) if ANY of the labelmaps are absent.

    label_dir, label_name = os.path.split(label_path)
    contours_path = os.path.join(label_dir, os.path.splitext(label_name)[0]+'_contours.pkl')

    # If the contours don't exist for this sample, create them
    if not os.path.exists(contours_path):
        extract_gt_data(label_path, contours_path, gtcodes_df, resize_fraction)

    with open(contours_path, 'rb') as f:
        _, obj_bounds = pickle.load(f)

    # Begin conversion to Histomics
    group = []
    color = []
    ymin = []
    ymax = []
    xmin = []
    xmax = []
    has_holes = []
    touches_edge_top = []
    touches_edge_left = []
    touches_edge_bottom = []
    touches_edge_right = []
    coords_x = []
    coords_y = []

    # Now that we've extracted the boundaries, load up the npz and adjust the coordinates
    for gt_group in gtcodes_df:
        for obj in obj_bounds[gt_group['id']]:
            obj_x = obj[:,0]
            obj_y = obj[:,1]

            #print(f"obj_x: {obj_x}")
            #print(f"obj_y: {obj_y}")

            obj_x_str = ','.join([str(int(x)) for x in obj_x])
            obj_y_str = ','.join([str(int(y)) for y in obj_y])

            group.append('ai_'+gt_group['id'])
            color.append(gt_group['lineColor'])
            ymin.append(int(np.min(obj_y)))
            xmin.append(int(np.min(obj_x)))
            ymax.append(int(np.max(obj_y)))
            xmax.append(int(np.max(obj_x)))

            has_holes.append(0)
            touches_edge_top.append(0)
            touches_edge_left.append(0)
            touches_edge_bottom.append(0)
            touches_edge_right.append(0)

            coords_x.append(obj_x_str)
            coords_y.append(obj_y_str)

    # Put it all into a dataframe
    return pd.DataFrame({
        'group': group,
        'color': color,
        'ymin': ymin,
        'ymax': ymax,
        'xmin': xmin,
        'xmax': xmax,
        'has_holes': has_holes,
        'touches_edge-top': touches_edge_top,
        'touches_edge-left': touches_edge_left,
        'touches_edge-bottom': touches_edge_bottom,
        'touches_edge-right': touches_edge_right,
        'coords_x': coords_x,
        'coords_y': coords_y
    })

def extract_gt_data(label_path, contours_path, gtcodes_df, resize_fraction):
    '''Extracts the ground truth data into contours.

    Consider moving outside this script (into a makefile?)'''

    img = Image.open(label_path)
    w, h = img.size

    print(f"Image mode: {img.mode}")

    #img = img.convert("P")
    #print(f"Image mode: {img.mode}")

    # Original image shape
    # Resize the image to make analysis easier / faster
    img = img.resize((int(w*0.25), int(h*0.25)), Image.NEAREST)

    print(f"Converting to array and uint8...")
    img = np.array(img)
    img = np.uint8(img)

    # Separate the image labelmap into its channels
    print(f"Splitting image...")
    img_red = img[:,:,0]
    img_green = img[:,:,1]
    img_blue = img[:,:,2]

    obj_centroids = {}
    obj_bounds = {}

    # Cycle through the groups, and pull out binary images as necessary
    for gt_group in gtcodes_df:
        # Parse the color of this group
        #print(f"GT Group: {gt_group}")
        label_rgb = gt_group['lineColor']
        rgb_color = label_rgb.split('rgb(')[1][:-1].split(',')
        img_bin = (img_red == int(rgb_color[0])) & (img_green == int(rgb_color[1])) & (img_blue == int(rgb_color[2]))

        # Run the extraction of tumor boundaries -- this takes awhile
        centroids, bounds = extract_object_boundaries(img_bin)

        # Adjust each of the centroids and boundaries by the resize ratio
        for cent_idx, (centroid, bound) in enumerate(zip(centroids, bounds)):
            centroids[cent_idx] = (centroid[0] * 4 * resize_fraction,
                                   centroid[1] * 4 * resize_fraction)
            bounds[cent_idx] = bound * 4 * resize_fraction

        #print(f"Centroids: {centroids}")
        #print(f"bounds: {bounds}")

        obj_centroids[gt_group['id']], obj_bounds[gt_group['id']] = centroids, bounds

    # Save the object centroids and boundaries
    # Use ** syntax with a dictionary to unpack keys as variable names
    # See: https://stackoverflow.com/questions/26427666/use-variables-as-key-names-when-using-numpy-savez
    # np.savez_compressed(contours_path, obj_centroids=obj_centroids,
    # obj_bounds=obj_bounds)

    with open(contours_path, 'wb') as f:
        print(f"Saving to: {contours_path}")
        pickle.dump([obj_centroids, obj_bounds], f)

def get_class_codes_dict(htk_class_dict):
    """Assuming a Histomics-formatted JSON file, apply an index to each class.
    """
    class_codes = {}
    for idx, item in enumerate(htk_class_dict):
        class_codes[item['label']['value']] = idx
    return class_codes
def extract_object_boundaries(img_bin):
    '''Given a binary image, extract the centroid and boundary points of each unique object.

    Returns:
    - obj_centroids: a list of object centroids (from regionprops)
    - obj_bounds: a list of point vertices for each object
    '''
    img_labels = measure.label(img_bin)
    img_props = measure.regionprops(img_labels)

    # Build the sat centroids output
    print(f"Getting centroids...")
    obj_centroids = []
    for img_prop in img_props:
        obj_centroids.append(img_prop.centroid)

    print(f"Centroids obtained")

    # Get the list of labels for this class
    objs = np.unique(img_labels)

    print(f"Found {len(objs)} unique objects.")
    obj_bounds = []

    # Pad by 1 -- need to operate correctly at the boundaries
    img_labels = np.pad(img_labels, 1, 'constant', constant_values=0)

    for obj in objs[1:]:
        # Grab a binary image containing only the current object
        this_obj = img_labels == obj

        # Pad by 1 -- need to operate correctly at the boundaries
        #this_obj = np.pad(this_obj, 1, 'constant', constant_values=0)

        # Use the contour function to grab the boundary points IN CORRECT ORDER
        # The order of the contour points is important; Histomics will freak out
        # if the points are not in "marching" order around the boundary
        cnt = plt.contour(this_obj)
        pts = cnt.collections[0].get_paths()[0].vertices

        obj_bounds.append(pts)

    return obj_centroids, obj_bounds

if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()

    label_path = args.label_path
    env_path = args.env_path
    gtcodes_path = args.gtcodes_path

    # Verify that all necessary local files exist
    verify_files([label_path, env_path, gtcodes_path])

    # Parse environment variables
    load_dotenv(dotenv_path=env_path)
    APIURL = os.getenv('APIURL')
    APIKEY = os.getenv('APIKEY')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME')
    FOLDER_NAME = os.getenv('FOLDER_NAME')

    print(f"Collection name: {COLLECTION_NAME}")
    print(f"FOLDER name: {FOLDER_NAME}")

    # Load and parse histomics class file
    with open(gtcodes_path, 'r') as f:
        gtcodes_dict = json.load(f)
    class_codes = get_class_codes_dict(gtcodes_dict)

    print("GT Codes:")
    for gtcode in gtcodes_dict:
        for k,v in gtcode.items():
            print(f"\t{k}: {v}")

    for k,v in class_codes.items():
        print(f"\t{k}: {v}")

    # Obtain a connection to the server
    conn = get_connection(APIURL, APIKEY)

    # Grab the corresponding WSI name for the provided PNG name
    wsi_name = os.path.splitext(os.path.basename(label_path))[0]
    print(f"WSI name: {wsi_name}")

    # Get the IDs of the target sample
    wsi_id = get_sample_id(wsi_name+'.tiff', conn, COLLECTION_NAME, FOLDER_NAME)

    print(f"WSI ID: {wsi_id}")

    # Get WSI metadata from Histomics
    item_metadata = get_image_metadata(conn, wsi_id)

    for k,v in item_metadata.items():
        print(f"\t{k}: {v}")

    # Get the info for the local image
    label_img = Image.open(f"{label_path}")
    label_w, label_h = label_img.size

    print(f"Width: {label_w}, Height: {label_h}")

    # Calculate the resize ratio between local PNG and Histomics WSI
    print(f"{item_metadata['sizeX']/label_w}")
    print(f"{item_metadata['sizeY']/label_h}")
    resize_ratio = np.mean([item_metadata['sizeX']/label_w, item_metadata['sizeY']/label_h])

    # Grab the contours from the PNG
    print(f"Resize ratio: {resize_ratio}")
    contours_df = load_contours_from_png(label_path, gtcodes_dict, resize_ratio)

    print(f"Contours DF: ")
    print(f"{contours_df.head()}")

    print(f"{contours_df.info()}")
    print(f"{contours_df.describe()}")

    # Modify the annotation boundaries -- for cases where mpp differs between
    # Create the annotation "document", aka the JSON object that will be pushed to the server
    # Here's where we set up the offsets (top and left coords) plus the (non-color) display properties

    annprops = {
        'X_OFFSET': 0,
        'Y_OFFSET': 0,
        'opacity': 0.2,
        'lineWidth': 4.0,
    }

    annotation_docs = get_annotation_documents_from_contours(
        contours_df.copy(),
        separate_docs_by_group=True,
        annots_per_doc=200,
        docnamePrefix='AI_output',
        annprops=annprops,
        verbose=False,
        monitorPrefix=wsi_name + ": annotation docs")

    # Post the annotation documents you created to the server
    for annotation_doc in annotation_docs:
        resp = conn.post(
            "/annotation?itemId=" + wsi_id, json=annotation_doc)
