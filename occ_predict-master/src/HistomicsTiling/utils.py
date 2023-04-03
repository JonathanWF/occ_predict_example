import os
import pickle

import girder_client

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from skimage import segmentation
from imageio import imread

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    parse_slide_annotations_into_tables,
    get_image_from_htk_response # Given a girder response, get np array image
) 

from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours # Converts a contour table into a set of annotation docs for uploading
)

def verify_files(file_list):
    '''Given a list of files, verifies that they all exist.'''
    for f in file_list:
        assert os.path.exists(f), f"Required file or folder {f} does not exist. Please check the provided location and try again."

def get_histomics_connection(APIURL, APIKEY):
    # Authenticate with the server
    gc = girder_client.GirderClient(apiUrl=APIURL)
    gc.authenticate(apiKey=APIKEY)
    return gc

def gt_annotation_exists(slide_annotations):
    """Iterates through a slide annotations response, returns True if any of the annotations have a name that starts with 'groundtruth'."""

    ann_names = []
    for _, ann in enumerate(slide_annotations):
        ann_names.append(ann['annotation']['name'])

    if any([x.startswith('groundtruth_') for x in ann_names]):
        return True
    
    return False
    
def get_collection_id(collection_name, conn):
    '''Given a connection, grab the id of the target collection.'''
    collection_id = None
    
    # List all collections and find the target one
    collection_list = conn.listCollection()
    # assert len(collection_list) > 0, f"Cannot find collection named {collection_name} on Histomics, please check that the server connection is working, that you have access to the collection, and that you are spelling everything correctly."
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
    # assert len(folder_list) > 0, f"Cannot find folder list for collection id {collection_id} on Histomics. Please check that the server connection is working, that you have access to the collection, and that you are spelling everything correctly."
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

def grab_image_data(conn, sample_id, mm_x, mm_y, left, right, top, bottom):
    getStr = f"item/{sample_id}/tiles/region?"+ \
    f"left={left}&"+ \
    f"right={right}&"+ \
    f"top={top}&"+ \
    f"bottom={bottom}&"+ \
    f"mm_x={mm_x}&"+ \
    f"mm_y={mm_y}"
    
    resp = conn.get(getStr, jsonResp=False)
    img_roi = get_image_from_htk_response(resp)
    return img_roi

def extract_gt_data(label_path, contours_path, gtcodes_df, resize_fraction):
    '''Extracts the ground truth data into contours.
    
    Consider moving outside this script (into a makefile?)'''
   
    img = Image.open(label_path)
    w, h = img.size
        
    # Original image shape
    # Resize the image to make analysis easier / faster
    img = img.resize((int(w*resize_fraction), int(h*resize_fraction)), Image.NEAREST)
    img = np.array(img)
    img = np.uint8(img)
        
    # Separate the image labelmap into its channels
    img_red = img[:,:,0]
    img_green = img[:,:,1]
    img_blue = img[:,:,2]
    
    obj_centroids = {}
    obj_bounds = {}

    # Cycle through the groups, and pull out binary images as necessary
    for gt_group in gtcodes_df['group']:
        # Parse the color of this group
        label_rgb = gtcodes_df.loc[gt_group, 'color']
        rgb_color = label_rgb.split('rgb(')[1][:-1].split(',')
        img_bin = (img_red == int(rgb_color[0])) & (img_green == int(rgb_color[1])) & (img_blue == int(rgb_color[2]))
        
        # Run the extraction of tumor boundaries -- this takes awhile
        obj_centroids[gt_group], obj_bounds[gt_group] = extract_object_boundaries(img_bin)
        
    # Save the object centroids and boundaries
    # Use ** syntax with a dictionary to unpack keys as variable names
    # See: https://stackoverflow.com/questions/26427666/use-variables-as-key-names-when-using-numpy-savez
    # np.savez_compressed(contours_path, obj_centroids=obj_centroids,
    # obj_bounds=obj_bounds) 
    
    with open(contours_path, 'wb') as f:
        pickle.dump([obj_centroids, obj_bounds], f)

def extract_object_boundaries(img_bin):
    '''Given a binary image, extract the centroid and boundary points of each unique object.
    
    Returns:
    - obj_centroids: a list of object centroids (from regionprops)
    - obj_bounds: a list of point vertices for each object
    '''
    img_labels = measure.label(img_bin)
    img_props = measure.regionprops(img_labels)
    
    # Build the sat centroids output
    obj_centroids = []
    for img_prop in img_props:
        obj_centroids.append(img_prop.centroid)
    
    # Get the list of labels for this class
    objs = np.unique(img_labels)
    
    obj_bounds = []

    for obj in objs[1:]:
        # Grab a binary image containing only the current object
        this_obj = img_labels == obj

        # Pad by 1 -- need to operate correctly at the boundaries
        this_obj = np.pad(this_obj, 1, 'constant', constant_values=0)

        # Use the contour function to grab the boundary points IN CORRECT ORDER
        # The order of the contour points is important; Histomics will freak out
        # if the points are not in "marching" order around the boundary
        cnt = plt.contour(this_obj)
        pts = cnt.collections[0].get_paths()[0].vertices

        obj_bounds.append(pts)
    
    return obj_centroids, obj_bounds

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
    for gt_group in gtcodes_df['group']:
        for obj in obj_bounds[gt_group]:
            obj_x = obj[:,0]
            obj_y = obj[:,1]
            
            obj_x_str = ','.join([str(int(x / resize_fraction)) for x in obj_x[::3]])
            obj_y_str = ','.join([str(int(y / resize_fraction)) for y in obj_y[::3]])

            group.append(gt_group)
            color.append(gtcodes_df.loc[gt_group, 'color'])
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

def roi_wsi_data(roi_data_path, roi_name):
    '''Pulls a single row from the roi dataframe.'''
    # Read in the pandas dataframe
    roi_df = pd.read_excel(roi_data_path, sheet_name='region_of_interest key')

    # Drop any row that has NaN in the important columns
    roi_df.dropna(subset=['corresponding_wsi_number', 'top', 'left'], inplace=True)

    # Drop unnecessary columns
    roi_df.drop(columns=['roi_number_modified', 'notes', 'wsi_uploaded_histomics', 'gt_exists', 'gt_uploaded', 'gt_confirmed'], inplace=True)
    
    # Grab the row that corresponds with the target ROI
    roi_df = roi_df[roi_df.roi_number_orig==roi_name]

    return roi_df
