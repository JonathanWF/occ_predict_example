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

import utils

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    parse_slide_annotations_into_table,
    get_image_from_htk_response # Given a girder response, get np array image
)

from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours # Converts a contour table into a set of annotation docs for uploading
)

# Set up the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--label_path', help="Path to the .png label to upload.")
parser.add_argument('--env_path', default=".env", help="Path to an ENV file defining important data (default: /.env)")
parser.add_argument('--gtcodes_path', default=os.path.join('data', 'gt_codes.csv'), help="Path to the GT Codes file (default: data/gt_codes.csv)")
parser.add_argument('--roi_data_path', default=os.path.join('data', 'histomics_data.xlsx'), help="Path to the Excel spreadsheet containing roi-to-wsi conversions and offsets (default: data/histomics_data.xlsx)")

if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()

    label_path = args.label_path
    env_path = args.env_path
    gtcodes_path = args.gtcodes_path
    roi_data_path = args.roi_data_path

    # Verify that all necessary local files exist
    utils.verify_files([label_path, env_path, gtcodes_path, roi_data_path])

    # Parse environment variables
    load_dotenv(dotenv_path=env_path)
    APIURL = os.getenv('APIURL')
    APIKEY = os.getenv('APIKEY')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME')
    FOLDER_NAME = os.getenv('FOLDER_NAME')

    # Obtain a connection to the server
    conn = utils.get_histomics_connection(APIURL, APIKEY)

    # Grab the corresponding WSI name for the provided PNG name
    roi_name = os.path.splitext(os.path.basename(label_path))[0]
    print(f"ROI name: {roi_name}")

    # Read in the pandas dataframe, verify we've got a valid row
    roi_df = utils.multi_roi_wsi_data(roi_data_path, roi_name)
    assert len(roi_df) > 0, f"Could not find a valid row in "+ \
        f"{roi_data_path} related to ROI name {roi_name}. "+ \
        f"Please check that all necessary columns "+ \
        f"(corresponding_wsi_number, top, left, etc.) are filled out in the spreadsheet."

    # Grab the WSI name
    wsi_name = roi_df['corresponding_wsi_number'].values[0]

    # Get the difference in mpp
    mpp_difference = roi_df['mpp_roi'].values[0] / roi_df['mpp_wsi'].values[0]

    # Debug
    print(f"Image {wsi_name} has a MPP difference of {mpp_difference}")

    # Get the sample ID on Histomics
    wsi_id = utils.get_sample_id(wsi_name+".tiff", conn, COLLECTION_NAME, FOLDER_NAME)

    # Grab a list of all the annotations that exist for this WSI
    wsi_annotations = conn.get('/annotation/item/' + wsi_id)

    # Fail if any of them are prefixed with "groudtruth_"
    #assert not utils.gt_annotation_exists(wsi_annotations), f"Found ground truth annotations for {wsi_name} (id: {wsi_id})"

    # If we've found a labelmap, and we haven't found an existing ground truth,
    # then process the ROI to obtain our ground truth

    # First load the GT Codes
    # Read the external info for the ground truth classes and check it
    gtcodes_df = pd.read_csv(gtcodes_path)
    gtcodes_df.index = gtcodes_df.loc[:, 'group']

    contours_df = utils.load_contours_from_png(label_path, gtcodes_df, resize_fraction, mpp_difference=mpp_difference)

    # Modify the annotation boundaries -- for cases where mpp differs between
    # Create the annotation "document", aka the JSON object that will be pushed to the server
    # Here's where we set up the offsets (top and left coords) plus the (non-color) display properties

    annprops = {
        'X_OFFSET': roi_df.left.values[0],
        'Y_OFFSET': roi_df.top.values[0],
        'opacity': 0.2,
        'lineWidth': 4.0,
    }

    annotation_docs = get_annotation_documents_from_contours(
        contours_df.copy(),
        separate_docs_by_group=True,
        annots_per_doc=10,
        docnamePrefix='groundtruth',
        annprops=annprops,
        verbose=False,
        monitorPrefix=wsi_name + ": annotation docs")

    # Post the annotation documents you created to the server
    for annotation_doc in annotation_docs:
        resp = conn.post(
            "/annotation?itemId=" + wsi_id, json=annotation_doc)
