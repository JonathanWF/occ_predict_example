"""Given a histomics image collection, this file will download overlapping tiles and their corresponding labels and save them. 

If the image has an annotation stating with 'groundtruth' it will save that as a
PNG as well.
"""

import json
import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image

# Need to pull things like API key from a .env file
from dotenv import load_dotenv

# Custom imports
import utils
import annotation_and_mask_utils
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
    get_idxs_for_annots_overlapping_roi_by_bbox, _get_element_mask,
    _get_and_add_element_to_roi, scale_slide_annotations,
    get_image_from_htk_response, get_scale_factor_and_appendStr)
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_contours_from_mask)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_image_and_mask_from_slide)
Image.MAX_IMAGE_PIXELS = None

from histomicstk.saliency.tissue_detection import (
    get_tissue_mask,
    get_slide_thumbnail
)

#set up time to rest between re-requests because histomics is a poop
from time import sleep



# Set up the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target_image', 
    help="Name of the image to download")
parser.add_argument('--label_save_path', 
    default=os.path.join('./data/histomics_tile_labels'), 
    help="Path to save the .png labels")
parser.add_argument('--tile_save_path', 
    default=os.path.join('./data/histomics_tile_images'), 
    help="Base path to save the .png tiles")
parser.add_argument('--env_path', 
    default=".env", 
    help="Path to an ENV file defining important data (default: /.env)")
parser.add_argument('--gtcodes_path', 
    default=os.path.join('data', 'gt_codes.csv'), 
    help="Path to the GT Codes file (default: data/gt_codes.csv)")
parser.add_argument('--edge_effect', 
    default=5, 
    help="Size (in pixels) of the edge effect we want to compensate for -- image overlap.")
parser.add_argument('--nn_input', 
    default=2000, 
    help="Size (in pixels) of the desired input to the neural network.")
parser.add_argument('--tile_mpp', 
    default='2.0', 
    help="Scale (in microns per pixel) of the desired input to the neural network.")
parser.add_argument('--spreadsheet_path',
    default=os.path.join('data', 'region_of_interest_info.csv'),
    help="Path to the region of interest spreadsheet")
parser.add_argument('--max_retries',
    default=100,
    help="Number of retries if histomicstk server does not respond")


if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()
    tile_save_path_pre = args.tile_save_path
    label_save_path = args.label_save_path
    gtcodes_path = args.gtcodes_path
    edge_effect = int(args.edge_effect)
    nn_input = int(args.nn_input)
    tile_size = nn_input - (edge_effect*2)
    env_path = args.env_path
    tile_mpp = args.tile_mpp
    spreadsheet_path = args.spreadsheet_path
    max_retries=args.max_retries
    
    # Parse environment variables
    load_dotenv(dotenv_path=env_path)
    APIURL = os.getenv('APIURL')
    APIKEY = os.getenv('APIKEY')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME')
    FOLDER_NAME = os.getenv('FOLDER_NAME')
    
    # Create saving folders if they don't exist
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    if not os.path.exists(tile_save_path_pre):
        os.makedirs(tile_save_path_pre)

    # Verify that all necessary local files exist
    utils.verify_files([env_path, gtcodes_path, tile_save_path_pre, label_save_path, spreadsheet_path]) 

    # Obtain a connection to the server
    conn = utils.get_histomics_connection(APIURL, APIKEY)
   
    #get the folder id and iterate through items in the folder
    wsi_collection_id = utils.get_collection_id(COLLECTION_NAME, conn)
    wsi_folder_id = utils.get_folder_id(FOLDER_NAME, conn, wsi_collection_id)
    wsi_list = conn.listItem(wsi_folder_id)

    # load the GT Codes
    # Read the external info for the ground truth classes and check it
    gtcodes_df = pd.read_csv(gtcodes_path)
    gtcodes_df.index = gtcodes_df.loc[:, 'group']
    gtcodes_dict = gtcodes_df.to_dict()
    
    i=0
    length = 158
    for item in wsi_list:
        
        i=i+1
        #pull image from list
  
        wsi_id = item['_id']
        print(wsi_id)
        target_image = item['name']
        print(target_image)

        #determine if it belongs in train, test, or val...need check to see if train, test, or val folders exist
        folder_parser = i/length
        if folder_parser < 0.70:
            tile_save_path = os.path.join(tile_save_path_pre, "train")
        elif folder_parser < 0.85:
            tile_save_path = os.path.join(tile_save_path_pre, "test")
        else: 
            tile_save_path = os.path.join(tile_save_path_pre, "val")
  
        #calculate res to pass into string
        tile_mpp = int(float(tile_mpp))
        mm_x = tile_mpp/1000
        mm_y = tile_mpp/1000
    
     

        #create stuff to downsample the image by scale factor, create catch statement to where if it fails, continue to the next image
        #try, except statements
     
        retry_count = 0
        
        while retry_count<=max_retries:
            try:
                sf, appendStr = annotation_and_mask_utils.get_scale_factor_and_appendStr(conn, wsi_id, MPP=tile_mpp, MAG=None)
                print('Scale factor gotten on first try')
                break
            except:
                sleep(30)  
                conn = utils.get_histomics_connection(APIURL, APIKEY)
                sf, appendStr = annotation_and_mask_utils.get_scale_factor_and_appendStr(conn, wsi_id, MPP=tile_mpp, MAG=None)                
                retry_count = retry_count+1
                print(retry_count)    
        print(sf)
        print(appendStr)
        mod_tile_size = np.ceil(tile_size/sf)

        # Grab a list of all the ground-truth annotations that exist for this WSI
        wsi_annotations = conn.get('/annotation/item/' + wsi_id)
        wsi_annotations = scale_slide_annotations(wsi_annotations, sf)
        #get bounding boxes for annotations
        element_dict = get_bboxes_from_slide_annotations(wsi_annotations)  

        # Grab the sample height and width
        slide_info = conn.get("/item/%s/tiles" % wsi_id)
        slide_width = slide_info['sizeX']
        slide_height = slide_info['sizeY']
 
        #calculate the size of the image after downsampling
        mod_slide_width = np.ceil(slide_width*sf)
        mod_slide_height = np.ceil(slide_height*sf)

        # Now grab the slide thumbnail
        try:
            thumbnail_rgb = get_slide_thumbnail(conn, wsi_id)
        except:
            print(f"Image {target_image} has failed, moving to the next image")
            continue
        (thumb_height, thumb_width, _) = np.shape(thumbnail_rgb)

        # Debug messages
        print(f"Image {target_image}")
        print(f"\tHeight: {slide_height} and width: {slide_width}")
        print(f"\tThumb Height: {thumb_height} and width: {thumb_width}")

        # Calculate thumbnail rescale amount -- Take the average of x and y
        thumb_rescale = np.mean([thumb_height / slide_height, thumb_width / slide_width])

        print(f"\tThumb Rescale Fraction: {thumb_rescale}")
        print()

        # Run the tissue detection on this thumbnail
        labeled, mask = get_tissue_mask(
            thumbnail_rgb, deconvolve_first=False,
            n_thresholding_steps=1, sigma=0.1, min_size=5)
    
        # Modify label -- fill in 
        # Save the thumb and the label
        Image.fromarray(labeled>0).save(os.path.join(tile_save_path, f'{target_image}_thumb_labeled.png'))
        Image.fromarray(mask).save(os.path.join(tile_save_path, f'{target_image}_thumb_mask.png'))
        Image.fromarray(thumbnail_rgb).save(os.path.join(tile_save_path, f'{target_image}_thumbnail.png'))
    
        # After we open the image
        rows = int(np.ceil(slide_height/mod_tile_size))
        cols = int(np.ceil(slide_width/mod_tile_size))

        rows_padding = tile_size - int(np.mod(mod_slide_height, tile_size))+1
        cols_padding = tile_size - int(np.mod(mod_slide_width, tile_size))+1
   

        num_tiles = rows*cols

        print(f'\tSize of tiles: {mod_tile_size}')
        print(f'\tNumber of tiles: {num_tiles}')
        print(f'\tNumber of rows: {rows}')
        print(f'\tNumber of columns: {cols}')

        print(f'\tAmount of row padding: {rows_padding}')
        print(f'\tNumber of column padding: {cols_padding}')

        # Get a list of coordinates from each tile
        left_coords = np.arange(0, slide_width+cols_padding, mod_tile_size)
        top_coords = np.arange(0, slide_height+rows_padding, mod_tile_size)

        for row_idx, row in enumerate(top_coords):
            for col_idx, col in enumerate(left_coords):
                top = row
                left = col

                on_right = False
                on_bot = False
            
                # Check to see whether we're on the right or bottom sides
                if row_idx == rows-1:
                    on_bot = True
                if col_idx == cols-1:
                    on_right = True
            
                # Calculate this tile's coordinates in the thumbnail
                thumb_top = int(np.floor(top * thumb_rescale))
                thumb_left = int(np.floor(left * thumb_rescale))
                thumb_tile_size = int(np.ceil(mod_tile_size * thumb_rescale))
                retry_count = 0
                # If we aren't on bottom or right
                if on_bot:
                    if on_right:
                        # Bottom-right corner; pad tile on bottom and right
                        thumb_tile = mask[thumb_top:thumb_height, thumb_left:thumb_width]
                        if np.sum(thumb_tile) == 0:
                            continue
                        bound_dict = {'XMIN': col, 'XMAX': col+mod_tile_size-cols_padding, 'YMIN': row, 'YMAX': row+mod_tile_size-rows_padding}
                        get_roi_mask_dict = {'crop_to_roi': True}
                        while retry_count<=max_retries:
                            try:
                                result = get_image_and_mask_from_slide(
                                    conn, wsi_id, gtcodes_dict,
                                    MPP=2.0, MAG=None, mode='manual_bounds',
                                    bounds=bound_dict, idx_for_roi=None,
                                    slide_annotations=wsi_annotations, element_infos=element_dict,
                                    get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                    get_rgb=True, get_contours=False, get_visualization=False)
                                tile = result['rgb']
                                tile_annotation = result['ROI']
                                tile_np = np.array(tile)
                                break                                                        
                            except:
                                sleep(30)
                                conn = utils.get_histomics_connection(APIURL, APIKEY)
                                result = get_image_and_mask_from_slide(
                                    conn, wsi_id, gtcodes_dict,
                                    MPP=2.0, MAG=None, mode='manual_bounds',
                                    bounds=bound_dict, idx_for_roi=None,
                                    slide_annotations=wsi_annotations, element_infos=element_dict,
                                    get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                    get_rgb=True, get_contours=False, get_visualization=False)
                                tile = result['rgb']
                                tile_annotation = result['ROI']
                                tile_np = np.array(tile)                              
                                rety_count = retry_count+1
                                print(count)
                        tile_np = np.pad(tile_np, ((0, rows_padding), (0, cols_padding), (0,0)), 'reflect')
                        tile_annotation = np.pad(tile_annotation, ((0, rows_padding), (0, cols_padding)), 'reflect')
                        # Print output
                        print("Bottom right tile")
                        print((col, row, col+mod_tile_size-cols_padding, row+mod_tile_size-rows_padding))
                    else:
                        # Just on the bottom; pad the tile on bottom
                        thumb_tile = mask[thumb_top:thumb_height, thumb_left:thumb_left+thumb_tile_size]
                        if np.sum(thumb_tile) == 0:
                            continue
                        bound_dict = {'XMIN': col, 'XMAX': col+mod_tile_size, 'YMIN': row, 'YMAX': row+mod_tile_size-rows_padding}
                        get_roi_mask_dict = {'crop_to_roi': True}
                        while retry_count<=max_retries:
                            try:
                                result = get_image_and_mask_from_slide(
                                    conn, wsi_id, gtcodes_dict,
                                    MPP=2.0, MAG=None, mode='manual_bounds',
                                    bounds=bound_dict, idx_for_roi=None,
                                    slide_annotations=wsi_annotations, element_infos=element_dict,
                                    get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                    get_rgb=True, get_contours=False, get_visualization=False)
                                tile = result['rgb']
                                tile_annotation = result['ROI']
                                tile_np = np.array(tile) 
                                break                            
                               
                            except:
                                sleep(30)
                                conn = utils.get_histomics_connection(APIURL, APIKEY)
                                result = get_image_and_mask_from_slide(
                                    conn, wsi_id, gtcodes_dict,
                                    MPP=2.0, MAG=None, mode='manual_bounds',
                                    bounds=bound_dict, idx_for_roi=None,
                                    slide_annotations=wsi_annotations, element_infos=element_dict,
                                    get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                    get_rgb=True, get_contours=False, get_visualization=False)
                                tile = result['rgb']
                                tile_annotation = result['ROI']
                                tile_np = np.array(tile)                              
                                retry_count=retry_count+1
                        print(tile_np.shape)
                        print(tile_annotation.shape)
                        tile_np = np.pad(tile_np, ((0, rows_padding), (0, 0), (0,0)), 'reflect')
                        tile_annotation = np.pad(tile_annotation, ((0, rows_padding), (0,0)), 'reflect')
                        # Print output
                        print("Bottom tile")
                        print((col, row, col+mod_tile_size, row+mod_tile_size-rows_padding))
                elif on_right:
                    # Just on the right; pad the tile on the right
                    thumb_tile = mask[thumb_top:thumb_top+thumb_tile_size, thumb_left:thumb_width]
                    if np.sum(thumb_tile) == 0:
                        continue
                    bound_dict = {'XMIN': col, 'XMAX': col+mod_tile_size-cols_padding, 'YMIN': row, 'YMAX': row+mod_tile_size}
                    get_roi_mask_dict = {'crop_to_roi': True}
                    while retry_count<=max_retries:
                        try:
                            result = get_image_and_mask_from_slide(
                                conn, wsi_id, gtcodes_dict,
                                MPP=2.0, MAG=None, mode='manual_bounds',
                                bounds=bound_dict, idx_for_roi=None,
                                slide_annotations=wsi_annotations, element_infos=element_dict,
                                get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                get_rgb=True, get_contours=False, get_visualization=False)
                            tile = result['rgb']
                            tile_annotation = result['ROI']
                            tile_np = np.array(tile)    
                            break                        
                            
                        except:
                            sleep(30)
                            conn = utils.get_histomics_connection(APIURL, APIKEY)
                            result = get_image_and_mask_from_slide(
                                conn, wsi_id, gtcodes_dict,
                                MPP=2.0, MAG=None, mode='manual_bounds',
                                bounds=bound_dict, idx_for_roi=None,
                                slide_annotations=wsi_annotations, element_infos=element_dict,
                                get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                get_rgb=True, get_contours=False, get_visualization=False)
                            tile = result['rgb']
                            tile_annotation = result['ROI']
                            tile_np = np.array(tile)                            
                            retry_count=retry_count+1
                    tile_np = np.pad(tile_np, ((0, 0), (0, cols_padding), (0,0)), 'reflect')
                    tile_annotation = np.pad(tile_annotation, ((0, 0), (0, cols_padding)), 'reflect')
                    # Print output
                    print("Right tile")
                    print((col, row, col+mod_tile_size-cols_padding, row+mod_tile_size))
                else:
                    # In the middle; don't pad anywhere
                    thumb_tile = mask[thumb_top:thumb_top+thumb_tile_size, thumb_left:thumb_left+thumb_tile_size]
                    if np.sum(thumb_tile) == 0:
                        continue
                    #define bound dictionary
                    bound_dict = {'XMIN': col, 'XMAX': col+mod_tile_size, 'YMIN': row, 'YMAX': row+mod_tile_size}
                    get_roi_mask_dict = {'crop_to_roi': True}
                    while retry_count<= max_retries:
                        try:
                            result = get_image_and_mask_from_slide(
                                conn, wsi_id, gtcodes_dict,
                                MPP=2.0, MAG=None, mode='manual_bounds',
                                bounds=bound_dict, idx_for_roi=None,
                                slide_annotations=wsi_annotations, element_infos=element_dict,
                                get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                get_rgb=True, get_contours=False, get_visualization=False)
                            tile = result['rgb']
                            tile_annotation = result['ROI']
                            tile_np = np.array(tile)
                            break

                        except:
                            sleep(30)
                            conn = utils.get_histomics_connection(APIURL, APIKEY)
                            result = get_image_and_mask_from_slide(
                                conn, wsi_id, gtcodes_dict,
                                MPP=2.0, MAG=None, mode='manual_bounds',
                                bounds=bound_dict, idx_for_roi=None,
                                slide_annotations=wsi_annotations, element_infos=element_dict,
                                get_roi_mask_kwargs=get_roi_mask_dict, get_contours_kwargs=None, linewidth=0.2,
                                get_rgb=True, get_contours=False, get_visualization=False)
                            tile = result['rgb']
                            tile_annotation = result['ROI']
                            tile_np = np.array(tile)                            
                            retry_count=retry_count+1
                            print(retry_count)
                    
                # Save the tile
                tile = Image.fromarray(tile_np)
                tile.save(os.path.join(tile_save_path, f'{target_image}_row_{row}_col_{col}.png'))
                tile_annotation = Image.fromarray(tile_annotation)
                tile_annotation.save(os.path.join(tile_save_path, f'{target_image}_row_{row}_col_{col}_annotation.png'))
        print('Next image starting')
    