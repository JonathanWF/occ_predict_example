import json
import logging
import os
import shutil
from PIL import Image
import torch
import model.net as net
import numpy as np
import argparse
import scipy.misc
import random
import matplotlib
import numpy as np
from math import floor
from math import ceil
Image.MAX_IMAGE_PIXELS = None

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, params=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if params.cuda:
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
	
def rgb2label(img, color_codes = None):
    if color_codes is None:
        #color_codes = {val:i for i,val in enumerate(set( tuple(v) for m2d in img for v in m2d ))}
        print('You need to provide a label map')
        return img
    n_labels = len(color_codes)
    vals = list(img.getdata())
    h, w, d = np.shape(img)
    if len(vals[0]) == 4:
        vals = [x[:-1] for x in vals]
    result = np.ndarray(shape=(h,w), dtype=int)
    result[:,:] = 0
    for rgb, idx in color_codes.items():
        vals_bin = [v == rgb for v in vals]
        vals_bin = np.reshape(vals_bin, (h,w))
        result[vals_bin] = idx	
    return result

def probs2rgb(model_output, color_codes = None):
    """Turns a model output -- where each channel is a class probability -- into the correct color in a single output image.

    Assume the output image has the class in the last axis.
    """
    if color_codes is None:
        #color_codes = {val:i for i,val in enumerate(set( tuple(v) for m2d in img for v in m2d ))}
        print('You need to provide a label map')
        return model_output
    
    # Create a placeholder for the returned RGB image
    h,w,c = np.shape(model_output)
    model_maximum = np.argmax(model_output,2)

    output_img = np.zeros((h,w,3))


    # Cycle through the output of the model and place the color codes into the corresponding indices in the output image
    for k,v in color_codes.items():
        x,y = np.where(model_maximum == v)
        output_img[x,y,:] = k
    
    return output_img

def Pad_Image(image, pad_size):
    array = np.array(image)
    array_pad_rows = np.pad(array, ((pad_size, pad_size), (0, 0), (0, 0)), 'symmetric')
    array_padded = np.pad(array_pad_rows, ((0,0), (pad_size, pad_size), (0, 0)), 'symmetric')
    return array_padded

def Tile_Image(data_dir, image_path, label_path, pad_size, nn_input):
    tile_size = nn_input - (pad_size*2)
    original = Image.open(os.path.join(data_dir, image_path))
    width, height = original.size
    rows = ceil(height/tile_size)
    cols = ceil(width/tile_size)

    rows_padding = np.mod(height, tile_size)
    cols_padding = np.mod(width, tile_size)

    num_tiles = rows*cols

 
    left_coords = np.arange(0, width+cols_padding, tile_size)
    top_coords = np.arange(0, height+rows_padding, tile_size)

    

    for row_idx, row in enumerate(top_coords):
        for col_idx, col in enumerate(left_coords):
            top = row
            left = col

            on_right = False
            on_bot = False
            print(f'Row {row_idx}, Col {col_idx}: {row}, {col}')
            # Check to see whether we're on the right or bottom sides
            if row_idx == rows:
                on_bot = True
                #print('BOT')
            if col_idx == cols:
                on_right = True
                #print("RIGHT")


            # If we aren't on bottom or right
            if on_bot:
                if on_right:
                    # Bottom-right corner
                    tile = original.crop((col, row, col+tile_size-cols_padding, row+tile_size-rows_padding))
                    # pad the tile on both bottom and right
                    tile_np = np.array(tile)
                    tile_np = np.pad(tile_np, ((0, rows_padding), (0, cols_padding), (0,0)), 'reflect')
                    tile = Image.fromarray(tile_np)
                    print((col, row, col+tile_size-cols_padding, row+tile_size-rows_padding))
                else:
                    # Just on the bottom
                    print('Bottom')
                    tile = original.crop((col, row, col+tile_size, row+tile_size-rows_padding))
                    tile_np = np.array(tile)
                    tile_np = np.pad(tile_np, ((0, rows_padding), (0, 0), (0,0)), 'reflect')
                    tile = Image.fromarray(tile_np)
                    print((col, row, col+tile_size, row+tile_size-rows_padding))
            elif on_right:
                    # Just on the right
                    print('Right')
                    tile = original.crop((col, row, col+tile_size-cols_padding, row+tile_size))
                    tile_np = np.array(tile)
                    tile_np = np.pad(tile_np, ((0, 0), (0, cols_padding), (0,0)), 'reflect')
                    tile = Image.fromarray(tile_np)
                    print((col, row, col+tile_size-cols_padding, row+tile_size))
            else:
                # In the middle
                tile = original.crop((col, row, col+tile_size, row+tile_size))
            tile = Pad_Image(tile, pad_size)
            tile = Image.fromarray(tile, 'RGB')
            tile.save(os.path.join(data_dir, f'{image_path}_row_{row}_col_{col}.png'))

            """Generates dummy labels for all images in the folder.  Needs to be have a switch added so it only activates if labels are not already present."""
            label = Image.open(os.path.join(data_dir, label_path))
            label_crop = label.crop((0, 0, nn_input, nn_input))
            label_crop.save(os.path.join(data_dir, f'{image_path}_row_{row}_col_{col}label.png'))
