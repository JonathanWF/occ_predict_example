"""Evaluates the model"""

import argparse
import logging
import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import utils
import model.net as net
import model.data_loader as data_loader
import scipy.misc
import random
import matplotlib
from math import floor
from math import ceil
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/processed', 
    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', 
    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', 
    help="name of the file in --model_dir containing weights to load ('best' or 'last')")

def evaluate(model, loss_fn, dataloader, metrics, params, model_dir='experiments/base_model', save_figs=True):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) a directory to save the output files
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # Create the output directory if it doesn't exist
    img_save_dir = os.path.join(model_dir, 'output_imgs')
    logging.info(f"Saving to {img_save_dir}")
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    logging.info(f"Length of the dataloader: {len(dataloader)}")
    # compute metrics over the dataset
    for sample in dataloader:

        data_batch, labels_batch, names_batch = sample['image'], sample['mask'], sample['name']

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda() #same as train, why is this not working?
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch).float(), Variable(labels_batch).float()
        
        # compute model output
        logging.info(f"Running model on new batch")
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch.long())

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        data_batch = data_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data.item()
        summ.append(summary_batch)

        if save_figs:
            # Save images
            logging.info(f"Saving figures")
            for idx in np.arange(np.shape(output_batch)[0]):
                thisoutput = output_batch[idx,:,:,:]
                thisimg = data_batch[idx,:,:,:]

                # Move the class / channel axis to the end
                thisoutput = np.moveaxis(thisoutput, 0, -1)
                thisimg = np.moveaxis(thisimg, 0, -1)

                # Save the input image
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                logging.info(os.getcwd())
                savepath = os.path.join(img_save_dir, '{}_input.png'.format(names_batch[idx]))
                #savepath = os.path.join("/user", "jwfolmsb", '{}_input.png'.format(names_batch[idx]))
                logging.info(f"Saving to {savepath}")
                im = Image.fromarray(thisimg.astype('uint8'))
                
                # Have to convert image to RGB mode
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(savepath)
                logging.info(f"Image Saved")
                # Use the utilities function to create the RGB output map
                output_dictionary = {
                    (255, 255, 255): 0, # Ignore
                    (128, 128, 128): 1, # Background
                    (255, 255, 0): 2,   # Lymphocytes
                    (255, 0, 0): 3,     # Stroma
                    (0, 0, 255): 4,     # Tumor
                    (0, 255, 255): 5,   # Mucosa
                    (128, 0, 0): 6,     # Adipose 
                    (0, 128, 0): 7,     # Blood
                    (128, 128, 0): 8,   # Muscle Tissue
                    (255, 128, 0): 9,   # Nerves
                    (0, 0, 0): 10,      # Necrosis
                    (0, 0, 128): 11,    # Keratin Pearls
                    (255, 0, 255): 12   # Junk
                }

                im = Image.fromarray(utils.probs2rgb(thisoutput, output_dictionary).astype('uint8'))
                savepath = os.path.join(img_save_dir, '{}_output.png'.format(names_batch[idx]))
                #savepath = os.path.join("/user", "jwfolmsb", '{}_output.png'.format(names_batch[idx]))

                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(savepath)
                logging.info(f"Label saved")
                # Cycle through the class probabilities
                for c in np.arange(np.shape(thisoutput)[2]):

                    # Save this output image
                    savepath = os.path.join(img_save_dir, '{}_class_{}.png'.format(names_batch[idx], c))
                    im = Image.fromarray(thisoutput[:,:,c]*255)
                    if im.mode != 'RGB':
                        im = im.convert('RGB')
                    im.save(savepath)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
    
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'infer.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['infer'], args.data_dir, params)
    test_dl = dataloaders['infer']

    print('Length of dataloader: {}'.format(len(test_dl)))
    logging.info("- done.")

    # Define the model
    model = net.Unet(params).cuda() if params.cuda else net.Unet(params)
    
    # Define loss function and associated metrics
    loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    weight_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    assert os.path.exists(weight_path), f"Could not find weight file at {weight_path}"
    utils.load_checkpoint(weight_path, model, [], params)
    logging.info("Weights loaded")

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params, args.model_dir, save_figs=True)
    save_path = os.path.join(args.model_dir, "metrics_infer_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
