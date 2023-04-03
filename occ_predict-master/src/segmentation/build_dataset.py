"""Split the images dataset into train/val/test.

Assumes this contains two sub-folders, "train" (which also contains labelmaps), and "test" (which may not contain labels).

Args:
    data_dir: Directory that holds the raw dataset.
    output_dir: Place where the split folders will live ('train', 'test', 'val')
"""

import argparse
import random
import os
import numpy as np
from PIL import Image
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/raw', 
    help="Directory with the dataset")
parser.add_argument('--output_dir', default='data/processed', 
    help="Where to write the new data")

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the existing train / test directories
    train_data_dir = os.path.join(args.data_dir, 'train')
    test_data_dir = os.path.join(args.data_dir, 'test')

    # Get the filenames in each directory
    # Change these lines if your filenames end in something else
    filenames = os.listdir(train_data_dir)
    filenames = [f for f in filenames if f.endswith('.jpg')]

    print('Found {} files in {}'.format(len(filenames), train_data_dir))
	
    # Do the same for the test dataset
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [f for f in test_filenames if f.endswith('.jpg')]

    # Split the images into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Process each split in turn
    for split in ['train', 'val', 'test']:
        
        # Define where this split's data will live
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))

        # Switch for the input directory
        # This section will need to change depending on how your dataset is 
        # organized -- if your raw images are separate from your masks / labels,
        # you'll need to specify different input directories here.
        if split in ['train', 'val']:
            input_dir_split = os.path.join(args.data_dir, 'train')
        else:
            input_dir_split = os.path.join(args.data_dir, 'test')

        if not os.path.exists(output_dir_split):
            os.makedirs(output_dir_split)
        else:
            print("Warning: split dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))

        for filename in filenames[split]:
            # Grab the input and output filenames for your images & masks.
            # This section also changes based on how your names are set up.
            img_input = os.path.join(input_dir_split, filename)
            mask_input = os.path.join(input_dir_split, filename.replace('.jpg', '.png'))

            img_output = os.path.join(output_dir_split, filename)
            mask_output = os.path.join(output_dir_split, filename.replace('.jpg', '.png'))

            # Open your images and masks, and perform any pre-processing here 
            # before saving them to the output directories.
            image = Image.open(img_input)
            mask = np.array(Image.open(mask_input))

            # Now save them again to the output filenames
            image.save(img_output)
            Image.fromarray(mask).save(mask_output)

    print("Done building dataset")
