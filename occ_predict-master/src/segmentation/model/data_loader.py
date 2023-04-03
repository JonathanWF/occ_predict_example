import random
import os
import matplotlib
import numpy as np
import torch

from utils import rgb2label, Pad_Image, Tile_Image

# from skimage import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Note: It is necessary to define a custom set of functions for cropping, flipping, etc. because these transforms are applied randomly to their inputs. In normal cases the inputs are just images and the label doesn't have to be transformed, so we can use regular torchvision.transforms. However in the UNet case our samples are dicts containing both an image and a mask -- BOTH of which need to be transformed (or not) in the same manner. Thus we apply the transforms to `sample`, and have to define the custom transform behavior for what happens to each element of `sample`.

class UniformCrop(object):
    """Crop uniformly from the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask, name = sample['image'], sample['mask'], sample['name']

        # h, w = image.shape[:2]
        h, w = image.size
        new_h, new_w = self.output_size

        top = h//2 - new_h//2
        left = w//2 - new_w//2
        image = image.crop((left, top, left+new_w, top+new_h))
        mask = mask.crop((left, top, left+new_w, top+new_h))
        return {'image': image, 'mask': mask, 'name': name}

class RandomVFlip(object):
    """Randomly flip both image and mask.

    Args:
        p (float): probability of applying a horizontal flip to the image / mask.
    """
    def __init__(self, prob_flip):
        assert isinstance(prob_flip, float)
        self.prob_flip = prob_flip
        
    def __call__(self, sample):
        image, mask, name = sample['image'], sample['mask'], sample['name']

        if np.random.rand() < self.prob_flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': image, 'mask': mask, 'name': name}

class RandomHFlip(object):
    """Randomly flip both image and mask.

    Args:
        p (float): probability of applying a horizontal flip to the image / mask.
    """
    def __init__(self, prob_flip):
        assert isinstance(prob_flip, float)
        self.prob_flip = prob_flip
        
    def __call__(self, sample):
        image, mask, name = sample['image'], sample['mask'], sample['name']

        if np.random.rand() < self.prob_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'mask': mask, 'name': name}

class RandomRotation(object):
    """Randomly rotate both image and mask.

    Args:
        P (float): Probability to enable rotation
        angles (float): (-angles, +angles) angle range to rotate
    """
    def __init__(self, P, angles):
        assert isinstance(angles, int)
        self.angles = (-angles, angles)

        assert isinstance(P, float)
        self.P = P
        
    def __call__(self, sample):
        image, mask, name = sample['image'], sample['mask'], sample['name']
        
        # Get a random angle to rotate
        angle = random.uniform(self.angles[0], self.angles[1])

        if np.random.rand() < self.P:
            image = image.rotate(angle)
            mask = mask.rotate(angle)

        return {'image': image, 'mask': mask, 'name': name}

class RandomJitterHSV(object):
    """Randomly jitter the HSV channels of the image

    Args: 
	p (float): probability to enable each jitter
	hues (float): (-hues, +hues) color range to jitter hue
    	sats (float): (-sats, +sats) color range to jitter sat
    	vals (float): (-vals, +vals) color range to jitter val
    """
    def __init__(self, p, hues, sats, vals):
    	assert isinstance(hues, float)
    	self.hues = (-hues, +hues)
    	assert isinstance(sats, float)
    	self.sats = (-sats, +sats)
    	assert isinstance(vals, float)
    	self.vals = (-vals, +vals)

    	assert isinstance(p, float)
    	self.p = p

    def __call__(self, sample):
    	image, mask, name = sample['image'], sample['mask'], sample['name']
    	
	#convert from HSV to RGB
    	matplotlib.colors.rgb_to_hsv(image)
    	image = np.array(image)

        #Get random hue to jitter by
    	hue = random.gauss(0, self.hues[1])

    	if np.random.rand() < self.p:
    		image[:,:,0]+hue


        #Get random sat to jitter by
    	sat = random.gauss(0, self.sats[1])

    	if np.random.rand() < self.p:
    		image[:,:,1]+sat

        #Get random val to jitter by 
    	val = random.gauss(0, self.vals[1])

    	if np.random.rand() < self.p:
    		image[:,:,2]+val
	
	#Convert back to RGB
    	image = Image.fromarray(image)
    	matplotlib.colors.hsv_to_rgb(image)

    	return {'image': image, 'mask': mask, 'name': name}

class PadImage(object):
    """Pads image by reflection based on the edge effect"""
    def __call__(self, sample):
        image, mask, name = sample['image'], sample['mask'], sample['name']
        
        image = np.array(image)
        image = np.pad(image, ((5, 5), (5, 5), (0, 0)), 'reflect')
        image = Image.fromarray(image)
        
        return {'image': image, 'mask': mask, 'name': name}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask, name = sample['image'], sample['mask'], sample['name']


        

        # Convert to numpy first
        image = np.array(image)
        mask = np.array(mask)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # Have to include .copy() in case the image was flipped
        # See: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy()),
                'name': name}

# Define the transforms by chaining them together
# Commenting out the random flipping because it was throwing an error
train_transformer = transforms.Compose([
    UniformCrop(2000),
    RandomHFlip(0.5),
    RandomVFlip(0.5),
    RandomRotation(0.5, 270),
    ToTensor()])

# Eval transform doesn't do random flipping, and probably won't do random cropping either (we will control the size of the data so that it's uniform)
eval_transformer = transforms.Compose([
    RandomHFlip(0.0),
    ToTensor()])

#inference transformer also doesn't do random transforms
infer_transformer = transforms.Compose([
    PadImage(),
    ToTensor()])

class OCCV3Dataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.

    Be sure to rename the dataset to whatever you want, as long as you also
    change the `fetch_dataloaders` function below.
    """
    def __init__(self, data_dir, transform, params):
        """
        Store the filenames of the pngs to use. Specifies transforms to apply on images. Tiles WSI's.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        """New stuff"""
        #self.data_dir = data_dir
        #self.filenames_preprocess = [f for f in os.listdir(self.data_dir) if "label" not in f]
        #self.labelnames_preprocess = [f for f in os.listdir(self.data_dir) if "label" in f]
        #for i in range(0, len(self.filenames_preprocess)):
            #Tile_Image(data_dir, self.filenames_preprocess[i], self.labelnames_preprocess[0], params.pad_size, params.nn_input)
        """New stuff end"""

        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg') and "annotation" not in f]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image. Also fetch the name of the image (for saving later on).

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            sample: (dict) Dictionary containing:
                image: (Tensor) transformed image
                label: (int) corresponding label of image
                name: (str) string of the sample name
        """
        # PIL images
        image = Image.open(os.path.join(self.data_dir, self.filenames[idx]))
        image = image.convert("RGB")
        mask = Image.open(os.path.join(self.data_dir, self.filenames[idx].replace('.jpg', '.png')))
        dictionary = {
	    (255, 255, 255): 0,
            (128, 128, 128): 1,
            (255, 255, 0): 2,
            (255, 0, 0): 3,
            (0, 0, 255): 4,
            (0, 255, 255): 5,
            (128, 0, 0): 6,
            (0, 128, 0): 7,
            (128, 128, 0): 8,
            (255, 128, 0): 9,
            (0, 0, 0): 10,
            (0, 0, 128): 11,
            (255, 0, 255): 12
            }

        mask = rgb2label(mask, dictionary)
        mask = 1.0*mask
        mask = mask.astype('uint8')
        mask = Image.fromarray(mask)
       
        
        
        
        # Record the filename root
        fnameroot = self.filenames[idx].replace('.png', '')
        
        sample = {'image': image, 'mask': mask, 'name': fnameroot}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class OCCV3InferDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.

    Be sure to rename the dataset to whatever you want, as long as you also
    change the `fetch_dataloaders` function below.
    """
    def __init__(self, data_dir, transform, params):
        """
        Store the filenames of the pngs to use. Specifies transforms to apply on images. Tiles WSI's.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(self.data_dir) if f.endswith('.png') and "label" not in f]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image. Also fetch the name of the image (for saving later on).

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            sample: (dict) Dictionary containing:
                image: (Tensor) transformed image
                label: (int) corresponding label of image
                name: (str) string of the sample name
        """
        # PIL images
        image = Image.open(os.path.join(self.data_dir, self.filenames[idx]))
        image = image.convert("RGB")
        mask = Image.open(os.path.join(self.data_dir, 'dummy_label.png'))
        dictionary = {
	        (255, 255, 255): 0,
            (128, 128, 128): 1,
            (255, 255, 0): 2,
            (255, 0, 0): 3,
            (0, 0, 255): 4,
            (0, 255, 255): 5,
            (128, 0, 0): 6,
            (0, 128, 0): 7,
            (128, 128, 0): 8,
            (255, 128, 0): 9,
            (0, 0, 0): 10,
            (0, 0, 128): 11,
            (255, 0, 255): 12
            }

        mask = rgb2label(mask, dictionary)
        mask = 1.0*mask
        mask = mask.astype('uint8')
        mask = Image.fromarray(mask)
       
        
        
        
        # Record the filename root
        fnameroot = self.filenames[idx].replace('.png', '')
        
        sample = {'image': image, 'mask': mask, 'name': fnameroot}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    for split in ['train', 'val', 'test', 'infer']:
        if split in types:
            path = os.path.join(data_dir, split)
            # use the train_transformer if training data, infer if inference data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(OCCV3Dataset(path, train_transformer, params), 
                    batch_size=params.batch_size, shuffle=True,
                    num_workers=params.num_workers, pin_memory=params.cuda)
            elif split == 'infer':
                dl = DataLoader(OCCV3InferDataset(path, infer_transformer, params),
                    batch_size=params.batch_size, shuffle=True,
                    num_workers=params.num_workers, pin_memory=params.cuda)
            else:
                dl = DataLoader(OCCV3Dataset(path, eval_transformer, params),
                    batch_size=params.batch_size, shuffle=False,
                    num_workers=params.num_workers, pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
