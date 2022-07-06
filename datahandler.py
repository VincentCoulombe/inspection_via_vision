import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils, datasets
import os,glob,cv2

class DAGMDataset(Dataset):
    """DAGM Dataset"""

    def __init__(self,root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
         
         on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted(glob.glob(os.path.join(self.root_dir,'Image','*.PNG')))
        self.mask_names = sorted(glob.glob(os.path.join(self.root_dir,'Mask','*.PNG')))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = io.imread(img_name)
        
        msk_name = self.mask_names[idx]
        mask = io.imread(msk_name)
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image.reshape((1,)+image.shape)
        mask = cv2.resize(mask,(128,128),cv2.INTER_AREA)
        mask = mask.reshape((1,)+mask.shape)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class Normalize(object):
    #Normalize image
    def __call__(self,sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)/255}



def segmentation_dataset(data_dir = '../Class8/', batch_size: int = 32):
    #Returns the dataset for segmentation
    data_transforms = {
        'Train': transforms.Compose([ToTensor(),Normalize()]), #TODO Ajouter de l'augmentation
        'Test': transforms.Compose([ToTensor(),Normalize()]),
    }

    image_datasets = {x: DAGMDataset(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=1)
                  for x in ['Train', 'Test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test']}
    return dataloaders


