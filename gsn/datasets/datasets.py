import csv
import copy
from typing import List, Tuple

from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset

class SN8Dataset(Dataset):
    def __init__(self,
                 csv_filename: str,
                 data_to_load: List[str] = ["preimg","postimg","building","road","roadspeed","flood"],
                 img_size: Tuple[int, int] = (1300,1300),
                 crop_size: Tuple[int, int] = (1024,1024)):
        """ pytorch dataset for spacenet-8 data. loads images from a csv that contains filepaths to the images

        Parameters:
        ------------
        csv_filename (str): absolute filepath to the csv to load images from. the csv should have columns: preimg, postimg, building, road, roadspeed, flood.
            preimg column contains filepaths to the pre-event image tiles (.tif)
            postimg column contains filepaths to the post-event image tiles (.tif)
            building column contains the filepaths to the binary building labels (.tif)
            road column contains the filepaths to the binary road labels (.tif)
            roadspeed column contains the filepaths to the road speed labels (.tif)
            flood column contains the filepaths to the flood labels (.tif)
        data_to_load (list): a list that defines which of the images and labels to load from the .csv.
        img_size (tuple): the size of the input pre-event image in number of pixels before any augmentation occurs.
        crop_size (tuple): the crop size of the input pre-event image in number of pixels, anchor is at top left corner.

        """
        self.all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]

        self.img_size = img_size
        self.crop_size = crop_size
        self.data_to_load = data_to_load

        self.files = []

        dict_template = {}
        for i in self.all_data_types:
            dict_template[i] = None

        with open(csv_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                in_data = copy.copy(dict_template)
                for j in self.data_to_load:
                    in_data[j]=row[j]
                self.files.append(in_data)

        print("loaded", len(self.files), "image filepaths")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_dict = self.files[index]
        
        returned_data = []
        crop_h = self.crop_size[0]
        crop_w = self.crop_size[1]
        for i in self.all_data_types:
            filepath = data_dict[i]
            if filepath is not None:
                # need to resample postimg to same spatial resolution/extent as preimg and labels.
                image = io.imread(filepath)
                if i == "postimg":
                    image = transform.resize(image, (self.img_size[1], self.img_size[0]), preserve_range=True)
                if len(image.shape)==2: # add a channel axis if read image is only shape (H,W).
                    image = image[:crop_h, :crop_w]
                    returned_data.append(torch.unsqueeze(torch.from_numpy(image), dim=0).float())
                else:
                    image = np.moveaxis(image, -1, 0)
                    image = image[:, :crop_h, :crop_w]
                    returned_data.append(torch.from_numpy(image).float())
            else:
                returned_data.append(0)

        return returned_data


flood_classes = torch.Tensor([1, 2, 3, 4])
def get_flood_mask(flood_batch):
    assert flood_batch.shape[1] == 4, f"invalid flood shape: {flood_batch.shape}"
    mask = torch.einsum("bchw,c->bhw", flood_batch, flood_classes.to(flood_batch.device))
    assert torch.max(mask) <= 4, f"overlapping flood masks: {torch.max(mask)}"
    return mask.long()
