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
                 channel_last: bool = False):
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
        channel_last (bool): if true then tensor is of shape (H,W,3), otherwise (3,H,W)

        """
        self.all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]

        self.img_size = img_size
        self.channel_last = channel_last
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
        for i in self.all_data_types:
            filepath = data_dict[i]
            if filepath is not None:
                # need to resample postimg to same spatial resolution/extent as preimg and labels.
                image = io.imread(filepath)
                if i == "postimg":
                    # TODO check if is BILINEAR, required for flood
                    transform.resize(image, (self.img_size[1], self.img_size[0]), anti_aliasing=True)

                if not self.channel_last and len(image.shape)==3:
                    image = np.moveaxis(image, -1, 0)
                if len(image.shape)==2: # add a channel axis if read image is only shape (H,W).
                    returned_data.append(torch.unsqueeze(torch.from_numpy(image), dim=0).float())
                else:
                    returned_data.append(torch.from_numpy(image).float())
            else:
                returned_data.append(0)

        return returned_data

    def get_image_filename(self, index: int) -> str:
        """ return pre-event image absolute filepath at index """
        data_dict = self.files[index]
        return data_dict["preimg"]
