# Author: Antti Cederlöf
#
# Dataset: Panoramic Dental X-rays With Segmented Mandibles.
# A. H. Abdi, S. Kasaei, and M. Mehdizadeh, “Automatic segmentation of
# mandible in panoramic x-ray,” J. Med. Imaging, vol. 2, no. 4, p. 44003, 2015.

import torch
from torchdata.datapipes.iter import FileOpener, FileLister
from matplotlib import pyplot as plt
import numpy as np

PATH = 'Imagedata'

# Read from .tfrec-file. Can't parse the raw images at the moment
def read_tfrecord_data():
    datapipe1 = FileLister(PATH, '*.tfrec')
    datapipe2 = FileOpener(datapipe1, mode='b')
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()

    i = 0
    for example in tfrecord_loader_dp:
        i += 1
        if i == 5:
            example_tensor = torch.frombuffer(example['image_raw'].pop(),
                                              dtype=torch.uint8)
            print(example_tensor.shape)
            plt.imshow(torch.reshape(example_tensor,
                                     (example['height'],example['width'])))
            plt.show()

def main():
    read_tfrecord_data()


if __name__ == '__main__':
    main()
