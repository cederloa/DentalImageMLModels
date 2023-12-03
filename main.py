# Author: Antti Cederlöf
#
# Dataset: Panoramic Dental X-rays With Segmented Mandibles.
# A. H. Abdi, S. Kasaei, and M. Mehdizadeh, “Automatic segmentation of
# mandible in panoramic x-ray,” J. Med. Imaging, vol. 2, no. 4, p. 44003, 2015.

import torch
from torchdata.datapipes.iter import FileOpener, FileLister
from matplotlib import pyplot as plt
import numpy as np
import glob
from torchvision.io import read_image


PATH = 'Imagedata'

# Read from .tfrec-file. Can't parse the raw images with pytorch
def read_tfrecord_data():
    datapipe1 = FileLister(PATH, '*.tfrec')
    datapipe2 = FileOpener(datapipe1, mode='b')
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()

    i = 0
    for example in tfrecord_loader_dp:
        i += 1
        if i == 1:
            print(example['image_raw'].pop().decode('ascii'))
            example_np = np.frombuffer(example['image_raw'].pop(), dtype=np.uint8)
            print(example_np.shape)
            plt.imshow(np.reshape(example_np, (example['height'], example['width'])))
            plt.show()


def read_png_data():
    first = True
    for image_path in glob.glob(PATH + '/Images/*.png'):
        image = read_image(image_path)
        if first:
            first = False
            plt.imshow(image[0,:,:])
            plt.show()
            imstack = image
            continue
        
        print(image.shape)
        #imstack = torch.cat((imstack, image), 0)
    print(imstack.shape)


def main():
    read_png_data()


if __name__ == '__main__':
    main()
