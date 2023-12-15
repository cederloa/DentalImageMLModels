# Author: Antti Cederlöf
#
# Dataset:
# Zhang, Yifan; Ye, Fan; Chen, Lingxiao; Xu, Feng; Chen, Xiaodiao;
# Wu, Hongkun; et al. (2023). Children's Dental Panoramic Radiographs Dataset
# for Caries Segmentation and Dental Disease Detection. figshare. Collection.
# https://doi.org/10.6084/m9.figshare.c.6317013.v1

import torch
from torchdata.datapipes.iter import FileOpener, FileLister
import glob
from torchvision.io import read_image


DATASET_PATH = 'Imagedata/ChildrensPanoramic'


def read_data(folder_path):
    first = True
    for image_path in glob.glob(folder_path):
        image = read_image(image_path)
        if first:
            first = False
            imstack = image
            continue
        imstack = torch.cat((imstack, image), 0)
    return imstack


def main():
    # Save the test images and their paths
    X_test = read_data(DATASET_PATH + '/Children\'s dental caries segmentation dataset/Train/images/*.png')
    Y_test = read_data(DATASET_PATH + '/Children\'s dental caries segmentation dataset/Train/mask/*.png')


if __name__ == '__main__':
    main()
