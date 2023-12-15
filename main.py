# Author: Antti Cederlöf

from readChildrensPanoramic import read_data
from matplotlib import pyplot as plt
import torch


DATASET_PATH = 'Imagedata/ChildrensPanoramic'


def main():
    # Save the test images and their paths
    X_test = read_data(DATASET_PATH + '/Children\'s dental caries segmentation dataset/Train/images/*.png')
    Y_test = read_data(DATASET_PATH + '/Children\'s dental caries segmentation dataset/Train/mask/*.png')
    plt.imshow(Y_test[0,:,:])
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()