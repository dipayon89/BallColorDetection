import numpy as np
import os
from skimage import io
from skimage.color import rgb2gray

from roipoly import RoiPoly

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def process_file(file_to_read):
    original = io.imread(file_to_read.path)
    img = rgb2gray(original)

    # Show the image
    fig = plt.figure()
    plt.imshow(original, interpolation='nearest', cmap="Greys")
    plt.title("left click: line segment         right click or double click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    roi = RoiPoly(color='r', fig=fig)

    # binary mask to integer mask conversion
    mask = np.uint8(roi.get_mask(img) * 255)

    # 1 Channel mask to 3 Channel mask conversion
    three_d_mask = np.array([mask, mask, mask])
    # [3, 160, 120] to [160, 120, 3] axis conversion
    three_d_mask = np.einsum('ijk->jki', three_d_mask)
    print(three_d_mask.shape)

    # applied masked image over original image
    masked_image = np.bitwise_and(original, three_d_mask)

    plt.imshow(masked_image, interpolation='nearest', cmap="Greys")

    # Show ROI masks
    plt.title('Output image after applying mask')
    plt.show()

    fileToSave = os.path.join('./samples', file_to_read.name)
    io.imsave(fileToSave, masked_image)


def main():
    directory = './data/train/'
    for filename in os.scandir(directory):
        if filename.is_file():
            process_file(filename)


if __name__ == "__main__":
    main()
