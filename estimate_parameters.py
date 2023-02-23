import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from skimage import filters
from skimage import io
from skimage import measure
from skimage.color import rgb2gray


def compute_parameters():
    sample_pixels = []
    directory = './samples'
    for filename in os.scandir(directory):
        if filename.is_file():
            image = io.imread(filename.path)
            sample_pixels.extend(image[image[:, :, 0] != 0])

    sample_pixels = np.array(sample_pixels)

    # calculate mean
    SUM = [0, 0, 0]
    SUM = np.array(SUM)
    for Xi in sample_pixels:
        SUM = SUM + Xi
    mu = SUM / len(sample_pixels)

    # calculate sigma
    sigma_sum = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    sigma_sum = np.array(sigma_sum)
    for Xi in sample_pixels:
        dif = (Xi - mu)
        # transpose of 1D array returns same 1D array.
        # So for Transpose to Work properly we need to make it 2D
        difT = np.atleast_2d(dif).T
        sigma_sum += dif * difT

    sigma = sigma_sum / len(sample_pixels)

    # print(np.mean(sample_pixels, axis=0))
    # print(np.cov(sample_pixels.T))
    return mu, sigma


def detect_yellow_ball(image_file, mu, sigma, threshold):
    image = io.imread(image_file)
    mv_pdf = multivariate_normal(mean=mu, cov=sigma)
    masked_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_color = image[i, j]
            probability_of_target_color = mv_pdf.pdf(pixel_color)
            if probability_of_target_color >= threshold:
                masked_image[i, j] = 0xFF

    im = filters.gaussian(masked_image, sigma=6)
    gray = rgb2gray(im)
    binary = gray > gray.max() * .5

    labels = measure.label(binary)

    regions = measure.regionprops(labels)

    x = 0
    y = 0
    for props in regions:
        y, x = props.centroid
        break

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].set_title('input image')
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 1].set_title('masked image')
    ax[0, 1].imshow(masked_image, cmap='gray')
    ax[1, 0].set_title('blurred image')
    ax[1, 0].imshow(gray, cmap='gray')
    ax[1, 1].set_title('output image')
    ax[1, 1].imshow(labels, cmap='gray')
    ax[1, 1].plot(x, y, linewidth=2, marker='+')
    plt.show()

    return x, y, binary


def main():
    threshold = 0.70e-05  # min 6.039705441729687e-90 max 4.805736134101579e-05
    mu, sigma = compute_parameters()
    print(mu, sigma)

    directory = './test'
    for filename in os.scandir(directory):
        if filename.is_file():
            x, y, binary = detect_yellow_ball(filename.path, mu, sigma, threshold)
            print("center of yellow ball is at : ", x, y)


if __name__ == "__main__":
    main()
