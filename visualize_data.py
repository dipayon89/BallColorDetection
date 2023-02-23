import numpy as np
import os
from skimage import io
from skimage.color import rgb2hsv
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_file_rgb(file_to_read):
    image = io.imread(file_to_read.path)
    image_r, image_g, image_b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    (w, h) = image_r.shape
    print(("dimension", w, h))

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title('red')
    ax[0, 0].imshow(image_r, cmap='gray')
    ax[0, 1].set_title('green')
    ax[0, 1].imshow(image_g, cmap='gray')
    ax[0, 2].set_title('blue')
    ax[0, 2].imshow(image_b, cmap='gray')

    red = image_r.flatten()
    green = image_g.flatten()
    blue = image_b.flatten()

    x = np.linspace(0, 256, 256)

    ax[1, 0].set_title('red')
    ax[1, 0].hist(red[red != 0], density=True, bins=x, histtype='stepfilled', alpha=0.4, color='r')
    mur, stdr = norm.fit(red[red != 0])
    print((file_to_read, "red", mur, stdr))
    pr = norm.pdf(x, mur, stdr)
    ax[1, 0].plot(x, pr, 'k', linewidth=1)

    ax[1, 1].set_title('green')
    ax[1, 1].hist(green[green != 0], density=True, bins=x, histtype='stepfilled', alpha=0.4, color='g')
    mug, stdg = norm.fit(green[green != 0])
    print((file_to_read, "green", mug, stdg))
    pg = norm.pdf(x, mug, stdg)
    ax[1, 1].plot(x, pg, 'k', linewidth=1)

    ax[1, 2].set_title('blue')
    ax[1, 2].hist(blue[blue != 0], density=True, bins=x, histtype='stepfilled', alpha=0.4, color='b')
    mub, stdb = norm.fit(blue[blue != 0])
    print((file_to_read, "blue", mub, stdb))
    pb = norm.pdf(x, mub, stdb)
    ax[1, 2].plot(x, pb, 'k', linewidth=1)
    plt.show()

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('red')
    ax.set_ylabel('green')
    ax.set_zlabel('blue')
    ax.scatter(red[red != 0], green[green != 0], blue[blue != 0])
    plt.show()


def visualize_file_hsv(file_to_read):
    image = io.imread(file_to_read.path)
    hsv_image = rgb2hsv(image)
    image_h, image_s, image_v = hsv_image[:, :, 0] * 360, hsv_image[:, :, 1] * 100, hsv_image[:, :, 2] * 100

    (w, h) = image_h.shape

    hue = image_h.flatten()
    saturation = image_s.flatten()
    value = image_v.flatten()

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title('hue')
    ax[0, 0].imshow(image_h, cmap='gray')
    ax[0, 1].set_title('saturation')
    ax[0, 1].imshow(image_s, cmap='gray')
    ax[0, 2].set_title('value')
    ax[0, 2].imshow(image_v, cmap='gray')

    ax[1, 0].set_title('hue')
    ax[1, 0].hist(hue[hue != 0], density=True, bins='auto', histtype='stepfilled', color='r')
    mur, stdr = norm.fit(hue[hue != 0])
    print(mur)
    print(stdr)
    xr = np.linspace(0, 360, 360)
    pr = norm.pdf(xr, mur, stdr)
    ax[1, 0].plot(xr, pr, 'k-', lw=2, label='hue')

    ax[1, 1].set_title('saturation')
    ax[1, 1].hist(saturation[saturation != 0], density=True, bins='auto', histtype='stepfilled', color='g')
    mug, stdg = norm.fit(saturation[saturation != 0])
    xg = np.linspace(0, 100, 100)
    pg = norm.pdf(xg, mug, stdg)
    ax[1, 1].plot(xg, pg, 'k-', lw=2, label='saturation')

    ax[1, 2].set_title('value')
    ax[1, 2].hist(value[value != 0], density=True, bins='auto', histtype='stepfilled', color='b')
    mub, stdb = norm.fit(value[value != 0])
    xb = np.linspace(0, 100, 100)
    pb = norm.pdf(xb, mub, stdb)
    ax[1, 2].plot(xb, pb, 'k-', lw=2, label='value')

    plt.show()

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('hue')
    ax.set_ylabel('saturation')
    ax.set_zlabel('value')
    ax.scatter(hue[hue != 0], saturation[saturation != 0], value[value != 0])
    plt.show()


def main():
    directory = './samples'
    i = 0
    for filename in os.scandir(directory):
        i = i + 1
        if filename.is_file():
            visualize_file_rgb(filename)
            visualize_file_hsv(filename)
            if i == 1:
                break


if __name__ == "__main__":
    main()
