import numpy as np
import matplotlib.pyplot as plt


def get_mean_and_cov(pixel_image):
    """
    Get the mean and covariance of the pixel values
    :param pixel_values:
        shape (x_size, y_size), each value indicating RF intensity
    :return:
        mean location with shape(2,) weighted by pixel,
        and covariance matrix with shape (2, 2), diagonal being variance in x and y directions
    """
    x_size, y_size = pixel_image.shape
    x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    x = x.flatten()
    y = y.flatten()
    flat_pixel_image = pixel_image.flatten()
    mean_x = np.sum(x * flat_pixel_image) / np.sum(flat_pixel_image)
    mean_y = np.sum(y * flat_pixel_image) / np.sum(flat_pixel_image)
    mean = np.array([mean_x, mean_y])
    cov = np.zeros((2, 2))
    for i in range(x_size):
        for j in range(y_size):
            coordinate = np.array([i, j])[:, np.newaxis]
            coordinate_mean = mean[:, np.newaxis]
            cov += pixel_image[i, j] * (coordinate - coordinate_mean) @ (coordinate - coordinate_mean).T
    cov /= np.sum(pixel_image)
    return mean, cov


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


if __name__ == "__main__":
    gaussian_image = makeGaussian(20, fwhm=2, center=(6, 6))
    print(gaussian_image.shape)
    estimated_mean, estimated_cov = get_mean_and_cov(gaussian_image)
    print("Estimated mean:", estimated_mean)
    print("Estimated covariance:", estimated_cov)
    plt.imshow(gaussian_image)
    plt.show()