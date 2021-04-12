"""
Script to simulate halftoning.
"""

# standard imports
import os

# third party imports
import numpy as np
from skimage import io
from matplotlib import pyplot as plt


class Cell:
    """Cell of a halftone screen."""

    def __init__(self, ulx, uly, size):
        """
        Input:
            -ulx    float
            -uly    float
            -size   float
        """

        self.ulx = ulx
        self.uly = uly
        self.size = size * 10


class Screen:
    """Halftone screen."""

    def __init__(self, shift, resolution, input_array):
        """
        Constructor of halftone screen class.
        Build a halftone screen based on a given input array.
        Input:
            -shift          (int, int)
            -resolution     int
            -input_array    np.array
        """

        # apply constructor arguments
        self.xshift, self.yshift = shift
        self.res = resolution
        self.xsize = int(np.floor(input_array.shape[0] / resolution))
        self.ysize = int(np.floor(input_array.shape[1] / resolution))

        # create screen array
        self.array = np.zeros((self.ysize, self.xsize))
        self.cells = []
        for row in range(self.array.shape[0]):
            for col in range(self.array.shape[1]):
                self.array[row, col] = np.mean(
                    input_array[
                        row * self.res : row * self.res + self.res,
                        col * self.res : col * self.res + self.res,
                    ]
                )
                cell_ulx = np.floor(col * self.res + self.res / 2) + self.xshift
                cell_uly = np.floor(row * self.res + self.res / 2) + self.yshift
                self.cells.append(Cell(cell_ulx, cell_uly, self.array[row, col]))

    def display(self, plt, colorstr="k"):
        """
        Display screen.
        Input:
            -plt        matplotlib.pyplot
            -colorstr   str
        """

        x = np.asarray([c.ulx for c in self.cells])
        y = np.asarray([c.uly for c in self.cells])
        s = np.asarray([c.size for c in self.cells])

        plt.scatter(x, self.array.shape[0] - y, s=s, c=colorstr, alpha=0.5)


def rgb_to_cmyk(img):
    """
    Simple formula for conversion from RGB to CMYK color model.
    Doesn't take into account true color representation on physical device.
    Third axis of input and output array is the color dimension (RGB, and CMYK respectively)
    Input:
        -img    np.array of floats
    Output:
        -       np.array of floats
    """
    black = np.minimum.reduce([1 - img[:, :, 0], 1 - img[:, :, 1], 1 - img[:, :, 2]])
    cyan = (1 - img[:, :, 0] - black) / (1 - black)
    magenta = (1 - img[:, :, 1] - black) / (1 - black)
    yellow = (1 - img[:, :, 2] - black) / (1 - black)
    return np.stack([cyan, magenta, yellow, black], axis=-1)


def main():
    """
    Simulate halftoning.
    """

    # create some constants
    main_path = "/Users/thibautvoirand/creation/programmation/halftone/halftone/test"
    input_image = os.path.join(main_path, "lenna.png")
    output_image = os.path.join(main_path, "output.png")

    # read, normalize input image, and convert from rgb to cmyk
    img = io.imread(input_image) / 255
    img = rgb_to_cmyk(img)

    cscreen = Screen((0, 0), 5, img[:, :, 0])
    mscreen = Screen((1, 0), 5, img[:, :, 1])
    yscreen = Screen((1, 1), 5, img[:, :, 2])
    kscreen = Screen((0, 1), 5, img[:, :, 3])
    cscreen.display(plt, colorstr="b")
    mscreen.display(plt, colorstr="r")
    yscreen.display(plt, colorstr="y")
    kscreen.display(plt, colorstr="k")
    plt.show()


if __name__ == "__main__":

    main()
