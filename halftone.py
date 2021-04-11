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
        self.size = size / 40


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


def main():
    """
    Simulate halftoning.
    """

    # create some constants
    main_path = "/Users/thibautvoirand/creation/programmation/halftone/halftone/test"
    input_image = os.path.join(main_path, "lenna.png")
    output_image = os.path.join(main_path, "output.png")

    # read input image
    img = io.imread(input_image)
    rscreen = Screen((0, 0), 5, img[:, :, 0])
    gscreen = Screen((0, 0), 5, img[:, :, 1])
    bscreen = Screen((0, 0), 5, img[:, :, 2])
    rscreen.display(plt, colorstr="r")
    gscreen.display(plt, colorstr="g")
    bscreen.display(plt, colorstr="b")
    plt.show()


if __name__ == "__main__":

    main()
