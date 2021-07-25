"""
Script to simulate halftoning.
"""

# standard imports
import os
import datetime
import argparse

# third party imports
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from shapely.geometry import LineString


def write_svg_file(
    output_file, svg_elements, canvas_width, canvas_height, title=None, description=None
):
    """
    Write SVG file.
    Input:
        -output_file    str
        -svg_elements   [str, ...]
            list of strings defining svg elements
        -canvas_width   int
        -canvas_height  int
        -title          str
        -description    str
    """

    with open(output_file, "w") as outfile:

        # write headers
        outfile.write("<?xml version='1.0' encoding='utf-8'?>\n")
        outfile.write(
            "<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='{}' height='{}'>\n".format(
                canvas_width, canvas_height
            )
        )
        if title:
            outfile.write("<title>{}</title>\n".format(title))
        if description:
            outfile.write("<description>{}</description>\n".format(description))

        # write svg elements
        for svg_element in svg_elements:
            outfile.write("{}\n".format(svg_element))

        # write footer
        outfile.write("</svg>")


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
        self.size = size
        self.line = LineString([ulx, uly], [ulx + s, uly + s])


class Screen:
    """Halftone screen."""

    def __init__(self, shift, angle, resolution, input_array):
        """
        Constructor of halftone screen class.
        Build a halftone screen based on a given input array.
        Input:
            -shift          (int, int)
            -angle          float
            -resolution     int
            -input_array    np.array
        """

        # apply constructor arguments
        self.xshift, self.yshift = shift
        self.angle = angle
        self.res = resolution

        # coordinates of image corners in image reference frame
        image_ulx1 = 0
        image_uly1 = 0
        image_urx1 = input_array.shape[1]
        image_ury1 = 0
        image_llx1 = 0
        image_lly1 = input_array.shape[0]
        image_lrx1 = input_array.shape[1]
        image_lry1 = input_array.shape[0]

        # coordinates of image corners in screen reference frame
        image_ulx2 = (
            image_ulx1 * np.cos(self.angle) - image_uly1 * np.sin(self.angle)
        ) / self.res
        image_uly2 = (
            image_ulx1 * np.sin(self.angle) + image_uly1 * np.cos(-self.angle)
        ) / self.res
        image_urx2 = (
            image_urx1 * np.cos(self.angle) - image_ury1 * np.sin(self.angle)
        ) / self.res
        image_ury2 = (
            image_urx1 * np.sin(self.angle) + image_ury1 * np.cos(-self.angle)
        ) / self.res
        image_llx2 = (
            image_llx1 * np.cos(self.angle) - image_lly1 * np.sin(self.angle)
        ) / self.res
        image_lly2 = (
            image_llx1 * np.sin(self.angle) + image_lly1 * np.cos(-self.angle)
        ) / self.res
        image_lrx2 = (
            image_lrx1 * np.cos(self.angle) - image_lry1 * np.sin(self.angle)
        ) / self.res
        image_lry2 = (
            image_lrx1 * np.sin(self.angle) + image_lry1 * np.cos(-self.angle)
        ) / self.res

        # image bounds in screen coordinates
        x2_min = int(np.floor(min([image_ulx2, image_urx2, image_llx2, image_lrx2])))
        x2_max = int(np.ceil(max([image_ulx2, image_urx2, image_llx2, image_lrx2])))
        y2_min = int(np.floor(min([image_uly2, image_ury2, image_lly2, image_lry2])))
        y2_max = int(np.ceil(max([image_uly2, image_ury2, image_lly2, image_lry2])))

        # initialize screen contents
        self.array = np.zeros((y2_max - y2_min, x2_max - x2_min))
        self.cells = []

        # loop through screen cells
        for row in range(self.array.shape[0]):
            for col in range(self.array.shape[1]):

                # cell coordinates in screen reference frame
                x2 = col + x2_min
                y2 = row + y2_min

                # cell coordinates in image reference frame
                x1 = (
                    (x2 * self.res + self.res / 2) * np.cos(-self.angle)
                    - (y2 * self.res + self.res / 2) * np.sin(-self.angle)
                    + self.xshift
                )
                y1 = (
                    (x2 * self.res + self.res / 2) * np.sin(-self.angle)
                    + (y2 * self.res + self.res / 2) * np.cos(-self.angle)
                    + self.yshift
                )

                # get area of image which will be covered by this cell
                xmin = int(np.floor(x1 - self.res / 2))
                xmax = int(np.ceil(x1 + self.res / 2))
                ymin = int(np.floor(y1 - self.res / 2))
                ymax = int(np.ceil(y1 + self.res / 2))

                # get mean color of area of image covered by this cell
                if (  # handle out of image areas
                    xmin < 0
                    or xmax > input_array.shape[1]
                    or ymin < 0
                    or ymax > input_array.shape[0]
                ):
                    self.array[row, col] = 0
                else:
                    self.array[row, col] = np.mean(input_array[ymin:ymax, xmin:xmax])

                self.cells.append(Cell(x1, y1, self.array[row, col]))

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

        plt.scatter(x, -y, s=s * 100, c=colorstr, marker=".", alpha=0.5, linewidths=0)


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


def halftone(input_file, output_file, display_preview=False):
    """
    Simulate halftoning.

    Parameters
    ----------
        input_file: str
        output_file: str
        display_preview: bool
    """

    # create some constants
    screens_res = 10  # in pixels

    # read, normalize input image, and convert from rgb to cmyk
    img = io.imread(input_file) / 255
    img = rgb_to_cmyk(img)

    # create cyan, magenta, yellow and black screens
    cscreen = Screen(
        (0, 0),  # x and y shift
        15 * np.pi / 180,  # angle
        screens_res,  # resolution
        img[:, :, 0],  # input array
    )
    mscreen = Screen(
        (0, 0),  # x and y shift
        75 * np.pi / 180,  # angle
        screens_res,  # resolution
        img[:, :, 1],  # input array
    )
    yscreen = Screen(
        (0, 0),  # x and y shift
        0,  # angle
        screens_res,  # resolution
        img[:, :, 2],  # input array
    )
    kscreen = Screen(
        (0, 0),  # x and y shift
        45 * np.pi / 180,  # angle
        screens_res,  # resolution
        img[:, :, 3],  # input array
    )

    # optionally display preview of result using matplotlib
    if display_preview:
        cscreen.display(plt, colorstr="cyan")
        mscreen.display(plt, colorstr="magenta")
        yscreen.display(plt, colorstr="yellow")
        kscreen.display(plt, colorstr="black")
        plt.xlim((-img.shape[1] * 0.1, img.shape[1] * 1.1))
        plt.ylim((-img.shape[0] * 1.1, img.shape[0] * 0.1))
        plt.show()

    # convert screens to svg elements
    svg_elements = [
        c.line.svg(stroke_color="cyan")
        for c in cscreen.cells
        if not c.line.coords[0] == c.line.coords[1]
    ]
    svg_elements += [
        c.line.svg(stroke_color="magenta")
        for c in mscreen.cells
        if not c.line.coords[0] == c.line.coords[1]
    ]
    svg_elements += [
        c.line.svg(stroke_color="yellow")
        for c in yscreen.cells
        if not c.line.coords[0] == c.line.coords[1]
    ]
    svg_elements += [
        c.line.svg(stroke_color="black")
        for c in kscreen.cells
        if not c.line.coords[0] == c.line.coords[1]
    ]

    # write svg output file
    write_svg_file(
        output_file,
        svg_elements,
        img.shape[1],
        img.shape[0],
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_arguments = parser.add_argument_group("Required Arguments")
    required_arguments.add_argument("-i", "--input_file", required=True)
    required_arguments.add_argument(
        "-o", "--output_file", required=True, help="Output SVG file"
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        action="store_true",
        help="Add timestamp to output filename",
    )
    parser.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="Display preview of output file using matplotlib",
    )
    args = parser.parse_args()

    # optionally add timestamp to output filename
    if args.timestamp:
        output_file = "{}_{}{}".format(
            os.path.splitext(args.output_file)[0],
            datetime.datetime.now().strftime("%Y%m%d_%H%M"),
            os.path.splitext(args.output_file)[1],
        )
    else:
        output_file = args.output_file

    halftone(args.input_file, output_file, args.preview)
