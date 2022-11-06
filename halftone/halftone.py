"""
Script to simulate halftoning.
"""

# standard imports
import os
import datetime
import argparse
import configparser
from pathlib import Path
from xml.dom import minidom

# third party imports
import numpy as np
import skimage.io
import skimage.transform
from shapely.geometry import MultiLineString
import pandas as pd

# current project imports
from svg_utils import write_svg_file
from svg_utils import transform_multiline


class Dot:
    """Dot of a halftone screen."""

    def __init__(self, value, ulx, uly, size):
        """
        Parameters
        ----------
        value : int
        ulx : float
        uly : float
        size : float
        """

        self.value = value
        self.ulx = ulx
        self.uly = uly
        self.size = size

    def draw_glyph(self, svg_paths_dir, margins, angle, scale_factor=1.0, stroke_color=None):
        """Define SVG elements to draw glyph based on this dot's value, position and size.

        Parameters
        ----------
        svg_paths_dir : str
            directory containing glyphs paths as SVG
        margins : [float, float, float, float]
            top, right, bottom, left
        angle : float
            degrees
        scale_factor : float
            multiplication factor for the SVG stroke-width, default is 1
        stroke_color : str or None
            hex string for stroke color, default is to use "#66cc99" if geometry is valid, and
            "#ff3333" if invalid

        Returns
        -------
        string
        """

        # read glyph path svg file as multiline
        glyph_file = os.path.join(svg_paths_dir, "{:02d}.svg".format(self.value))
        doc = minidom.parse(glyph_file)
        width = float(doc.getElementsByTagName("svg")[0].getAttribute("width"))
        height = float(doc.getElementsByTagName("svg")[0].getAttribute("height"))
        lines = [
            [
                (float(point.split(",")[0]), float(point.split(",")[1]))
                for point in path.getAttribute("points").split(" ")
            ]
            for path in doc.getElementsByTagName("polyline")
        ]
        doc.unlink()
        multiline = MultiLineString(lines)

        # compute cell bounds
        target_xmin = self.ulx + margins[3]
        target_xmax = self.ulx + self.size[0] - margins[1]
        target_ymin = self.uly + margins[0]
        target_ymax = self.uly + self.size[1] - margins[2]

        # compute glyph bounds
        xmin, ymin, xmax, ymax = multiline.bounds
        target_xmin += xmin * (self.size[0] - margins[1] - margins[3]) / width
        target_xmax -= (width - xmax) * (self.size[0] - margins[1] - margins[3]) / width
        target_ymin += ymin * (self.size[1] - margins[0] - margins[2]) / height
        target_ymax -= (height - ymax) * (self.size[1] - margins[0] - margins[2]) / height

        # transform multiline based on cell position and size and angle
        multiline = transform_multiline(
            multiline, target_xmin, target_xmax, target_ymin, target_ymax, angle
        )

        return multiline.svg(scale_factor, stroke_color)


class Screen:
    """Halftone screen."""

    def __init__(self, shift, angle, resolution, input_array):
        """Constructor of halftone screen class.

        Build a halftone screen based on a given input array.

        Parameters
        ----------
        shift : (int, int)
        angle : float
        resolution : int
        input_array : np.array
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
        image_ulx2 = (image_ulx1 * np.cos(self.angle) - image_uly1 * np.sin(self.angle)) / self.res
        image_uly2 = (image_ulx1 * np.sin(self.angle) + image_uly1 * np.cos(-self.angle)) / self.res
        image_urx2 = (image_urx1 * np.cos(self.angle) - image_ury1 * np.sin(self.angle)) / self.res
        image_ury2 = (image_urx1 * np.sin(self.angle) + image_ury1 * np.cos(-self.angle)) / self.res
        image_llx2 = (image_llx1 * np.cos(self.angle) - image_lly1 * np.sin(self.angle)) / self.res
        image_lly2 = (image_llx1 * np.sin(self.angle) + image_lly1 * np.cos(-self.angle)) / self.res
        image_lrx2 = (image_lrx1 * np.cos(self.angle) - image_lry1 * np.sin(self.angle)) / self.res
        image_lry2 = (image_lrx1 * np.sin(self.angle) + image_lry1 * np.cos(-self.angle)) / self.res

        # image bounds in screen coordinates
        x2_min = int(np.floor(min([image_ulx2, image_urx2, image_llx2, image_lrx2])))
        x2_max = int(np.ceil(max([image_ulx2, image_urx2, image_llx2, image_lrx2])))
        y2_min = int(np.floor(min([image_uly2, image_ury2, image_lly2, image_lry2])))
        y2_max = int(np.ceil(max([image_uly2, image_ury2, image_lly2, image_lry2])))

        # initialize screen contents
        self.array = np.zeros((y2_max - y2_min, x2_max - x2_min))
        self.dots = []

        # loop through screen dots
        for row in range(self.array.shape[0]):
            for col in range(self.array.shape[1]):

                # dot coordinates in screen reference frame
                x2 = col + x2_min
                y2 = row + y2_min

                # dot coordinates in image reference frame
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

                # get area of image which will be covered by this dot
                xmin = int(np.floor(x1 - self.res / 2))
                xmax = int(np.ceil(x1 + self.res / 2))
                ymin = int(np.floor(y1 - self.res / 2))
                ymax = int(np.ceil(y1 + self.res / 2))

                # get mean value of area of image covered by this dot
                if (  # handle out of image areas
                    xmin < 0
                    or xmax > input_array.shape[1]
                    or ymin < 0
                    or ymax > input_array.shape[0]
                ):
                    self.array[row, col] = 0
                else:
                    self.array[row, col] = np.mean(input_array[ymin:ymax, xmin:xmax])

                # convert this screen array to integers
                self.array = np.nan_to_num(self.array).astype(np.int)

                self.dots.append(Dot(self.array[row, col], xmin, ymin, (self.res, self.res)))


def rgb_to_cmyk(img):
    """Simple formula for conversion from RGB to CMYK color model.

    Doesn't take into account true color representation on physical device.
    Third axis of input and output array is the color dimension (RGB, and CMYK respectively)

    Parameters
    ----------
    img : np.array of floats

    Returns
    -------
    output : np.array of floats
    """
    black = np.minimum.reduce([1 - img[:, :, 0], 1 - img[:, :, 1], 1 - img[:, :, 2]])
    cyan = (1 - img[:, :, 0] - black) / (1 - black)
    magenta = (1 - img[:, :, 1] - black) / (1 - black)
    yellow = (1 - img[:, :, 2] - black) / (1 - black)
    output = np.stack([cyan, magenta, yellow, black], axis=-1)
    output[np.isnan(output)] = 0
    return output


def write_info_file(input_file, output_stem, colors, image_width, image_height, config):
    """Write info file along with output file, containing command line and config parameters.

    Parameters
    ----------
    input_file : str
    output_stem : str
    colors : [str, ...]
    image_width : int or None
    image_height : int or None
    config : configparser.ConfigParser
    """

    info_file = "{}_info.txt".format(output_stem)

    with open(info_file, "w") as outfile:

        outfile.write("Halftone info file\n")

        outfile.write(
            "Processing date: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        )

        outfile.write("Command line arguments\n")
        outfile.write("Input file: {}\n".format(input_file))
        outfile.write("Output files stem: {}\n".format(output_stem))
        outfile.write("Colors: {}\n".format(", ".join(colors)))
        outfile.write("Image width (px): {}\n".format(image_width))
        outfile.write("Image height (px): {}\n".format(image_height))

        outfile.write("Config parameters\n")
        for section in config.sections():
            outfile.write("{}\n".format(section))
            for param, value in config.items(section):
                outfile.write("{}: {}\n".format(param, value))


def halftone(
    input_file,
    output_stem,
    output_colors=["c", "m", "y", "k", "a"],
    image_width=None,
    image_height=None,
    do_info_file=False,
):
    """Simulate halftoning.

    Parameters
    ----------
    input_file : str
    output_stem : str
        files path and stem, a suffix will automatically be added for example '_c.svg' for cyan
    output_colors : [str, ...]
        output files to produce, named after the colors they contain
        choices: [c]yan [m]agenta [y]ellow [b]lack [a]ll
    image_width : int or None
    image_height : int or None
    do_info_file : bool
    """

    # read config
    config = configparser.ConfigParser()
    config.read(Path(__file__).resolve().parent.parent / "config" / "config.ini")

    # read, normalize, resample input image, and convert from rgb to cmyk
    img = skimage.io.imread(input_file) / 255
    if image_width is not None and image_height is not None:
        img = skimage.transform.resize(img, (image_height, image_width))
    img = rgb_to_cmyk(img)

    # load lut and compute glyph id for each pixel
    lut_file = Path(__file__).resolve().parent.parent / config["glyphs"]["lut_file"]
    lut = pd.read_csv(lut_file)
    min = np.min(img)
    max = np.max(img)
    img = np.digitize(img, np.arange(min, max, (max - min) / len(lut))) - 1

    # create cyan, magenta, yellow and black screens
    all_colors_svg_elements = []
    colors_dicts = [
        {"id": "c", "array_coord": 0, "name": "cyan", "hex": "#2bfafa"},
        {"id": "m", "array_coord": 1, "name": "magenta", "hex": "#ff00ff"},
        {"id": "y", "array_coord": 2, "name": "yellow", "hex": "#ffff00"},
        {"id": "k", "array_coord": 3, "name": "black", "hex": "#000000"},
    ]
    for color_dict in colors_dicts:

        # initiate screen by computing its dots coordinates
        screen = Screen(
            [
                int(v) for v in config["screens_shifts"][color_dict["name"]].split(",")
            ],  # x and y shift
            float(config["screens_angles"][color_dict["name"]]) * np.pi / 180,  # angle
            int(config["screens_res"][color_dict["name"]]),  # resolution
            img[:, :, color_dict["array_coord"]],  # input array
        )

        # draw svg elements on screen using glyphs
        svg_elements = [
            d.draw_glyph(
                Path(__file__).resolve().parent.parent / config["glyphs"]["svg_paths_dir"],
                [int(v) for v in config["glyphs"]["margins"].split(",")],
                float(config["glyphs"]["angle"]),
                stroke_color=color_dict["hex"],
            )
            for d in screen.dots
            if not d.value == 0
        ]
        all_colors_svg_elements += svg_elements

        # optionally store results to output file
        if color_dict["id"] in output_colors:
            write_svg_file(
                "{}_{}.svg".format(output_stem, color_dict["id"]),
                svg_elements,
                img.shape[1],
                img.shape[0],
            )

    # optionally write output svg file containing all colors
    if "a" in output_colors:
        write_svg_file(
            "{}_a.svg".format(output_stem), all_colors_svg_elements, img.shape[1], img.shape[0]
        )

    # optionally write info file
    if do_info_file:
        write_info_file(input_file, output_stem, output_colors, image_width, image_height, config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_arguments = parser.add_argument_group("Required Arguments")
    required_arguments.add_argument("-i", "--input_file", required=True)
    required_arguments.add_argument(
        "-o",
        "--output_stem",
        required=True,
        help="Output SVG file path and stem, a suffix will be added for example '_c.svg' for cyan",
    )
    parser.add_argument(
        "-t", "--timestamp", action="store_true", help="Add timestamp to output filename"
    )
    parser.add_argument(
        "-c",
        "--colors",
        nargs="+",
        choices=["c", "m", "y", "k", "a"],
        default=["c", "m", "y", "k", "a"],
        help="Output files colors: [c]yan [m]agenta [y]ellow [b]lack [a]ll (default c m y k a)",
    )
    parser.add_argument("-iw", "--image_width", help="Width to resample image")
    parser.add_argument("-ih", "--image_height", help="Heigth to resample image")
    parser.add_argument(
        "-if", "--info_file", action="store_true", help="Option to write info file with config"
    )
    args = parser.parse_args()

    # optionally add timestamp to output filename stem
    if args.timestamp:
        output_stem = "{}_{}".format(
            args.output_stem, datetime.datetime.now().strftime("%Y%m%d_%H%M")
        )
    else:
        output_stem = args.output_stem

    halftone(
        args.input_file,
        output_stem,
        args.colors,
        int(args.image_width) if args.image_width is not None else None,
        int(args.image_height) if args.image_height is not None else None,
        args.info_file,
    )
