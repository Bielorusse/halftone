"""
SVG utils module for Halftone.
"""

# standard imports
import os
import argparse
from xml.dom import minidom

# third party imports
import svg.path
from shapely.geometry import MultiLineString
import numpy as np


def translate(value, input_min, input_max, output_min, output_max):
    """Mapping a range of values to another"""

    # handle case of single value in left range
    if input_max == input_min:
        return (output_min + output_max) / 2

    else:

        # figure out how 'wide' each range is
        input_span = input_max - input_min
        output_span = output_max - output_min

        # convert the left range into a 0-1 range (float)
        value_scaled = (value - input_min) / input_span

        # convert the 0-1 range into a value in the right range.
        return output_min + (value_scaled * output_span)


def transform_multiline(input_multiline, target_xmin, target_xmax, target_ymin, target_ymax, angle):
    """Move and resize multiline to a given bounding box.

    Parameters
    ----------
    input_multiline : shapely.geometry.MultiLineString
    target_xmin : float
    target_xmax : float
    target_ymin : float
    target_ymax : float
    angle : float

    Returns
    -------
    shapely.geometry.MultiLineString
    """

    # read multiline bounds and initiate output multiline
    input_xmin, input_ymin, input_xmax, input_ymax = input_multiline.bounds
    output_multiline = []

    # compute rotation matrix and cell center if angle is given
    if angle != 0:
        angle = angle * np.pi / 180.0
        rotation_matrix = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        cell_center = (
            target_xmin + (target_xmax - target_xmin) / 2,
            target_ymin + (target_ymax - target_ymin) / 2,
        )

    for line in input_multiline:

        # read and translate coordinates
        coords = np.asarray(list(line.coords))
        coords[:, 0] = translate(coords[:, 0], input_xmin, input_xmax, target_xmin, target_xmax)
        coords[:, 1] = translate(coords[:, 1], input_ymin, input_ymax, target_ymin, target_ymax)

        # apply rotation if angle is given
        if angle != 0:
            coords = np.matmul(coords - cell_center, rotation_matrix) + cell_center

        output_multiline.append(coords)

    return MultiLineString(output_multiline)


def write_svg_file(
    output_file, svg_elements, canvas_width, canvas_height, title=None, description=None
):
    """Write SVG file.

    Parameters
    ----------
    output_file : str
    svg_elements : [str, ...]
        list of strings defining svg elements
    canvas_width : int
    canvas_height : int
    title : str
    description : str
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


def path_to_lines(path_str):
    """
    Read SVG path string using svg.path and convert it to a shapely MultiLineString object.

    Parameters
    ----------
    path_str : str

    Returns
    -------
    shapely.geometry.MultiLineString
    """

    # parse svg path
    path = svg.path.parse_path(path_str)

    # initiate lists and coords
    lines = []
    points = []
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0

    for element in path:
        if isinstance(element, svg.path.Move) and not (x1 == 0 and y1 == 0):
            # element is a move (pen up): close line with previous element end point
            points.append((x1, y1))
            lines.append(points)
            points = []
        elif isinstance(element, svg.path.Line):
            # element is a line (pen down): add element start point
            x0 = element.start.real
            y0 = element.start.imag
            x1 = element.end.real
            y1 = element.end.imag
            points.append((x0, y0))
    if points != []:
        # close line with previous element end point
        points.append((x1, y1))
        lines.append(points)
        points = []

    return MultiLineString(lines)


def crop_svg_paths(input_file, output_dir, width, height, margins):
    """
    Read input SVG file and write single paths in separate SVG output files.

    Parameters
    ----------
    input_file : str
    output_dir : str
    width : float
    height : float
    margins : [float, float, float, float]
        top, right, bottom, left
    """

    # create output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read and parse paths of input svg file
    doc = minidom.parse(input_file)
    path_strings = [path.getAttribute("d") for path in doc.getElementsByTagName("path")]
    doc.unlink()

    for i, path_str in enumerate(path_strings):

        # read path strings and convert to shapely lines
        multiline = path_to_lines(path_str)

        # resize coords
        multiline = transform_multiline(
            multiline,
            0.0 + margins[3],
            width - margins[1],
            0.0 + margins[0],
            height - margins[2],
            0,
        )

        # write single path to separate svg file
        write_svg_file(
            os.path.join(output_dir, "{:02d}.svg".format(i + 1)), [multiline.svg()], width, height
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_arguments = parser.add_argument_group("required arguments")
    required_arguments.add_argument("-i", "--input_file", required=True)
    required_arguments.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-w", "--width", help="output svg files width (default=30)", default="30")
    parser.add_argument(
        "-he", "--height", help="output svg files height (default=30)", default="30"
    )
    parser.add_argument(
        "-m",
        "--margins",
        nargs=4,
        help="output svg files margins (top, right, bottom, left, default=0)",
        default=["0", "0", "0", "0"],
    )
    args = parser.parse_args()

    crop_svg_paths(
        args.input_file,
        args.output_dir,
        float(args.width),
        float(args.height),
        [float(m) for m in args.margins],
    )
