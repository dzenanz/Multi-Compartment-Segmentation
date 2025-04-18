import argparse
import os
import sys

import numpy as np
from lxml import etree as ET
from tqdm import tqdm

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()  # default is 2^30
import cv2
from mask_to_xml import mask_to_xml
from xml_to_mask_minmax import write_minmax_to_xml

# OpenCV uses BGR format, not RGB
TUB_NORMAL = np.array([50, 205, 50], dtype=np.uint8)
GLO_NORMAL = np.array([225, 105, 65], dtype=np.uint8)
GLO_ABNORMAL = np.array([255, 0, 255], dtype=np.uint8)
ARTERY = np.array([35, 142, 107], dtype=np.uint8)
ARTERY_IF = np.array([0, 255, 255], dtype=np.uint8)
INTERSTITIAL_SPACE_ABNORMALITY = np.array([0, 0, 255], dtype=np.uint8)
MISSING_STRUCTURE = np.array([0, 0, 0], dtype=np.uint8)  # background is white

COLORS = [INTERSTITIAL_SPACE_ABNORMALITY, MISSING_STRUCTURE, GLO_NORMAL, GLO_ABNORMAL, TUB_NORMAL, ARTERY, ARTERY_IF]

parser = argparse.ArgumentParser()
parser.add_argument('--min_size', dest='min_size', default=[30]*8, type=int,
                    help='min size region to be considered after prepass [in pixels]')
args = parser.parse_args("")


def convert1(in_mask, xml_file):
    intermediate_filename = xml_file[:-4] + "-label.png"
    if not os.path.isfile(intermediate_filename):
        image = cv2.imread(in_mask)
        assert image.shape[2] == 3, "Image must be in color format"
        int_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i, color in enumerate(tqdm(COLORS, desc="mask labels", position=0)):
            int_mask = np.where((image == color).all(axis=2), i + 1, int_mask)
        cv2.imwrite(xml_file[:-4] + "-label.png", int_mask)
        del image  # free memory before proceeding
    else:
        int_mask = cv2.imread(intermediate_filename, cv2.IMREAD_UNCHANGED)
        assert len(int_mask.shape) == 2, "Intermediate mask must be in grayscale format"

    annotations = mask_to_xml(wsiMask=int_mask,
                              args=args,
                              classNum=8,
                              downsample=1.0,
                              glob_offset=(0, 0),
                              )
    tree = ET.ElementTree(annotations)
    # tree.write(xml_file, pretty_print=True, xml_declaration=False, encoding='utf-8')
    write_minmax_to_xml(xml_file, tree)


def convert_all(in_directory):
    """
    Convert all the files in the given directory to XML format.
    """
    file_list = []
    for filename in os.listdir(in_directory):
        if filename.endswith("_original_size_new_mask_img.png"):
            file_list.append(filename)

    for i, filename in enumerate(tqdm(file_list, desc="Files")):
        in_mask = os.path.join(in_directory, filename)
        xml_file = os.path.join(in_directory, filename[:-31] + ".xml")
        print(f"{in_mask} -> {xml_file}")
        convert1(in_mask, xml_file)


def main():
    if len(sys.argv) < 2:
        convert_all(os.getcwd())
    elif len(sys.argv) == 2:
        convert_all(sys.argv[1])
    else:
        convert1(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
