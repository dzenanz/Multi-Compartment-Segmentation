import argparse
import os

import numpy as np
from lxml import etree as ET
from tiffslide import TiffSlide
from tqdm import tqdm

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()  # default is 2^30
import cv2
from mask_to_xml import mask_to_xml
from xml_to_mask_minmax import write_minmax_to_xml

INPUT_DIR = "M:/Histo/MSSM/annotated_xml/"
OUTPUT_DIR = "M:/Histo/MSSM/AperioXML/"

# to pass as argument to mask_to_xml
parser = argparse.ArgumentParser()
parser.add_argument('--min_size', dest='min_size', default=[30] * 8, type=int,
                    help='min size region to be considered after prepass [in pixels]')
args = parser.parse_args("")


def draw1(mask, xml, structure, label):
    """
    Draws the mask for a single structure on the mask image.
    """
    regions = xml.findall(f"./Annotations/Annotation[@PartOfGroup='{structure}']/Coordinates")
    for region in regions:
        points = [(int(round(float(v.get('X')))), int(round(float(v.get('Y'))))) for v in
                  region.findall('Coordinate')]
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], color=label)


def main():
    dir_list = os.listdir(INPUT_DIR)
    file_list = []
    file_dir_dict = {}
    os.chdir(INPUT_DIR)
    for d in dir_list:
        for filename in os.listdir(d):
            if filename.endswith(".svs"):
                if filename not in file_list:
                    file_list.append(filename)
                    file_dir_dict[filename] = [d]
                else:
                    file_dir_dict[filename].append(d)


    for i, filename in enumerate(tqdm(file_list, desc="Files")):
        xml_file = os.path.join(OUTPUT_DIR, filename[:-4] + ".xml")
        if len(file_dir_dict[filename]) > 1:
            print(f"Combining {file_dir_dict[filename]} -> {xml_file}")
        else:
            print(f"Converting {file_dir_dict[filename]} -> {xml_file}")
        intermediate_filename = xml_file[:-4] + "-label.png"

        if not os.path.isfile(intermediate_filename):
            in_xmls = []
            for d in file_dir_dict[filename]:
                in_xml_path = os.path.join(INPUT_DIR, d, filename[:-4] + ".xml")
                assert os.path.isfile(in_xml_path)
                in_xml = ET.parse(in_xml_path)
                in_xmls.append(in_xml)

            slide = TiffSlide(os.path.join(INPUT_DIR, file_dir_dict[filename][0], filename))
            dim_x, dim_y = slide.dimensions

            int_mask = np.zeros((dim_y, dim_x), dtype=np.uint8)
            structures = ["Missing", "Missing", "Glo_Normal", "Glo_Abnormal", "Tub_Normal", "Artery", "Artery_IF"]
            for i, structure in enumerate(structures):
                for in_xml in in_xmls:
                    draw1(int_mask, in_xml, structure, i + 1)  # draw the first xml
            cv2.imwrite(intermediate_filename, int_mask)
        else:
            int_mask = cv2.imread(intermediate_filename, cv2.IMREAD_UNCHANGED)
            assert len(int_mask.shape) == 2, "Intermediate mask must be in grayscale format"

        # the rst is the same as in color_mask_to_xml.py
        annotations = mask_to_xml(wsiMask=int_mask,
                                  args=args,
                                  classNum=8,
                                  downsample=1.0,
                                  glob_offset=(0, 0),
                                  )
        tree = ET.ElementTree(annotations)
        # tree.write(xml_file, pretty_print=True, xml_declaration=False, encoding='utf-8')
        write_minmax_to_xml(xml_file, tree)


if __name__ == "__main__":
    main()
