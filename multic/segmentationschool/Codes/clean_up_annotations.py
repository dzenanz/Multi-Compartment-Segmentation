import json
import os
import sys
import time

from lxml import etree as ET
from tqdm import tqdm

from xml_to_mask_minmax import write_minmax_to_xml


def clean1(xml_file):
    xml = ET.parse(xml_file)
    regions = xml.findall(f"//Vertices")
    updated = False
    for vertices in regions:
        if len(vertices.findall('Vertex')) < 4:
            print(f"Removing region with less than 4 vertices:\n{ET.tostring(vertices, encoding='unicode')}")
            region = vertices.getparent()
            region.getparent().remove(region)
            updated = True

    if updated:
        xml.getroot().set("modtime", "{}".format(time.time()))
        xml.write(xml_file, pretty_print=True, xml_declaration=False, encoding='utf-8')
        # write_minmax_to_xml(xml_file, xml)

def main():
    """Clean up all the annotation files in the given directory."""
    in_directory = sys.argv[1]
    file_list = []
    for filename in os.listdir(in_directory):
        if filename.endswith(".xml"):
            file_list.append(filename)

    for filename in tqdm(file_list, desc="XMLs", position=0):
        xml_file = os.path.join(in_directory, filename)
        print(f"{xml_file}")
        clean1(xml_file)


if __name__ == "__main__":
    main()
