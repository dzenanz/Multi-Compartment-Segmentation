import json
import os
import sys

import girder_client
from lxml import etree as ET
from tqdm import tqdm
import numpy as np

from xml_to_json import convert_xml_json
from mask_to_xml import xml_add_region, xml_add_annotation
from xml_to_mask_minmax import write_minmax_to_xml

NAMES = ['cortical_interstitium', 'medullary_interstitium', 'non_globally_sclerotic_glomeruli',
         'globally_sclerotic_glomeruli', 'tubules', 'arteries/arterioles', 'intimal_fibrosis']

# do this just once, not once per file
GIRDER_API_URL = 'https://banff-aid.com/api/v1'
if "GIRDER_API_KEY" not in os.environ:
    print("Please set the GIRDER_API_KEY environment variable.")
    sys.exit(1)
gc = girder_client.GirderClient(apiUrl=GIRDER_API_URL)
girderToken = gc.authenticate(apiKey=os.environ["GIRDER_API_KEY"])


def download1(girder_item_id, xml_file):
    annotations = gc.get(path=f'/annotation/item/{girder_item_id}')
    assert len(annotations) == 1, f"Expected exactly one annotation for item {girder_item_id}, found {len(annotations)}"
    annotation = annotations[0]["annotation"]
    assert annotation["name"] == "prediction", f"Expected annotation name 'prediction', found '{annotation.name}'"
    elements = annotation['elements']

    xmlAnnot = ET.Element('Annotations')
    for i in range(len(NAMES)):
        xmlAnnot = xml_add_annotation(Annotations=xmlAnnot, annotationID=i + 1)

    for element in elements:
        pointList = []
        points = element['points']
        for point in points:
            pt_dict = {'X': point[0], 'Y': point[1]}
            pointList.append(pt_dict)
        pointList.pop()  # remove the last point, which is a duplicate of the first point

        a_name = element['group']
        if not a_name in NAMES:
            if a_name == "arteries":
                print(f"Warning: Annotation group '{a_name}' replaced by 'arteries/arterioles'.")
                a_name = 'arteries/arterioles'
            else:
                print(f"Warning: Annotation group '{a_name}' not found in predefined names.")
                a_name = None
        a_id = NAMES.index(a_name) + 1

        xmlAnnot = xml_add_region(Annotations=xmlAnnot, pointList=pointList, annotationID=a_id)

    tree = ET.ElementTree(xmlAnnot)
    # tree.write(xml_file, pretty_print=True, xml_declaration=False, encoding='utf-8')
    write_minmax_to_xml(xml_file, tree)


def main():
    if len(sys.argv) < 3:
        print("Usage: python download_xml.py <girder-folder-id> <local-directory>")
        sys.exit(1)
    if not os.path.exists(sys.argv[2]):
        print(f"Directory {sys.argv[2]} does not exist!")
        sys.exit(1)
    if not os.path.isdir(sys.argv[2]):
        print(f"{sys.argv[2]} is not a directory!")
        sys.exit(1)

    files = list(gc.listItem(sys.argv[1]))
    # dict to link filename to gc id
    item_dict = dict()
    for file in files:
        d = {file['name']: file['_id']}
        item_dict.update(d)

    for filename, id in tqdm(item_dict.items(), desc="XMLs", position=0):
        xml_file = os.path.join(sys.argv[2], filename[:-4] + ".xml")
        print(f"{xml_file}")
        download1(id, xml_file)


if __name__ == "__main__":
    main()
