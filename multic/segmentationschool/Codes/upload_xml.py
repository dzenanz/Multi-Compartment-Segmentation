import json
import os
import sys

import girder_client
from lxml import etree as ET
from tqdm import tqdm

from xml_to_json import convert_xml_json

NAMES = ['cortical_interstitium', 'medullary_interstitium', 'non_globally_sclerotic_glomeruli',
         'globally_sclerotic_glomeruli', 'tubules', 'arteries/arterioles', 'intimal_fibrosis']

# do this just once, not once per file
GIRDER_API_URL = 'https://banff-aid.com/api/v1'
if "GIRDER_API_KEY" not in os.environ:
    print("Please set the GIRDER_API_KEY environment variable.")
    sys.exit(1)
gc = girder_client.GirderClient(apiUrl=GIRDER_API_URL)
girderToken = gc.authenticate(apiKey=os.environ["GIRDER_API_KEY"])
files = list(gc.listItem('6818fa1e08cc9f7f924bb982'))  # "WSIs/To Review"
# dict to link filename to gc id
item_dict = dict()
for file in files:
    d = {file['name']: file['_id']}
    item_dict.update(d)


def upload1(image_filename, xml_file):
    if image_filename not in item_dict:
        print(f"Image {image_filename} not found in 'WSIs/To Review' folder.")
        return
    annotations_xml = ET.parse(xml_file)
    annotations_json = convert_xml_json(annotations_xml, NAMES)
    retval = gc.post(path='annotation', parameters={'itemId': item_dict[image_filename]},
                     data=json.dumps(annotations_json[0]))


def upload_all(in_directory):
    """
    Upload all the files in the given directory to Girder.
    """
    file_list = []
    for filename in os.listdir(in_directory):
        if filename.endswith(".xml"):
            file_list.append(filename)

    for filename in tqdm(file_list, desc="XMLs", position=0):
        xml_file = os.path.join(in_directory, filename)
        image_file = filename[:-4] + ".svs"
        print(f"{xml_file}")
        upload1(image_file, xml_file)


def main():
    if len(sys.argv) < 2:
        upload_all(os.getcwd())
    elif len(sys.argv) == 2:
        upload_all(sys.argv[1])
    else:
        upload1(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
