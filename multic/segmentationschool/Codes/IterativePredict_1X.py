import cv2
from multic.segmentationschool.Codes.upload_assetstore_files import uploadFilesToOriginalFolder
import numpy as np
import os
import json
import sys
import lxml.etree as ET
from .xml_to_json import convert_xml_json
from tqdm import tqdm
from shutil import rmtree
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from scipy.ndimage.morphology import binary_fill_holes
from tiffslide import TiffSlide
from skimage.color import rgb2hsv
from skimage.filters import gaussian

import torch
import torch.cuda


NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']
XML_COLOR = [65280, 16776960,65535, 255, 16711680, 33023]

"""
Pipeline code to segment regions from WSI

"""


def decode_panoptic(image, segments_info, organType, args):
    """
    Optimized decode_panoptic function using GPU acceleration
    """
    
    # Create output tensor on GPU
    out = torch.zeros_like(image, device='cuda')
    
    if organType == 'liver':
        for ids in segments_info:
            if ids['isthing']:
                out[image == ids['id']] = ids['category_id'] + 1
            else:
                out[image == ids['id']] = 0
                
    elif organType == 'kidney':
        for ids in segments_info:
            if ids['isthing']:
                out[image == ids['id']] = ids['category_id'] + 3
            else:
                if args.show_interstitium:
                    if ids['category_id'] in [1, 2]:
                        out[image == ids['id']] = ids['category_id']
    else:
        print('unsupported organType ')
        print(organType)
        exit()
    
    return out

def predict(args):
    """
    Optimized predict function with GPU acceleration while maintaining original structure
    """
    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSize*(downsample))
    step = int((region_size-(args.bordercrop*2))*(1-args.overlap_percentHR))

    print('Building network configuration ...\n')
    
    #TF32 format accelarates GPU processing on A100 by 20x with a very slight hit to accuarcy. 
    # torch.backends.cuda.matmul.allow_tf32 = True  
    # torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()  
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[.1,.2,0.33, 0.5, 1.0, 2.0, 3.0,5,10]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]
    cfg.DATALOADER.NUM_WORKERS = 2  # Reduced from 10 to minimize CPU usage
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Increased for A100
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    
    if not args.Mag20X:
        cfg.INPUT.MIN_SIZE_TEST=region_size
        cfg.INPUT.MAX_SIZE_TEST=region_size
    else:
        cfg.INPUT.MIN_SIZE_TEST=int(region_size/2)
        cfg.INPUT.MAX_SIZE_TEST=int(region_size/2)
        
    cfg.MODEL.WEIGHTS = args.modelfile
    tc=['G','SG','T','A']
    sc=['Ob','C','M','B']
    classNum=len(tc)+len(sc)-1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(tc)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =len(sc)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_thresh
    
    predictor = DefaultPredictor(cfg)

    wsi = args.file
    extsplit = os.path.splitext(wsi)
    basename = extsplit[0]
    extname = extsplit[-1]
    print(basename)

    try:
        slide = TiffSlide(wsi)
    except:
        raise Exception(f"The slide cannot be read!!")

    if extname=='.scn':
        dim_y=int(slide.properties['openslide.bounds-height'])
        dim_x=int(slide.properties['openslide.bounds-width'])
        offsetx=int(slide.properties['openslide.bounds-x'])
        offsety=int(slide.properties['openslide.bounds-y'])
    else:
        dim_x, dim_y=slide.dimensions
        offsetx=0
        offsety=0

    print(dim_x,dim_y)
    
    # Process Mask on GPU
    wsiMask = torch.zeros((dim_y, dim_x), dtype=torch.uint8, device='cuda')

    index_y=range(offsety,dim_y+offsety,step)
    index_x=range(offsetx,dim_x+offsetx,step)
    
    print('Getting thumbnail mask to identify predictable tissue...')
    fullSize=slide.level_dimensions[0]
    resRatio= args.chop_thumbnail_resolution
    ds_1=fullSize[0]/resRatio
    ds_2=fullSize[1]/resRatio
    thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
    
    if extname =='.scn':
        xStt=int(offsetx/resRatio)
        xStp=int((offsetx+dim_x)/resRatio)
        yStt=int(offsety/resRatio)
        yStp=int((offsety+dim_y)/resRatio)
        thumbIm=thumbIm[yStt:yStp,xStt:xStp]

    # Move thumbnail processing to GPU #TODO (Make it compatible with multi-GPU setup)
    with torch.cuda.device(0):
        hsv = torch.from_numpy(rgb2hsv(thumbIm)).cuda()
        g = gaussian(hsv[:,:,1].cpu().numpy(), 5)  
        binary = torch.from_numpy(binary_fill_holes((g > 0.05).astype('bool'))).cuda()

    print('Segmenting tissue ...\n')
    totalpatches=len(index_x)*len(index_y)
    
    with tqdm(total=totalpatches, unit='image', colour='green', desc='Total WSI progress') as pbar:
        for i,j in coordinate_pairs(index_y,index_x):
            yEnd = min(dim_y+offsety, i+region_size)
            xEnd = min(dim_x+offsetx, j+region_size)
            yStart_small = int(np.round((i-offsety)/resRatio))
            yStop_small = int(np.round(((yEnd-offsety))/resRatio))
            xStart_small = int(np.round((j-offsetx)/resRatio))
            xStop_small = int(np.round(((xEnd-offsetx))/resRatio))
            box_total=(xStop_small-xStart_small)*(yStop_small-yStart_small)
            pbar.update(1)
            
            if torch.sum(binary[yStart_small:yStop_small,xStart_small:xStop_small]) > (args.white_percent*box_total):
                xLen=xEnd-j
                yLen=yEnd-i
                
                dxS=j
                dyS=i
                dxE=j+xLen
                dyE=i+yLen
                
                im = np.array(slide.read_region((dxS,dyS),0,(xLen,yLen)))
                
                with torch.amp.autocast(device_type='cuda'):  # Use automatic mixed precision for accelarated calculations
                    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                
                maskpart = decode_panoptic(panoptic_seg, segments_info, 'kidney', args)
                
                if dxE != dim_x:
                    maskpart[:,-int(args.bordercrop/2):]=0
                if dyE != dim_y:
                    maskpart[-int(args.bordercrop/2):,:]=0
                if dxS != offsetx:
                    maskpart[:,:int(args.bordercrop/2)]=0
                if dyS != offsety:
                    maskpart[:int(args.bordercrop/2),:]=0
                    
                dyE-=offsety
                dyS-=offsety
                dxS-=offsetx
                dxE-=offsetx
                
                wsiMask[dyS:dyE,dxS:dxE] = torch.maximum(
                    maskpart,
                    wsiMask[dyS:dyE,dxS:dxE]
                )

    slide.close()
    print('\n\nStarting XML construction: ')
    
    if extname=='.scn':
        xml_suey(wsiMask=wsiMask, args=args, classNum=classNum, downsample=downsample,glob_offset=[offsetx,offsety])
    else:
        xml_suey(wsiMask=wsiMask, args=args, classNum=classNum, downsample=downsample,glob_offset=[0,0])

    torch.cuda.empty_cache()

def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def get_tiled_contour_points(mask, args, downsample, value, offset={'X': 0, 'Y': 0}, tile_size=5000):
    """
    Memory-efficient contour detection using tiling. This is to handle everything on GPU so that we can reduce system memory.
    mask: GPU tensor
    """
    height, width = mask.shape
    pointsList = []
    min_size = args.min_size[value-1]
    
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            tile = mask[y:y_end, x:x_end].cpu().numpy()
            
            if not np.any(tile):
                continue
            
            maskPoints, _ = cv2.findContours(
                tile, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_TC89_KCOS
            )
            
            for contour in maskPoints:
                if len(contour) > 2:
                    # Adjust contour coordinates to global space since this is chunked. 
                    contour[:, :, 0] += x
                    contour[:, :, 1] += y
                    
                    area = cv2.contourArea(contour)
                    if area > min_size:
                        points = contour[::4, 0]
                        pointList = [
                            {
                                'X': int((px * downsample) + offset['X']),
                                'Y': int((py * downsample) + offset['Y'])
                            }
                            for px, py in points
                        ]
                        pointsList.append(pointList)
            
            del tile
    
    return pointsList



def xml_suey(wsiMask, args, classNum, downsample, glob_offset):
    """
    GPU-optimized XML generation
    wsiMask is expected to be a torch tensor on GPU
    """
    
    Annotations = xml_create()
    for i in range(classNum)[1:]:
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i)
    
    for value in range(1, 7):
        print(f'\t Working on: annotationID {value}')
    
        if not torch.any(wsiMask == value):
            continue
        
        binary_mask = (wsiMask == value).to(torch.uint8)
        
        pointsList = get_tiled_contour_points(
            binary_mask, 
            args=args, 
            downsample=downsample,
            value=value,
            offset={'X': glob_offset[0], 'Y': glob_offset[1]},
            tile_size=5000  # Adjust based on RAM set on Node
        )
        
        for pointList in pointsList:
            Annotations = xml_add_region(
                Annotations=Annotations, 
                pointList=pointList, 
                annotationID=value
            )
        
        del binary_mask
        torch.cuda.empty_cache()
    
    gc = args.gc
    annots = convert_xml_json(Annotations, NAMES)
    output_files = []
    output_dir = '/tmp'
    
    print('uploading layers')
    for annot in annots:
        _ = gc.post(
            path='annotation',
            parameters={'itemId': args.item_id}, 
            data=json.dumps(annot)
        )
        
        output_filename = annot['name'].replace("/", "_") + '.json'
        file_path = os.path.join(output_dir, output_filename)
        with open(file_path, 'w') as f:
            json.dump(annot, f)
        print('output file: ', file_path)
        output_files.append(file_path)
    
    with open(os.path.join(output_dir, 'annotations.xml'), 'w') as f:
        f.write(ET.tostring(Annotations, pretty_print=True).decode('utf-8'))
    output_files.append(os.path.join(output_dir, 'annotations.xml'))
    
    print('output files: ', output_files)
    uploadFilesToOriginalFolder(
        gc, 
        output_files, 
        args.item_id, 
        'MultiCompartment_Segmentation', 
        args.girderApiUrl, 
        True
    )
    print('annotation uploaded...\n')

def get_contour_points(mask, args, downsample, value, offset={'X': 0, 'Y': 0}):
    """
    Optimized contour point extraction
    mask is expected to be a numpy array (CV2 requirement)
    """
    maskPoints, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_TC89_KCOS
    )
    
    pointsList = []
    min_size = args.min_size[value-1]
    
    for contour in maskPoints:
        if len(contour) > 2 and cv2.contourArea(contour) > min_size:
            points = contour[::4, 0]
            pointList = [
                {
                    'X': int((x * downsample) + offset['X']),
                    'Y': int((y * downsample) + offset['Y'])
                }
                for x, y in points
            ]
            pointsList.append(pointList)
    
    return pointsList

def xml_create():
    return ET.Element('Annotations')

def xml_add_annotation(Annotations, annotationID=None):
    if annotationID is None:
        annotationID = len(Annotations.findall('Annotation')) + 1
    
    visible = '0' if annotationID in [1, 2] else '1'
    
    Annotation = ET.SubElement(
        Annotations,
        'Annotation',
        attrib={
            'Type': '4',
            'Visible': visible,
            'ReadOnly': '0',
            'Incremental': '0',
            'LineColorReadOnly': '0',
            'LineColor': str(XML_COLOR[annotationID-1]),
            'Id': str(annotationID),
            'NameReadOnly': '0'
        }
    )
    ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID=-1, regionID=None):
    Annotation = Annotations.find(f"Annotation[@Id='{str(annotationID)}']")
    Regions = Annotation.find('Regions')
    
    if regionID is None:
        regionID = len(Regions.findall('Region')) + 1
    
    Region = ET.SubElement(
        Regions,
        'Region',
        attrib={
            'NegativeROA': '0',
            'ImageFocus': '-1',
            'DisplayId': '1',
            'InputRegionId': '0',
            'Analyze': '0',
            'Type': '0',
            'Id': str(regionID)
        }
    )
    
    Vertices = ET.SubElement(Region, 'Vertices')
    
    for point in pointList:
        ET.SubElement(
            Vertices,
            'Vertex',
            attrib={
                'X': str(point['X']),
                'Y': str(point['Y']),
                'Z': '0'
            }
        )
    
    ET.SubElement(
        Vertices,
        'Vertex',
        attrib={
            'X': str(pointList[0]['X']),
            'Y': str(pointList[0]['Y']),
            'Z': '0'
        }
    )
    
    return Annotations
