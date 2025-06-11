import cv2
import numpy as np
import os
import json
import sys
import girder_client
import glob

from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from lxml import etree as ET
from scipy.ndimage.morphology import binary_fill_holes
from tiffslide import TiffSlide
from skimage.color import rgb2hsv
from skimage.filters import gaussian

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xml_to_json import convert_xml_json
from get_dataset_list import decode_panoptic
from mask_to_xml import mask_to_xml
from xml_to_mask_minmax import write_minmax_to_xml


NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']

"""
Pipeline code to segment regions from WSI

"""


def decode_panoptic(image,segments_info,organType,args):

    detections=np.unique(image)
    detections=detections[detections>-1]

    out=np.zeros_like(image)
    if organType=='liver':
        for ids in segments_info:
            if ids['isthing']:
                out[image==ids['id']]=ids['category_id']+1

            else:
                out[image==ids['id']]=0

    elif organType=='kidney':
        for ids in segments_info:
            if ids['isthing']:
                out[image==ids['id']]=ids['category_id']+3

            else:
                if args.show_interstitium:
                    if ids['category_id'] in [1,2]:
                        out[image==ids['id']]=ids['category_id']



    else:
        print('unsupported organType ')
        print(organType)
        exit()

    return out.astype('uint8')



def predict(args):
    dirs = {'outDir': args.base_dir}
    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSize*(downsample))
    step = int((region_size-(args.bordercrop*2))*(1-args.overlap_percentHR))
    # print('Handcoded iteration')
    iteration=1
    # print(iteration)
    dirs['xml_save_dir'] = args.base_dir
    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:

        print('Building network configuration ...\n')

        os.environ["CUDA_VISIBLE_DEVICES"]="0"

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
        cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']
        # cfg.MODEL.PIXEL_MEAN=[189.409,160.487,193.422]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[.1,.2,0.33, 0.5, 1.0, 2.0, 3.0,5,10]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]
        cfg.DATALOADER.NUM_WORKERS = 10
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
        if not args.Mag20X:
            cfg.INPUT.MIN_SIZE_TEST=region_size
            cfg.INPUT.MAX_SIZE_TEST=region_size
        else:
            cfg.INPUT.MIN_SIZE_TEST=int(region_size/2)
            cfg.INPUT.MAX_SIZE_TEST=int(region_size/2)
        cfg.MODEL.WEIGHTS = args.modelfile

        tc=['G','SG','T','A', 'IF']
        sc=['Ob','C','M','B']
        classNum=len(tc)+len(sc)-1
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(tc)
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =len(sc)

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_thresh
        predictor = DefaultPredictor(cfg)
        broken_slides=[]
        for f, wsi in enumerate(args.files):
            extsplit = os.path.splitext(wsi)
            basename = extsplit[0]
            extname = extsplit[-1]
            print(wsi)

            try:
                slide=TiffSlide(wsi)
            except Exception as e:
                print(f"Error opening slide {wsi}: {e}")
                broken_slides.append(wsi)
                continue

            if extname=='.scn':
                dim_y=int(slide.properties['openslide.bounds-height'])
                dim_x=int(slide.properties['openslide.bounds-width'])
                offsetx=int(slide.properties['openslide.bounds-x'])
                offsety=int(slide.properties['openslide.bounds-y'])
            else:
                dim_x, dim_y=slide.dimensions
                offsetx=0
                offsety=0

            print(f"Size: {dim_x} x {dim_y}")
            fileID=basename.split(os.sep)
            dirs['fileID'] = fileID[-1]
            dirs['extension'] = extname
            dirs['file_name'] = wsi.split(os.sep)[-1]


            wsiMask = np.zeros([dim_y, dim_x], dtype='uint8')

            index_y=np.array(range(offsety,dim_y+offsety,step))
            index_x=np.array(range(offsetx,dim_x+offsetx,step))
            # print('Getting thumbnail mask to identify predictable tissue...')
            fullSize=slide.level_dimensions[0]
            resRatio= args.chop_thumbnail_resolution
            ds_1=fullSize[0]/resRatio
            ds_2=fullSize[1]/resRatio
            try:
                thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
            except Exception as e:
                print(f"Error opening thumbnail of slide {wsi}: {e}")
                broken_slides.append(wsi)
                continue
            if extname =='.scn':
                xStt=int(offsetx/resRatio)
                xStp=int((offsetx+dim_x)/resRatio)
                yStt=int(offsety/resRatio)
                yStp=int((offsety+dim_y)/resRatio)
                thumbIm=thumbIm[yStt:yStp,xStt:xStp]

            hsv=rgb2hsv(thumbIm)
            g=gaussian(hsv[:,:,1],5)
            binary=(g>0.05).astype('bool')
            binary=binary_fill_holes(binary)

            # print('Segmenting tissue ...\n')
            totalpatches=len(index_x)*len(index_y)
            with tqdm(total=totalpatches,unit='image',colour='green',desc=f"WSI {f + 1}/{len(args.files)}") as pbar:
                for i,j in coordinate_pairs(index_y,index_x):

                    yEnd = min(dim_y+offsety,i+region_size)
                    xEnd = min(dim_x+offsetx,j+region_size)
                    yStart_small = int(np.round((i-offsety)/resRatio))
                    yStop_small = int(np.round(((yEnd-offsety))/resRatio))
                    xStart_small = int(np.round((j-offsetx)/resRatio))
                    xStop_small = int(np.round(((xEnd-offsetx))/resRatio))
                    box_total=(xStop_small-xStart_small)*(yStop_small-yStart_small)
                    pbar.update(1)
                    if np.sum(binary[yStart_small:yStop_small,xStart_small:xStop_small])>(args.white_percent*box_total):

                        xLen=xEnd-j
                        yLen=yEnd-i

                        dxS=j
                        dyS=i
                        dxE=j+xLen
                        dyE=i+yLen
                        im=np.array(slide.read_region((dxS,dyS),0,(xLen,yLen)))[:,:,:3]

                        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                        maskpart=decode_panoptic(panoptic_seg.to("cpu").numpy(),segments_info,'kidney',args)
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

                        wsiMask[dyS:dyE,dxS:dxE]=np.maximum(maskpart,
                            wsiMask[dyS:dyE,dxS:dxE])


            slide.close()

            if args.base_dir and os.path.exists(args.base_dir):
                mask_filename = args.base_dir + "/" + dirs['fileID'] + ".png"
                print(f"Writing mask to file: {mask_filename}")
                cv2.imwrite(mask_filename, wsiMask)

            # print('\n\nStarting XML construction: ')
            if extname=='.scn':
                # print('here writing 1')
                xml_suey(wsiMask=wsiMask, dirs=dirs, args=args, classNum=classNum, downsample=downsample,glob_offset=[offsetx,offsety])
            else:
                # print('here writing 2')
                xml_suey(wsiMask=wsiMask, dirs=dirs, args=args, classNum=classNum, downsample=downsample,glob_offset=[0,0])

        print('\nand run [--option train]\033[0m\n')
        print('The following slides were not openable by openslide:')
        print(broken_slides)




def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j
def get_iteration(args):
    currentmodels=os.listdir(args.base_dir)
    if not currentmodels:
        return 'none'
    else:
        currentmodels=list(map(int,currentmodels))
        Iteration=np.max(currentmodels)
        return Iteration

def get_test_model(modeldir):
    pretrains=glob.glob(modeldir + '/*.pth')
    if os.path.isfile(modeldir+'/model_final2.pth'):
        return modeldir+'/model_final.pth'
    else:
        maxmodel=0
        for modelfiles in pretrains:
            modelID=modelfiles.split('.')[0].split('_')[-1]
            print(modelID)
            try:
                modelIDi = int(modelID)
                if modelIDi>maxmodel:
                    maxmodel=modelID
            except: pass
        return ''.join([modeldir,'/model_',maxmodel,'.pth'])

def make_folder(directory):
    print(directory,'predict dir')
    #if not os.path.exists(directory):
    try:
        os.makedirs(directory) # make directory if it does not exit already # make new directory
    except:
        print('folder exists!')

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def getWsi(path): #imports a WSI
    import openslide
    slide = openslide.TiffSlide(path)
    return slide

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    if 'i' in locals():
        return i + 1

    else:
        return 0


def xml_suey(wsiMask, dirs, args, classNum, downsample, glob_offset):
    Annotations = mask_to_xml(wsiMask, args, classNum, downsample, glob_offset)

    # save xml
    folder = args.base_dir
    girder_folder_id = folder.split(os.sep)[-2]
    # _ = os.system("echo 'Using data from girder_client Folder: {}\n'".format(folder))
    file_name = dirs['file_name']
    # print(file_name)
    tree = ET.ElementTree(Annotations)
    xml_file = dirs['xml_save_dir'] + "/" + dirs['fileID'] + ".xml"
    # tree.write(xml_file, pretty_print=True, xml_declaration=False, encoding='utf-8')
    write_minmax_to_xml(xml_file, tree)
    return  # skip upload to girder for now

    # upload to girder
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    files = list(gc.listItem(girder_folder_id))
    # dict to link filename to gc id
    item_dict = dict()
    for file in files:
        d = {file['name']:file['_id']}
        item_dict.update(d)
    print(item_dict)
    print(item_dict[file_name])
    annots = convert_xml_json(Annotations, NAMES)
    for annot in annots:
        _ = gc.post(path='annotation',parameters={'itemId':item_dict[file_name]}, data = json.dumps(annot))
        print('uploading layers')
    print('annotation uploaded...\n')
