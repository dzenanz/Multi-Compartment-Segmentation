import os,girder_client
import numpy as np
from glob import glob
from shutil import rmtree,copy#,move,copyfile

def purge_training_set(args):
    rmtree(args.base_dir + '/' + args.project + '/Permanent/')
    rmtree(args.base_dir + '/' + args.project + '/TempHR/')
    rmtree(args.base_dir + '/' + args.project + '/TempLR/')
    initFolder(args=args)

def prune_training_set(args):
    # prune HR dataset
    regions_path = args.base_dir + '/' + args.project + '/Permanent/HR/regions/'
    masks_path = args.base_dir + '/' + args.project + '/Permanent/HR/masks/'
    prune_percent = args.prune_HR
    prune_data(regions_path, masks_path, prune_percent, args)

    # prune LR dataset
    regions_path = args.base_dir + '/' + args.project + '/Permanent/LR/regions/'
    masks_path = args.base_dir + '/' + args.project + '/Permanent/LR/masks/'
    prune_percent = args.prune_LR
    prune_data(regions_path, masks_path, prune_percent, args)

def prune_data(regions_path, masks_path, prune_percent, args):
    imgs = glob(regions_path + '*' + args.imBoxExt)
    if imgs == None:
        return

    keep = (np.random.rand(len(imgs))) >= prune_percent
    for idx,img in enumerate(imgs):
        if keep[idx] == False:
            filename = os.path.basename(img)
            mask = masks_path + '/' + os.path.splitext(filename)[0] + '.png'
            os.remove(img) # remove region
            os.remove(mask) # remove mask
            print(img)


def initFolder(args):
    dirs = {'imExt': '.jpeg'}
    dirs['basedir'] = args.base_dir
    dirs['maskExt'] = '.png'
    dirs['modeldir'] = 'MODELS'
    dirs['tempdirLR'] = 'TempLR'
    dirs['tempdirHR'] = 'TempHR'
    dirs['pretraindir'] = 'Deeplab_network'
    dirs['training_data_dir'] = 'TRAINING_data'
    dirs['validation_data_dir'] = 'HOLDOUT_data'
    dirs['model_init'] = 'deeplab_resnet.ckpt'
    dirs['project']= '/' + args.project
    dirs['data_dir_HR'] = args.base_dir + args.project + 'Permanent/HR'
    dirs['data_dir_LR'] = args.base_dir + args.project + 'Permanent/LR'
    initializeFolderStructure(dirs,args)
    print('Please add xmls/svs files to the newest TRAINING_data folder.')


def initializeFolderStructure(dirs,args):

    folder_base = args.base_dir
    base_directory_id = folder_base.split('/')[-2]
    _ = os.system("echo '\nIn the base directory: {} {}\n'".format(base_directory_id,folder_base))

    folder_project = args.project
    project_directory_id = folder_project.split('/')[-2]
    _ = os.system("echo '\nIn the base directory: {}{}\n'".format(project_directory_id,folder_base))
    
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    # modeldir = gc.createFolder(project_directory_id, dirs['modeldir'])
    

    # make_folder(dirs['basedir'] +dirs['project'] + dirs['modeldir'] + str(0) + '/LR/')
    # make_folder(dirs['basedir'] +dirs['project']+ dirs['modeldir'] + str(0) + '/HR/')
    
    if args.transfer==' ':
        pass
    else:

        modelsCurrent=os.listdir(dirs['basedir'] + '/' + args.transfer + dirs['modeldir'])
        gens=map(int,modelsCurrent)
        modelOrder=np.argsort(gens)
        modelLast=np.max(gens)
        pretrainsLR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/LR/' + 'model*')
        pretrainsHR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/HR/' + 'model*')

        maxmodel=0
        for modelfiles in pretrainsLR:
            modelID=modelfiles.split('.')[-2].split('-')[1]
            if int(modelID)>maxmodel:
                maxmodelLR=int(modelID)
        for modelfiles in pretrainsHR:
            modelID=modelfiles.split('.')[-2].split('-')[1]
            if int(modelID)>maxmodel:
                maxmodelHR=int(modelID)

        pretrain_filesLR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/LR/' + 'model.ckpt-' + str(maxmodelLR) + '*')
        pretrain_filesHR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/HR/' + 'model.ckpt-' + str(maxmodelHR) + '*')
        for file in pretrain_filesLR:
            copy(file,dirs['basedir'] +dirs['project']+ dirs['modeldir'] + str(0) + '/LR/')

        for file in pretrain_filesHR:
            copy(file,dirs['basedir'] +dirs['project']+ dirs['modeldir'] + str(0) + '/HR/')

    training_data_dir = gc.createFolder(project_directory_id, dirs['training_data_dir'])

    print(training_data_dir, 'this is training data dir')
