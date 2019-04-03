# encoding: utf-8

"""
The main CheXNet model implementation.
"""
import re
import sys
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from skimage.measure import label
from model import Densenet121_AG, Fusion_Branch
from PIL import Image

#np.set_printoptions(threshold = np.nan)


CKPT_PATH = ''

CKPT_PATH_G = '/best_model/AG_CNN_Global_epoch_1.pkl' 
CKPT_PATH_L = '/best_model/AG_CNN_Local_epoch_2.pkl' 
CKPT_PATH_F = '/best_model/AG_CNN_Fusion_epoch_23.pkl'

N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

DATA_DIR = '/path/to/ur/data'
TRAIN_IMAGE_LIST = '/labels/train_list.txt'
TEST_IMAGE_LIST = '/labels/test_list.txt'

num_epochs = 50
BATCH_SIZE = 32

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize,
])


def Attention_gen_patchs(ori_image, fm_cuda):
    # fm => mask =>(+ ori-img) => crop = patchs
    feature_conv = fm_cuda.data.cpu().numpy()
    size_upsample = (224, 224) 
    bz, nc, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().cuda()

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h,w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        
        # to ori image 
        image = ori_image[i].numpy().reshape(224,224,3)
        image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh,minw:maxw,:] * 256 # because image was normalized before
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3,224,224).unsqueeze(0).cuda())

        patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    return patchs_cuda


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 


def main():
    print('********************load data********************')
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=128,
                             shuffle=False, num_workers=4, pin_memory=True)
    print('********************load data succeed!********************')


    print('********************load model********************')
    # initialize and load the model
    Global_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Local_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()

    if os.path.isfile(CKPT_PATH_G):
        checkpoint = torch.load(CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_L):
        checkpoint = torch.load(CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_F):
        checkpoint = torch.load(CKPT_PATH_F)
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")

    cudnn.benchmark = True
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    test(Global_Branch_model, Local_Branch_model, Fusion_Branch_model,test_loader)

def test(model_global, model_local, model_fusion, test_loader):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_global = torch.FloatTensor().cuda()
    pred_local = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_global.eval()
    model_local.eval()
    model_fusion.eval()
    cudnn.benchmark = True

    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            if i % 2000 == 0:
                print('testing process:',i)
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())
            #output = model_global(input_var)

            output_global, fm_global, pool_global = model_global(input_var)
            
            patchs_var = Attention_gen_patchs(inp,fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global,pool_local)

            pred_global = torch.cat((pred_global, output_global.data), 0)
            pred_local = torch.cat((pred_local, output_local.data), 0)
            pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
            
    AUROCs_g = compute_AUCs(gt, pred_global)
    AUROC_avg = np.array(AUROCs_g).mean()
    print('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

    AUROCs_l = compute_AUCs(gt, pred_local)
    AUROC_avg = np.array(AUROCs_l).mean()
    print('\n')
    print('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

    AUROCs_f = compute_AUCs(gt, pred_fusion)
    AUROC_avg = np.array(AUROCs_f).mean()
    print('\n')
    print('Fusion branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_f[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

if __name__ == '__main__':
    main()