import os
import argparse
import torch
import numpy as np

import logging
import warnings

from osgeo import gdal

import math

from torchvision.transforms import Normalize, Compose, ToTensor

def get_model():
    models = args.models
    print(models)
    assert models in ['danet', 'bisenetv2', 'pspnet',
                      'deeplabv3', 'fcn', 'fpn', 'unet', 'unet_vgg', 'unet_resnet']
    if models == 'danet':
        from models.danet import DANet
        model = DANet(nclass=args.nclass, backbone='resnet50', pretrained_base=False)
    if models == 'bisenetv2':
        from models.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(nclass=args.nclass)
    if models == 'pspnet':
        from models.pspnet import PSPNet
        model = PSPNet(nclass=args.nclass, backbone='resnet50', pretrained_base=False)
    if models == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(nclass=args.nclass, backbone='resnet50', pretrained_base=False)
    if models == 'fcn':
        from models.fcn import FCN16s
        model = FCN16s(nclass=args.nclass)
    if models == 'fpn':
        from models.fpn import FPN
        model = FPN(nclass=args.nclass)
    if models == 'unet':
        from models.unet import UNet
        model = UNet(nclass=args.nclass)
    if models == 'unet_vgg':
        from models.unet_vgg import Unet_vgg
        model = Unet_vgg(nclass=args.nclass, backbone='vgg16')
    if models == 'unet_resnet':
        from models.unet_resnet import Unet_resnet
        model = Unet_resnet(nclass=args.nclass, backbone='resnet50', pretrained_base=args.pretrained)

    model = model.to(device)
    return model

def Result(shape, TifArray, npyfile, num_class, RepetitiveLength, RowOver, ColumnOver):
    #  获得结果矩阵
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i, img in enumerate(npyfile):
        img = img.astype(np.uint8)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 640 - RepetitiveLength, 0 : 640-RepetitiveLength] = img[0 : 640 - RepetitiveLength, 0 : 640 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 640 - RepetitiveLength] = img[640 - ColumnOver - RepetitiveLength : 640, 0 : 640 - RepetitiveLength]
            else:
                result[j * (640 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (640 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:640-RepetitiveLength] = img[RepetitiveLength : 640 - RepetitiveLength, 0 : 640 - RepetitiveLength]
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 640 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 640 - RepetitiveLength, 640 -  RowOver: 640]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[640 - ColumnOver : 640, 640 - RowOver : 640]
            else:
                result[j * (640 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (640 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 640 - RepetitiveLength, 640 - RowOver : 640]
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 640 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (640 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (640 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 640 - RepetitiveLength, RepetitiveLength : 640 - RepetitiveLength]
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (640 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (640 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[640 - ColumnOver : 640, RepetitiveLength : 640 - RepetitiveLength]
            else:
                result[j * (640 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (640 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (640 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (640 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 640 - RepetitiveLength, RepetitiveLength : 640 - RepetitiveLength]
    return result

def TifCroppingArray(img, SideLength):
    #  tif裁剪（tif像素数据，裁剪边长）
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (640 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (640 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (640 - SideLength * 2) : i * (640 - SideLength * 2) + 640,
                          j * (640 - SideLength * 2) : j * (640 - SideLength * 2) + 640]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (640 - SideLength * 2) : i * (640 - SideLength * 2) + 640,
                      (img.shape[1] - 640) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 640) : img.shape[0],
                      j * (640-SideLength*2) : j * (640 - SideLength * 2) + 640]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 640) : img.shape[0],
                  (img.shape[1] - 640) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (640 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (640 - SideLength * 2) + SideLength

    return TifArrayReturn, RowOver, ColumnOver

def process_images():
    nclasses = args.nclass
    
    model.eval()
    with torch.no_grad():
        model_state_file = args.weight_path
        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            warnings.warn('weight is not existed !!!"')
    
    area_perc = 0.5625
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * 640 / 2) 
    
    image_files = os.listdir(args.input_path)
    
    for image_file in image_files:
        image_path = os.path.join(args.input_path, image_file)
        print(image_path)
        dataset_img = gdal.Open(image_path)
        
        
        width = dataset_img.RasterXSize # 栅格矩阵的列数
        height = dataset_img.RasterYSize # 栅格矩阵的行数
        projection = dataset_img.GetProjection()
        geotransform = dataset_img.GetGeoTransform()
        img = dataset_img.ReadAsArray(0, 0, width, height) # 获取数据
        dataset_img = None
        img = img.transpose(1, 2, 0)
        TifArray, RowOver, ColumnOver = TifCroppingArray(img, RepetitiveLength)
        
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
        
        total_tasks = len(TifArray) * len(TifArray[0])
        task_count = 0
        
        results = []
        for j in range(len(TifArray)):
            for k in range(len(TifArray[0])):
                image = TifArray[j][k]  
                h, w, _ = image.shape
                image = transform(image)
                image = image.unsqueeze(0)#增加一个维度
                image = image.to(device)
                logits = model(image)
                logits = logits.argmax(dim=1)
                logits = logits.squeeze().cpu().detach().numpy()
                results.append(logits) 
                
                task_count += 1
                progress = int(task_count / total_tasks * 100)
                print(progress)
                
        result_shape = (img.shape[0], img.shape[1])
        result_data = Result(result_shape, TifArray, results, nclasses, RepetitiveLength, RowOver, ColumnOver)
               
        out_save = os.path.join(args.save_dir, 'result')            
        if not os.path.exists(out_save):
            os.makedirs(out_save)
            
        out_path = os.path.join(out_save, image_file)
                
        driver = gdal.GetDriverByName("GTiff")
        dataset_tif = driver.Create(out_path, result_shape[1], result_shape[0], 1, gdal.GDT_Byte)
        dataset_tif.SetGeoTransform(geotransform) #写入仿射变换参数
        dataset_tif.SetProjection(projection) #写入投影
        
        dataset_tif.GetRasterBand(1).WriteArray(result_data)
        del dataset_tif                

def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--class_name", type=tuple, default=('background', 'tailing_pond'))
    parser.add_argument("--palette", type=list, default=[0,0,0, 0,255,0])
    parser.add_argument("--models", type=str, default='deeplabv3',
                        choices=['danet', 'bisenetv2', 'pspnet',
                                 'deeplabv3', 'fcn', 'fpn', 'unet', 'unet_vgg', 'unet_resnet'])     
    parser.add_argument("--weight_path", type=str, default='F:/2.尾矿库/7.segmodel/work_dir/deeplabv3_lr0.0001_epoch100_batchsize2/weights/deeplabv3_lr0.0001_epoch100_batchsize2_epoch_27.pkl')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Select device (cpu/gpu)')
    parser.add_argument("--input_path", type=str, default='C:/Users/wangpan/Desktop/test')
    parser.add_argument("--save_dir", type=str, default='C:/Users/wangpan/Desktop/result')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = get_model()      
    process_images()
