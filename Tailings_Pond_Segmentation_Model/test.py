import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    ConvertImageMode,
    ImageToTensor,
    MaskToTensor,
)
from torchvision.transforms import Resize, CenterCrop, Normalize
from datasets import SlippyMapTilesConcatenation
from seg_metric import SegmentationMetric
import logging
import warnings

from PIL import Image

import time

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
        model = FCN16s(nclass=args.nclass, backbone='vgg16')
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
        model = Unet_resnet(nclass=args.nclass, backbone='resnet50', pretrained_base=False)

    model = model.to(device)
    return model

def process_images():
    nclasses = args.nclass
    model.eval()
    metric = SegmentationMetric(numClass=nclasses)
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

        for i, (image, label) in enumerate(val_loader):

            image, label = image.to(device), label.to(device)
            label = label.long()
            logits = model(image)
            print("test:{}/{}".format(i, len(val_loader)))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = label.cpu().detach().numpy()
            out = Image.fromarray(np.uint8(logits.squeeze())).convert('P')
            
            out_save = os.path.join(args.save_dir, 'result')            
            if not os.path.exists(out_save):
                os.makedirs(out_save)
            image_name = os.path.splitext(os.path.basename(val_dataset.inputs[i]))[0]  # 提取原始图像名称
            out_name = f"{image_name}.png"  # 使用原始图像名称作为输出PNG文件名
            out_path = os.path.join(out_save, out_name)
            out.putpalette(args.palette)
            out.save(out_path)
            
            if args.mode == 'evaluate':        
                metric.addBatch(logits, labels)
        if args.mode == 'evaluate':                              
            result_count(metric)

def result_count(metric):
    iou = metric.IntersectionOverUnion()
    miou = np.nanmean(iou[0:2])
    acc = metric.Accuracy()
    f1 = metric.F1()
    mf1 = np.nanmean(f1[0:2])
    precision = metric.Precision()
    mprecision = np.nanmean(precision[0:2])
    recall = metric.Recall()
    mrecall = np.nanmean(recall[0:2])
    

    iou = torch.from_numpy(np.array(iou)).to(device).cpu().numpy()
    miou = torch.from_numpy(np.array(miou)).to(device).cpu().numpy()
    acc = torch.from_numpy(np.array(acc)).to(device).cpu().numpy()
    f1 = torch.from_numpy(np.array(f1)).to(device).cpu().numpy()
    mf1 = torch.from_numpy(np.array(mf1)).to(device).cpu().numpy()
    precision = torch.from_numpy(np.array(precision)).to(device).cpu().numpy()
    mprecision = torch.from_numpy(np.array(mprecision)).to(device).cpu().numpy()
    recall = torch.from_numpy(np.array(recall)).to(device).cpu().numpy()
    mrecall = torch.from_numpy(np.array(mrecall)).to(device).cpu().numpy()

    print('\n')
    logging.info('####################### full image val ###########################')
    print('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                 str('Precision').rjust(10), str('Recall').rjust(10),
                                 str('F1').rjust(10), str('IOU').rjust(10)))
    logging.info('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                        str('Precision').rjust(10), str('Recall').rjust(10),
                                        str('F1').rjust(10), str('IOU').rjust(10)))
    for i in range(len(iou)):
        print('|{}:{}{}{}{}|'.format(str(args.class_name[i]).ljust(24),
                                     str(round(precision[i], 4)).rjust(10), str(round(recall[i], 4)).rjust(10),
                                     str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
        logging.info('|{}:{}{}{}{}|'.format(str(args.class_name[i]).ljust(24),
                                            str(round(precision[i], 4)).rjust(10),
                                            str(round(recall[i], 4)).rjust(10),
                                            str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
    print('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                  round(acc * 100, 2), round(mf1 * 100, 2),
                                                                  round(mprecision * 100, 2),
                                                                  round(mrecall * 100, 2)))
    logging.info('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                         round(acc * 100, 2), round(mf1 * 100, 2),
                                                                         round(mprecision * 100, 2),
                                                                         round(mrecall * 100, 2)))
    print('\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')
    parser.add_argument('--mode', choices=['evaluate', 'predict'], default='predict', help='Choose mode: evaluate or predict')
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--class_name", type=tuple, default=('background', 'tailing_pond'))
    parser.add_argument("--palette", type=list, default=[0,0,0, 0,255,0])
    parser.add_argument("--models", type=str, default='deeplabv3',
                        choices=['danet', 'bisenetv2', 'pspnet',
                                 'deeplabv3', 'fcn', 'fpn', 'unet', 'unet_vgg', 'unet_resnet'])    
    parser.add_argument("--weight_path", type=str, default='F:/2.tailing/7.segmodel/work_dir/deeplabv3_lr0.0001_epoch100_batchsize2/weights/deeplabv3_lr0.0001_epoch100_batchsize2_epoch_96.pkl')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Select device (cpu/gpu)')
    parser.add_argument("--crop_size", type=int, nargs='+', default=[640, 640], help='H, W')
    parser.add_argument("--input_path", type=str, default='F:/2.tailing/12.sentinel-2/4.tailing')
    parser.add_argument("--target_path", type=str, default='F:/2.tailing/12.sentinel-2/4.tailing')
    parser.add_argument("--save_dir", type=str, default='F:/2.tailing/12.sentinel-2/5.tailing-result')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    save_log = os.path.join(args.save_dir, 'test.log')
    logging.basicConfig(filename=save_log, level=logging.INFO)
    
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = get_model() 
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(args.crop_size, Image.Resampling.BILINEAR), Resize(args.crop_size, Image.Resampling.NEAREST)),
            JointTransform(CenterCrop(args.crop_size), CenterCrop(args.crop_size)),
            JointRandomHorizontalFlip(0),
            JointRandomRotation(0, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    val_dataset = SlippyMapTilesConcatenation(args.input_path, args.target_path, transform)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    
    total_start_time = time.time()
    process_images()
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total execution time: {total_elapsed_time:.2f} seconds.")