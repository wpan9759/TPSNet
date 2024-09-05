import argparse
import torch
import random
import numpy as np
from torch import nn

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
from PIL import Image
from torch.utils.data import DataLoader
from datasets import SlippyMapTilesConcatenation

from loss import CrossEntropyLoss
from seg_metric import SegmentationMetric
import tqdm
import os

import shutil
import time
import logging

def get_dataset_loaders(args):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(args.crop_size, Image.Resampling.BILINEAR), Resize(args.crop_size, Image.Resampling.NEAREST)),
            JointTransform(CenterCrop(args.crop_size), CenterCrop(args.crop_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    train_dataset = SlippyMapTilesConcatenation(args.train_input, args.train_target, transform)

    val_dataset = SlippyMapTilesConcatenation(args.val_input, args.val_target, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True, drop_last=True, num_workers=0)#设置batchsize,是否打乱，是否丢弃，线程
    val_loader = DataLoader(val_dataset, batch_size=args.train_batchsize, shuffle=False, drop_last=True, num_workers=0)
        
    return train_loader, val_loader

class FullModel(nn.Module):

    def __init__(self, model, args2):
        super(FullModel, self).__init__()
        self.model = model
        self.ce_loss = CrossEntropyLoss()

    def forward(self, input, label=None, train=True):

        output = self.model(input)
        losses = 0
        if isinstance(output, (list, tuple)):
            for i in range(len(output)):
                loss = self.ce_loss(output[i], label)
                losses += loss
            return losses, output[0]
        else:
            losses = self.ce_loss(output, label)
            return losses, output


def get_model(args, device, models):
    print(models)

    nclass = args.nclass
    assert models in ['danet', 'bisenetv2', 'pspnet',
                      'deeplabv3', 'fcn', 'fpn', 'unet', 'unet_vgg', 'unet_resnet']
    if models == 'danet':
        from models.danet import DANet
        model = DANet(nclass=nclass, backbone='resnet50', pretrained_base=args.pretrained)
    if models == 'bisenetv2':
        from models.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(nclass=nclass)
    if models == 'pspnet':
        from models.pspnet import PSPNet
        model = PSPNet(nclass=nclass, backbone='resnet50', pretrained_base=args.pretrained)
    if models == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(nclass=nclass, backbone='resnet50', pretrained_base=args.pretrained)
    if models == 'fcn':
        from models.fcn import FCN16s
        model = FCN16s(nclass=nclass, backbone='vgg16')
    if models == 'fpn':
        from models.fpn import FPN
        model = FPN(nclass=nclass)
    if models == 'unet':
        from models.unet import UNet
        model = UNet(nclass=nclass)
    if models == 'unet_vgg':
        from models.unet_vgg import Unet_vgg
        model = Unet_vgg(nclass=nclass, backbone='vgg16')
    if models == 'unet_resnet':
        from models.unet_resnet import Unet_resnet
        model = Unet_resnet(nclass=nclass, backbone='resnet50', pretrained_base=args.pretrained)

    model = FullModel(model, args)
    model = model.to(device)
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def train(dataloader_train, device, model, args2, optimizer, epoch):
    model.train() 
    MIOU = [0]
    ACC = [0]
    F1 = [0]
    Precision = [0]
    Recall = [0]
    
    nclass = args2.nclass
    metric = SegmentationMetric(nclass)
    ave_loss = AverageMeter()
        
    for image, label in tqdm.tqdm(dataloader_train):
        image, label = image.to(device), label.to(device)
        label = label.long()
        losses, logit = model(image, label)

        loss = losses.mean()
        ave_loss.update(loss.item())
        
        logit = logit.argmax(dim=1)
        logit = logit.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        metric.addBatch(logit, label)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    reduced_loss = ave_loss.average()
    print_loss = torch.from_numpy(np.array(reduced_loss)).to(device).cpu().item()    

    iou = metric.IntersectionOverUnion()
    acc = metric.Accuracy()
    precision = metric.Precision()
    recall = metric.Recall()
    miou = np.nanmean(iou[0:args2.nclass])
    mprecision = np.nanmean(precision[0:args2.nclass])
    mrecall = np.nanmean(recall[0:args2.nclass])

    MIOU = MIOU + miou
    ACC = ACC + acc
    Recall = Recall + mrecall
    Precision = Precision + mprecision
    F1 = F1 + 2 * Precision * Recall / (Precision + Recall)

    MIOU = torch.from_numpy(MIOU).to(device).item()
    ACC = torch.from_numpy(ACC).to(device).item()
    F1 = torch.from_numpy(F1).to(device).item()
    Recall = torch.from_numpy(Recall).to(device).item()
    Precision = torch.from_numpy(Precision).to(device).item()
    Lr = optimizer.param_groups[0]['lr']
    return {
        "miou_t":MIOU,
        "acc_t":ACC,
        "f1_t":F1,
        "precision_t":Precision,
        "recall_t":Recall,
        "loss_t":print_loss,
        "lr_t":Lr
        }

def validate(dataloader_val, device, model, args):
    model.eval()
    MIOU = [0]
    ACC = [0]
    F1 = [0]
    Precision = [0]
    Recall = [0]
       
    nclass = args.nclass
    metric = SegmentationMetric(nclass)
    ave_loss = AverageMeter()
    with torch.no_grad():
        for image, label in tqdm.tqdm(dataloader_val):
            image, label = image.to(device), label.to(device)
            label = label.long()
            losses, logit = model(image, label, train=False)
            
            loss = losses.mean()
            ave_loss.update(loss.item())
            
            logit = logit.argmax(dim=1)
            logit = logit.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            metric.addBatch(logit, label)
    
    reduced_loss = ave_loss.average()
    print_loss = torch.from_numpy(np.array(reduced_loss)).to(device).cpu().item()
        
    iou = metric.IntersectionOverUnion()
    acc = metric.Accuracy()
    precision = metric.Precision()
    recall = metric.Recall()
    miou = np.nanmean(iou[0:nclass])
    mprecision = np.nanmean(precision[0:nclass])
    mrecall = np.nanmean(recall[0:nclass])

    MIOU = MIOU + miou
    ACC = ACC + acc
    Recall = Recall + mrecall
    Precision = Precision + mprecision
    F1 = F1 + 2 * Precision * Recall / (Precision + Recall)

    MIOU = torch.from_numpy(MIOU).to(device).item()
    ACC = torch.from_numpy(ACC).to(device).item()
    F1 = torch.from_numpy(F1).to(device).item()
    Recall = torch.from_numpy(Recall).to(device).item()
    Precision = torch.from_numpy(Precision).to(device).item()
    return {
        "miou_v":MIOU,
        "acc_v":ACC,
        "f1_v":F1,
        "precision_v":Precision,
        "recall_v":Recall,
        "loss_v":print_loss,
        }    

def save_model_file(save_dir, save_name):
    save_dir = os.path.join(save_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/weights/')
    for file in os.listdir('.'):
        if os.path.isfile(file):
            shutil.copy(file, save_dir)
    if not os.path.exists(os.path.join(save_dir, 'models')):
        shutil.copytree('./models', os.path.join(save_dir, 'models'))

def main():
    args2 = parse_args()
    
    torch.manual_seed(args2.seed)
    torch.cuda.manual_seed(args2.seed)
    torch.cuda.manual_seed_all(args2.seed)
    random.seed(args2.seed)
    np.random.seed(args2.seed)
    
    save_name = "{}_lr{}_epoch{}_batchsize{}".format(args2.models, args2.lr, args2.end_epoch,
                                                        args2.train_batchsize)
    save_dir = args2.save_dir
    save_model_file(save_dir=save_dir, save_name=save_name)
    
    weight_save_dir = os.path.join(save_dir, save_name + '/weights')
    logging.basicConfig(filename=os.path.join(save_dir, save_name) + '/train.log', level=logging.INFO)
    
    train_loader, val_loader = get_dataset_loaders(args2)
    
    if args2.device == 'cpu':
        device = torch.device('cpu')
    elif args2.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    model = get_model(args2, device, models=args2.models)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args2.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    
    best_miou = 0
    
    # 添加早停相关的参数和计数器
    if args2.early_stop:
        early_stop_patience = args2.early_stop_patience
    else:
        early_stop_patience = None
    best_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(args2.end_epoch):
        print("Epoch: {}/{}".format(epoch + 1, args2.end_epoch))
        start_train = time.time()
        train_hist = train(train_loader, device, model, args2, optimizer, epoch)
        end_train = time.time()
        train_time = end_train - start_train
        print("train_epoch:[{}/{}], miou:{:.4f}, acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f}, lr:{:.6f}, time:{:.4f}s".
              format(epoch + 1,args2.end_epoch, train_hist['miou_t'], train_hist['acc_t'], train_hist['f1_t'], train_hist['precision_t'], train_hist['recall_t'], train_hist['loss_t'], train_hist['lr_t'], train_time))
        lr_scheduler.step()
        
        start_val = time.time() 
        val_hist = validate(val_loader, device, model, args2)#开始验证
        end_val = time.time()
        val_time = end_val - start_val
        print("valid_epoch:[{}/{}], miou:{:.4f}, acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f}, time:{:.4f}s".
              format(epoch + 1,args2.end_epoch, val_hist['miou_v'], val_hist['acc_v'], val_hist['f1_v'], val_hist['precision_v'], val_hist['recall_v'], val_hist['loss_v'], val_time))
        
        logging.info("train_epoch:[{}/{}], miou:{:.4f}, acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f},lr:{:.6f}, time:{:.4f}s, valid_epoch:[{}/{}], miou:{:.4f}, acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f}, time:{:.4f}s".
                     format(epoch + 1,args2.end_epoch,train_hist['miou_t'], train_hist['acc_t'], train_hist['f1_t'], train_hist['precision_t'], train_hist['recall_t'], train_hist['loss_t'], train_hist['lr_t'], train_time, 
                            epoch + 1,args2.end_epoch,val_hist['miou_v'], val_hist['acc_v'], val_hist['f1_v'], val_hist['precision_v'], val_hist['recall_v'], val_hist['loss_v'], val_time)
                     )
        torch.save(model.state_dict(),
                       weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_epoch_{}.pkl'
                       .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, epoch))
        if val_hist['miou_v'] >= best_miou and val_hist['miou_v'] != 0:
            best_miou = val_hist['miou_v']
            best_weight_name = weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_best_epoch_{}.pkl'.format(
                    args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, epoch)
            torch.save(model.state_dict(), best_weight_name)
            torch.save(model.state_dict(), weight_save_dir + '/best_weight.pkl')
        
        if early_stop_patience is not None:
            if val_hist['loss_v'] < best_loss:
                best_loss = val_hist['loss_v']
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print("early_stop_counter:",early_stop_counter)

            # 判断是否触发早停
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered. No improvement in {} epochs.".format(early_stop_patience))
                #break
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--end_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--train_batchsize", type=int, default=2)
    parser.add_argument("--val_batchsize", type=int, default=2)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[640, 640], help='H, W')
    parser.add_argument("--models", type=str, default='unet_vgg',
                        choices=['danet', 'bisenetv2', 'pspnet',
                                 'deeplabv3', 'fcn', 'fpn', 'unet', 'unet_vgg', 'unet_resnet'])
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default='./work_dir')
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Select device (cpu/gpu)')
    parser.add_argument("--train_input", type=str, default='data/train/images')
    parser.add_argument("--train_target", type=str, default='data/train/labels')
    parser.add_argument("--val_input", type=str, default='data/valid/images')
    parser.add_argument("--val_target", type=str, default='data/valid/labels')
    parser.add_argument("--pretrained", type=str, default= 'True')
    parser.add_argument("--early_stop", type=str, default='True')
    parser.add_argument("--early_stop_patience", type=int, default=10)

    args2 = parser.parse_args()
    return args2

if __name__ == '__main__':
    main()
