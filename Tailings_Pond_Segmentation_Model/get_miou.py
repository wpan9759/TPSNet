from PIL import Image
from pathlib import Path
import numpy as np

"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
__all__ = ['SegmentationMetric']
 
"""
confusionMetric
L\P     P    N
 
P      TP    FN
 
N      FP    TN
 
"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def overallAccuracy(self):
        # return all class overall pixel accuracy,AO评价指标
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
  
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=0) + np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        print('IoU:', IoU)
        mIoU = np.nanmean(IoU)
        return mIoU
    def precision(self):
        #precision = TP / TP + FP
        p = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return p
    
    def recall(self):
        #recall = TP / TP + FN
        r = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return r
 
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)#过滤掉其它类别
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
 
 
if __name__ == '__main__':   
    true_path = 'F:/pond_luanping/5.semantic_segmentation/exp1/dataset/test/labels/'
    pred_path = 'F:/pond_luanping/5.semantic_segmentation/exp1/dataset/test/results/'
    class_num = 2
    metric = SegmentationMetric(class_num)
    true_list = Path(true_path).rglob('*')
    pred_list = Path(pred_path).rglob('*')
    
    for true, pred in zip(true_list, pred_list):
        true_img = np.array(Image.open(true), dtype=np.uint8).flatten()
        pred_img = np.array(Image.open(pred), dtype=np.uint8).flatten()
        metric.addBatch(pred_img, true_img)
      
    oa = metric.overallAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    p = metric.precision()
    mp = np.nanmean(p)
    r = metric.recall()
    mr = np.nanmean(r)
    f1 = (2*p*r) / (p + r)
    mf1 = np.nanmean(f1)

    print(f'类别0,类别1,...\n  oa:{oa}, mIou:{mIoU}, p:{p}, mp:{mp}, r:{r}, mr:{mr}, f1:{f1}, mf1:{mf1}')