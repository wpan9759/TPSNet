from path import Path

import torch
from PIL import Image
import torch.utils.data
from torchvision.transforms import Compose, Normalize
from transforms import ConvertImageMode, ImageToTensor, MaskToTensor

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class SlippyMapTilesConcatenation(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format.
    """

    def __init__(self, inputs, target, joint_transform=None,test = False):
        super().__init__()

        self.test = test        
        self.joint_transform = joint_transform
        self.test_transform1 =Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])
        self.test_transform2 = Compose([MaskToTensor()])

        self.inputs = Path(inputs).files()
        self.target = Path(target).files()

        
    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, i):
        # at this point all transformations are applied and we expect to work with raw tensors

        images = Image.open(self.inputs[i])
        if self.test == False:
            mask = Image.open(self.target[i]).convert('P')
        
            if self.joint_transform is not None:
                images, mask = self.joint_transform(images, mask)

            if len(mask.shape) == 3: 
                mask = mask.squeeze(0)
            return images, mask
        else:           
            images = self.test_transform1(images)
            mask = Image.open(self.target[i]).convert('P')
            mask = self.test_transform2(mask)
            if len(mask.shape) == 3: 
                mask = mask.squeeze(0)            
            return images, mask