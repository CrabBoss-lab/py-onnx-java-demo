# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2023/5/16 9:20
# @Author :yujunyu
# @Site :
# @File :reset_transforms_resize.py
# @software: PyCharm

import torch
import numpy as np
from PIL import Image
import cv2
from collections.abc import Sequence

############
# 重写torchvision.transforms中Resize类
# 重写内容：将原本的内的resize改用cv2的resize
# 重写目的：使用cv2的interpolation
# 参考：https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py#L283
############

class CV2_Resize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
        interpolation (int, optional): Desired interpolation method. Default is `cv2.INTER_LINEAR`.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set to True for
            ``InterpolationMode.BILINEAR`` only mode. This can help making the output for PIL images and tensors
            closer.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.

    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, max_size=None, antialias=None):
        """
        :param size:
        :param interpolation: "cv2.INTER_LINEAR","cv2.INTER_CUBIC","cv2.INTER_AREA"
        :param max_size:
        :param antialias:
        """
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(img, torch.Tensor):
            # convert tensor to numpy array
            img = img.permute(1, 2, 0).cpu().numpy()
            # resize using cv2
            img = cv2.resize(img, self.size[::-1], interpolation=self.interpolation)
            # convert back to tensor
            img = torch.from_numpy(img).permute(2, 0, 1)
        else:
            # convert PIL image to numpy array
            img = np.array(img)
            # resize using cv2
            img = cv2.resize(img, self.size[::-1], interpolation=self.interpolation)
            # convert back to PIL image
            img = Image.fromarray(img)
            # resize again if necessary
            if self.max_size is not None:
                w, h = img.size
                if w > h:
                    new_w = self.max_size
                    new_h = int(h * (self.max_size / w))
                else:
                    new_h = self.max_size
                    new_w = int(w * (self.max_size / h))
                img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        return img

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"
