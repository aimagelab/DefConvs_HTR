import cv2
import numpy as np
from albumentations import Compose as AlbumentationsCompose


class Resize(object):
    def __init__(self, img_height):
        # type: (int) -> None
        self.img_height = img_height

    def __call__(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        if img.shape[0] != self.img_height:
            scale_percent = self.img_height / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=scale_percent, fy=scale_percent,
                             interpolation=cv2.INTER_CUBIC)
        return img


class Normalize(object):
    def __call__(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        img = img.astype(np.float32)
        # img = img / 127.5 - 1
        img = img / 128 - 1
        return img
    
    
class AlbumentationsTransforms(object):
    def __init__(self, *args):
        self.composed_transforms = AlbumentationsCompose(list(args))

    def __call__(self, img):
        augmented = self.composed_transforms(image=img)
        return augmented['image']
