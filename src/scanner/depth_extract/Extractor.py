import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

from .lib.multi_depth_model_woauxi import RelDepthModel
from .lib.net_tools import load_ckpt
from .lib.spvcnn_classsification import SPVCNN_CLASSIFICATION
from .lib.test_utils import refine_focal, refine_shift

from ..Pipeline import PipelineStageBase

class DepthExtractLeReS(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("depth-extractor (LeReS)")

    def initialize(self, options):
        super().initialize(options)
        self.__depth_model = RelDepthModel(backbone=self._opts.depth_leres.backbone)
        self.__depth_model.eval()

        self.__shift_model, self.__focal_model = self.__make_shift_focallength_models()

        if not os.path.exists(self._opts.depth_leres.weights):
            raise Exception("Can not open weights file")

        load_ckpt(self._opts.depth_leres.weights, self.__depth_model, self.__shift_model, self.__focal_model)
        
        self.__depth_model.cuda()
        self.__shift_model.cuda()
        self.__focal_model.cuda()

    def __make_shift_focallength_models(self):
        shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                            num_classes=1,
                                            cr=1.0,
                                            pres=0.01,
                                            vres=0.01
                                            )
        focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                            num_classes=1,
                                            cr=1.0,
                                            pres=0.01,
                                            vres=0.01
                                            )
        shift_model.eval()
        focal_model.eval()
        return shift_model, focal_model

    def __recover_shift(self, ori_size, pred_depth):
        cam_u0 = ori_size[1] / 2.0
        cam_v0 = ori_size[0] / 2.0
        pred_depth_norm = pred_depth - pred_depth.min() + 0.5

        dmax = np.percentile(pred_depth_norm, 98)
        pred_depth_norm = pred_depth_norm / dmax

        # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
        proposed_scaled_focal = (ori_size[0] // 2 / np.tan((60/2.0)*np.pi/180))

        # recover focal
        focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, self.__focal_model, u0=cam_u0, v0=cam_v0)
        predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()

        # recover shift
        shift_1 = refine_shift(pred_depth_norm, self.__shift_model, predicted_focal_1, cam_u0, cam_v0)
        shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
        depth_scale_1 = pred_depth_norm - shift_1.item()

        # recover focal
        # focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, self.__focal_model, u0=cam_u0, v0=cam_v0)
        # predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

        return shift_1, depth_scale_1

    def __scale_torch(self, img):
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        if img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img

    def execute(self, in_obj):
        rgb_c = in_obj[:, :, ::-1].copy()
        A_resize = cv2.resize(rgb_c, (448, 448))

        img_torch = self.__scale_torch(A_resize)[None, :, :, :]
        pred_depth = self.__depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (in_obj.shape[1] // 2, in_obj.shape[0] // 2))
        _, depth_scaled = self.__recover_shift(in_obj.shape, pred_depth_ori)
        # plt.imsave("depth.png", pred_depth_ori, cmap='rainbow')
        return depth_scaled