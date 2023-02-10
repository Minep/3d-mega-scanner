from ..Pipeline import PipelineStageBase
import open3d as o3d
import numpy as np
import cv2

class PointCloudStage(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("pcd")
    
    def initialize(self, options):
        super().initialize(options)
        self.__v_opts = self._opts.visualization

    def focal_to_pixel_focal(self, f_mm, sensor_w, img_w):
        return f_mm * img_w / sensor_w

    def depth_conversion(self, in_obj, f, z_scale=1, chop_range=[0.0, 1.0]):
        D, rgb, f = in_obj
        D = D / D.max() * 5
        w, h = D.shape
        pt_cloud = np.zeros((w * h, 3, 2))
        U = w / 2 - np.arange(0, w, 1)
        V = h / 2 - np.arange(0, h, 1)
        Dflatten = D.ravel().reshape(-1, 1)
        UV = np.dstack(np.meshgrid(V, U)).reshape(-1, 2)
        pt_cloud[:,:,0] = np.hstack([
            UV * Dflatten / f,
            Dflatten
        ])
        pt_cloud[:,:,1] = rgb.reshape((w * h, 3)) / 255
        return pt_cloud

    def execute(self, in_obj):
        f = self._opts.focal_len
        if self.__v_opts.use_real_focal:
            f = self.focal_to_pixel_focal(f, self.__v_opts.sensor_width, in_obj[0].shape[0])
        pts = self.depth_conversion(in_obj, f, self.__v_opts.z_scale, chop_range=self.__v_opts.chop_range)
        ptc = o3d.geometry.PointCloud()
        ptc.points = o3d.utility.Vector3dVector(pts[:,:,0])
        ptc.colors = o3d.utility.Vector3dVector(pts[:,:,1])
        pc, _ = ptc.remove_statistical_outlier(15, 3)
        return pc

    def cleanup(self):
        pass