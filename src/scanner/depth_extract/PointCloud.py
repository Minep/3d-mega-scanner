from ..Pipeline import PipelineStageBase
import open3d as o3d
import numpy as np

class PointCloudStage(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("pcd")

    def focal_to_pixel_focal(self, f_mm, sensor_w, img_w):
        return f_mm * img_w / sensor_w

    def depth_conversion(self, D, f, z_scale=1, chop_range=[0.0, 1.0]):
        M = np.diag([1, 1, f, 1])
        u_vec = np.zeros((2,))
        pt_cloud = []
        w, h = D.shape
        upper_thr = D.max() * chop_range[1]
        lower_thr = D.max() * chop_range[0]
        for u_ in range(w):
            for v_ in range(h):
                d = D[u_,v_]
                if (d > upper_thr or d < lower_thr):
                    continue
                zt = d / f
                u = w / 2 - u_
                v = h / 2 - v_
                u_vec = np.array([v, u, 1, 1 / zt])
                pos = zt * (M @ u_vec.T)
                pt_cloud.append(pos.ravel())
        return np.array(pt_cloud)

    def execute(self, in_obj):
        f = self._opts.focal_len
        if self.__v_opts.use_real_focal:
            f = self.focal_to_pixel_focal(f, self.__v_opts.sensor_width, in_obj.shape[0])
        pts = self.depth_conversion(in_obj, f, self.__v_opts.z_scale, chop_range=self.__v_opts.chop_range)
        v3v = o3d.utility.Vector3dVector(pts[:,:3])
        return o3d.geometry.PointCloud(v3v)

    def cleanup(self):
        pass