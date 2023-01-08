import numpy as np
import open3d as o3d
import cv2
from ..Pipeline import PipelineStageBase

class PointCloudVisualizer(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("visualizer (open3d)")

    def initialize(self, options):
        super().initialize(options)
        self.__v_opts = self._opts.visualization
    
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
        img = (in_obj / in_obj.max() * 255).astype(np.uint8)
        cv2.imwrite("depth.png", img)
        o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud("pcloud.ply", pcd)

class Viewer3D(object):
    def __init__(self, title):
        self.CLOUD_NAME = 'cloud3d'
        self.first_cloud = True
        #app = o3d.visualization.gui.Application.instance
        #app.initialize()

        self.main_vis = o3d.visualization.Visualizer()
        self.main_vis.create_window()
        #self.main_vis.show_skybox(False)
        #self.main_vis.enable_raw_mode(True)
        # app.add_window(self.main_vis)

    def tick(self):
        #app = o3d.visualization.gui.Application.instance
        #tick_return = app.run_one_tick()
        #if tick_return:
        self.main_vis.update_renderer()
        self.main_vis.poll_events()
        #return tick_return

    def update_cloud(self, geometries):
        if self.first_cloud:
            self.main_vis.add_geometry(geometries)
            # self.main_vis.reset_camera_to_default()
            # self.main_vis.setup_camera(60,
            #                            [4, 2, 5],
            #                            [0, 0, -1.5],
            #                            [0, -1, 0])
            self.first_cloud = False
        else:
            #self.main_vis.remove_geometry(geometries)
            self.main_vis.update_geometry(geometries)


class PointCloudVisualizerRT(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("visualizer (open3d)")

    def initialize(self, options):
        super().initialize(options)
        self.__v_opts = self._opts.visualization
        self.__vis = Viewer3D("cloud")
        self.__pcd = o3d.geometry.PointCloud()
    
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
        
        self.__pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
        # img = (in_obj / in_obj.max() * 255).astype(np.uint8)
        # cv2.imwrite("depth.png", img)
        self.__vis.update_cloud(self.__pcd)
        self.__vis.tick()
        # o3d.io.write_point_cloud("pcloud.ply", pcd)

    def cleanup(self):
        #self.__vis.destroy_window()
        pass