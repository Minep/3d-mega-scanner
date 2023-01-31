from .Pipeline import PipelineStageBase
import open3d as o3d
import numpy as np

class AccRegistration:
    def __init__(self) -> None:
        self.__merged_pcd = None
        self.__prev_reg = None
        self.__prev_pcd = None
        self.__T_acc = np.identity(4)
        self.__prev_T = None
        self.__T_seq = []
        self.__voxelsz = 0.05

    def preprocess(self, pcd):
        pcd_lores = pcd.voxel_down_sample(self.__voxelsz)
        kdsearch2 = o3d.geometry.KDTreeSearchParamHybrid(radius=self.__voxelsz * 2, max_nn=30)
        kdsearch5 = o3d.geometry.KDTreeSearchParamHybrid(radius=self.__voxelsz * 5, max_nn=30)
        pcd_lores.estimate_normals(kdsearch2)
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_lores, kdsearch5)
        return pcd_lores, fpfh

    def global_pass(self, src_lores, src_fpfh):
        dest_lores, dest_fpfh = self.__prev_reg
        d = self.__voxelsz * 1.5
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(d)
        ]
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_lores, dest_lores, src_fpfh, dest_fpfh, True,
            d,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, checkers, criteria)

        return result
    
    def local_pass(self, src, initial):
        d = self.__voxelsz * 0.4
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15)
        solver = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        return o3d.pipelines.registration.registration_icp(
            src, self.__prev_pcd, d, initial.transformation,
            estimation_method=solver, criteria=criteria)

    def pairwise_reg(self, current):
        if self.__prev_reg is not None:
            src_lores, src_fpfh = self.preprocess(current)
            estimated = self.global_pass(src_lores, src_fpfh)
            aligned = self.local_pass(current, estimated)
            T = aligned.transformation
            self.add_transform(T)
            current.transform(self.__T_acc)
            self.__merged_pcd += current
            self.__merged_pcd.remove_duplicated_points()
        else:
            self.__merged_pcd = current
        self.__prev_pcd = current
        self.__prev_reg = self.preprocess(current)
        return self.__merged_pcd
    
    def get_merged(self):
        return self.__merged_pcd

    def add_transform(self, T):
        self.__T_acc = T @ self.__T_acc
        self.__prev_T = T
        self.__T_seq.append(T)

class RegistrationStage(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("registration")
        self.__voxelsz = 0.05

    def initialize(self, options):
        self.__accreg = AccRegistration()

    def execute(self, pcd):
        self.__accreg.pairwise_reg(pcd)

        return self.__accreg
