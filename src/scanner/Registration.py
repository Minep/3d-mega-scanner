from .Pipeline import PipelineStageBase
import open3d as o3d
import numpy as np

class AccRegistration:
    def __init__(self) -> None:
        self.__merged_pcd = None
        self.__prev_reg = None
        self.__prev_pcds = []
        self.__T_acc = np.identity(4)
        self.__prev_T = np.identity(4)
        self.__T_seq = []
        self.__voxelsz = 0.03
        self.pyraimd_scales = [0.05, 0.03, 0.02, 0.01]

    def preprocess(self, pcd, sz):
        pcd_lores = pcd.voxel_down_sample(sz)
        # kdsearch5 = o3d.geometry.KDTreeSearchParamHybrid(radius=self.__voxelsz * 5, max_nn=30)
        kdsearch2 = o3d.geometry.KDTreeSearchParamHybrid(sz * 2, max_nn=30)
        pcd_lores.estimate_normals(kdsearch2)
        # fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_lores, kdsearch5)
        return pcd_lores

    def estimate_normal(self, pcd):
        kdsearch2 = o3d.geometry.KDTreeSearchParamHybrid(radius=self.__voxelsz * 2, max_nn=30)
        pcd.estimate_normals(kdsearch2)
        # fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_lores, kdsearch5)
        return pcd, None

    def coarse_pass(self, src_lores, src_fpfh):
        dest_lores, dest_fpfh = self.__prev_reg
        d = self.__voxelsz * 2
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(d)
        ]
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.99)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_lores, dest_lores, src_fpfh, dest_fpfh, True,
            d,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            3, checkers, criteria)

        return result
    
    def local_pass(self, src, target, initT, scale):
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        # solver = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        return o3d.pipelines.registration.registration_colored_icp(
            src, target, scale, initT, criteria=criteria)

    def pairwise_reg(self, current):
        if len(self.__prev_pcds) > 0:
            T = np.identity(4)
            for i, target in enumerate(self.__prev_pcds):
                scale = self.pyraimd_scales[i]
                src = self.preprocess(current, scale)
                aligned = self.local_pass(src, target, T, scale)
                T = aligned.transformation
                self.__prev_pcds[i] = src
            self.add_transform(T)
            self.__merged_pcd += current.transform(self.__T_acc)
            # self.__merged_pcd, _ = self.__merged_pcd.remove_statistical_outlier(20, 10)
        else:
            self.__merged_pcd = current
            for scale in self.pyraimd_scales:
                self.__prev_pcds.append(self.preprocess(current, scale))
        return self.__merged_pcd
    
    def get_merged(self):
        return self.__merged_pcd

    def add_transform(self, T):
        self.__T_acc =  self.__T_acc @ T
        self.__prev_T = T
        self.__T_seq.append(T)

class MultiwayRegistration:
    def __init__(self) -> None:
        self.__pcds = []
        self.__pose_graph = o3d.pipelines.registration.PoseGraph()
        self.T_g = np.identity(4)

    def preprocess(self, pcd, sz):
        pcd_lores = pcd.voxel_down_sample(sz)
        kdsearch2 = o3d.geometry.KDTreeSearchParamHybrid(sz * 2, max_nn=30)
        pcd_lores.estimate_normals(kdsearch2)
        return pcd_lores

    def coarse_align(self, src, target):
        d = 0.06
        src_lores = self.preprocess(src, d)
        target_lores = self.preprocess(target, d)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        # solver = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        T = np.identity(4)
        T = o3d.pipelines.registration.registration_icp(
                src_lores, target_lores, d, T, criteria=criteria).transformation
        T = o3d.pipelines.registration.registration_icp(
                src_lores, target_lores, d / 2, T, criteria=criteria).transformation
        L = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                            src_lores, target_lores, d / 2, T)
        return T, L

    def add_pairwise(self, pcd):
        i = len(self.__pcds)
        if i == 0:
            self.__pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.T_g))
            self.__pcds.append(pcd)
            return
        for j in range(0, i):
            j = i - j - 1
            T, L = self.coarse_align(pcd, self.__pcds[j])
            if j == i - 1:
                self.T_g = self.T_g @ T
                self.__pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.T_g))
                self.__pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(i, j, T, L, uncertain=False))
                break
            else:
                self.__pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(i, j, T, L, uncertain=True))
        self.__pcds.append(pcd)


    def optimize(self):
        option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=0.02,
                    edge_prune_threshold=0.25,
                    reference_node=0)
        o3d.pipelines.registration.global_optimization(
                self.__pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)

    def get_merged(self):
        pcds = o3d.geometry.PointCloud()
        for i, pcd in enumerate(self.__pcds):
            T = self.__pose_graph.nodes[i].pose
            pcds += pcd.transform(T)
        return pcds

class RegistrationStage(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("registration")
        self.__voxelsz = 0.05

    def initialize(self, options):
        self.__accreg = AccRegistration()

    def execute(self, pcd):
        try:
            self.__accreg.pairwise_reg(pcd)
        except:
            print("WARN: Unable to find correspondence.")

        return self.__accreg

class MultiwayRegistrationStage(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("registration (multi-way)")
        self.__voxelsz = 0.05

    def initialize(self, options):
        self.__mwreg = MultiwayRegistration()
        self.__count = 0

    def execute(self, pcd):
        self.__mwreg.add_pairwise(pcd)
        print("Add pairwise")

        if (self.__count % 10 == 0):
            print("Optimize")
            self.__mwreg.optimize()

        self.__count+=1
        return self.__mwreg