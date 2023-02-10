import argparse
from scanner.Pipeline import Pipeline, PipelineStageBase, PipelineOptions
from scanner.depth_extract.Extractor import DepthExtractLeReS
from scanner.depth_extract.Visualize import AccPtCVisualizerRT, PointCloudVisualizer, PointCloudVisualizerRT
from scanner.depth_extract.PointCloud import PointCloudStage
from scanner.Registration import MultiwayRegistrationStage, RegistrationStage

from v4l2 import GstV4L2CaptureWrap
from video_file import OpenCVFileWrap
from options import global_options

import cv2

class CV2ReadImage(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("image_input (cv2)")
    def initialize(self, options):
        self.__opts = options

    def execute(self, in_obj):
        return cv2.imread(in_obj)

def main(file):
    options = PipelineOptions(global_options)
    pipeline = Pipeline([
        CV2ReadImage(),
        DepthExtractLeReS(),
        PointCloudStage(),
        PointCloudVisualizer()
    ])

    pipeline.initialize(options)
    pipeline.execute(file)

def main_seq(file):
    options = PipelineOptions(global_options)
    pipeline = OpenCVFileWrap(file,Pipeline([
        DepthExtractLeReS()
    ]))

    pipeline.initialize(options)
    pipeline.execute()

def main_seq_file(file):
    options = PipelineOptions(global_options)
    pipeline = OpenCVFileWrap(file, Pipeline([
        DepthExtractLeReS(),
        PointCloudStage(),
        RegistrationStage(),
        AccPtCVisualizerRT()
    ]))

    pipeline.initialize(options)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode')
    parser.add_argument('-f', '--file')
    args = parser.parse_args()
    if args.mode == "file":
        main_seq_file(args.file)
    elif args.mode == "img":
        main(args.file)
    elif args.mode == "depth":
        main_seq(args.file)
