
global_options = {
    "depth_leres": {
        "backbone": "resnext101",
        "weights": "scanner/weights/res101.pth"
    },
    "visualization": {
        "sensor_width": 7.6,
        "use_real_focal": False,
        "z_scale": 20,
        "chop_range": [0,1]
    },
    "focal_len": 270,
    "capture_dev": "/dev/video0",
    "fps": 2
}