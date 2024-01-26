#!/bin/bash

# Define the Python command
python_command="python demo.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --ckpt ../pointrcnn_7870.pth --data_path ../data/kitti/testing/velodyne/000002.bin"

# Execute the Python command
$python_command

