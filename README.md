# OFRMask
a mask method for monocular depth estimation using optical flow network and reconstructed points



## Data Preparation

Please refer to Monodepth2[GitHub](https://github.com/nianticlabs/monodepth2) to prepare your KITTI data.

## For training the network







python tain.py --data_path
../datasets/kitti
--model
lite-mono-8m
--num_epochs
30
--batch_size
12
--mypretrain
./pretrain/encoder.pth
--lr
0.0001
5e-6
31
0.0001
1e-5
31
















## For testing the network









python evaluate_depth.py --load_weights_folder
/home/jsw/1litemono/Lite-Mono-optic_flow_central_normalize_normalize_theta_0.98/tmp/lite-mono/models/weights_29
--data_path
../datasets/kitti/
--model
lite-mono-8m
