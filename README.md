# OFRMask
a mask method for monocular depth estimation using optical flow network and reconstructed points

## Environment Preparation
Please refer to my [CSDN]https://blog.csdn.net/qq_42108573/article/details/131694107?spm=1001.2014.3001.5502  
or   
conda create -n myenv python=3.7  
conda activate myenv  
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html  
conda install packaging  
pip install tqdm  
pip install tensorboardx  
pip install timm  
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple   
pip install einops  
pip install prettytable  
pip install matplotlib  
pip install future tensorboard  
## Data Preparation

Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare your KITTI data.
## Pretrained Weights Preparation  
[Litemono-8m-backbone]https://surfdrive.surf.nl/files/index.php/s/oil2ME6ymoLGDlL   
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
