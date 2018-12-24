# Fast_Stacked_Hourglass_Network_OpenVino
A fast stacked hourglass network for human pose estimation on OpenVino. Stacked hourglass network proposed by  [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937) is a very good network for single-person pose estimation regarding to speed and accuracy.
This repo contains a demo to show how to depoly model trained by Keras. It converts a Keras model to IR and shows how to use the generated IR to do inference.
Have fun with OpenVino!

## Installation
- Python3
- Install OpenVino 2018 R5
- Install python dependencies
```
keras==2.1.5
scipy==1.2.0
tensorflow==1.12.0
opencv-python==3.4.3.18
```

## How to convert pre-trained models
### Download pre-trained hourglass models 
- Download models from Google drive and save them to `models`. You are going to download two files, one is json file for network configuration while another is weight.
- [hg_s2_b1_mobile](https://drive.google.com/drive/folders/12ioJONmse658qc9fgMpzSy2D_JCdkFVg?usp=sharing), inputs: 256x256x3, Channel Number: 256, pckh 78.86% @MPII.
- [hg_s2_b1_tiny](https://drive.google.com/open?id=1noM_3hu_55STzghKOeapciMop7A1dllV), inputs:192x192x3, Channel Number: 128, pckh@75.11%MPII.

### Convert keras models to tensorflow forzen pb  
- Convert keras models to tf frozen pb 
```
python3 tools/keras_to_tfpb.py --input_model_json ./models/path/to/network/json --input_model_weights
./models/path/to/network/weight/h5 --out_tfpb ./models/hg_s2_b1_tf.pb
```

### Use OpenVino Model Optimizer to convert tf pb to IR. 
*  For CPU, please use mobile version `hg_s2_b1_mobile` and FP32 
```
~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py -w ./models/hg_s2_b1_tf.pb --input_shape [1,256,256,3] --data_type FP32 --output_dir ./models/ --model_name hg_s2_mobile
```
*  For NCS2(Myriad), please use tiny version `hg_s2_b1_tiny` and FP16 
```
~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py -w ./models/hg_s2_b1_tf.pb --input_shape [1,192,128,3] --data_type FP16 --output_dir ./models/ --model_name hg_s2_tiny
```
* `.xml` and `.bin` will be generated.

## Run demo
- Run single image demo on CPU
```sh
cd src
python3 stacked_hourglass.py -i ../models/sample.jpg -m ../models/hg_s2_mobile.xml -d CPU -l /path/to/cpu/extension/library
```
- Run single image demo on NCS2(MYRIAD)
```sh
cd src
python3 stacked_hourglass.py -i ../models/sample.jpg -m ../models/hg_s2_tiny.xml -d MYRIAD
```

- Run Aysnc demo with camera input on CPU
```sh
cd src
python3 stacked_hourglass_camera_async.py -i cam -m ../models/hg_s2_mobile.xml -d CPU -l /path/to/cpu/extension/library
```

## Reference 
- OpenVino: https://github.com/opencv/dldt 
- OpenCV OpenModelZoo: https://github.com/opencv/open_model_zoo 
- Keras implementation for stacked hourglass: https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras  
