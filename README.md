# Fast_Stacked_Hourglass_Network_OpenVino
A fast stacked hourglass network for human pose estimation on OpenVino

## Installation
- Install OpenVino 2018 R4 or above
- Install python dependencies
```
pip3 -r install requirements.txt  
```

## How to convert pre-trained models
- Download pre-trained model hourglass from [Google drive](https://drive.google.com/drive/folders/12ioJONmse658qc9fgMpzSy2D_JCdkFVg?usp=sharing) and save them to `models`. You are going to download two files, one is json file for network configuration while another is weight.

- Convert keras models to tensorflow forzen pb
```
python3 tools/keras_to_tfpb.py --input_model_json ./models/net_arch.json --input_model_weights
./models/weights_epoch96.h5 --out_tfpb ./models/hg_s2_b1_tf.pb
```
- Use OpenVino Model Optimizer to convert tf pb to IR. `hg_s2_b1_tf.xml` and `hg_s2_b1_tf.bin` will be generated under current folder.
```
~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py -w ./models/hg_s2_b1_tf.pb --input_shape=[1,256,256,3]
```

## Run demo
- Run single image demo on CPU
```
cd src
python3 stacked_hourglass.py -i ../models/sample.jpg -m ../models/hg_s2_b1_tf.xml -d CPU -l ~/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so
```
