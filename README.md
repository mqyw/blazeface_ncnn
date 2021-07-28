#介绍
移植pytorch的blazeface到ncnn框架下使用c++运行

##环境

ncnn
windows10
Visual Studio 2019 Community

##编译
使用vs2019建立自己的工程添加库连接，include连接，导入src下的main和face_det.cpp
然后编译运行即可

##Test
bin目录下的exe可查看相应效果

## 引用

https://github.com/Tencent/ncnn
https://github.com/hollance/BlazeFace-PyTorch
https://github.com/google/mediapipe（mediapipe/mediapipe/calculators/util/non_max_suppression_calculator.cc）
