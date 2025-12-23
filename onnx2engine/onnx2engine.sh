# export LD_LIBRARY_PATH=/home/greatek/zhangx/package/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
# /home/greatek/zhangx/package/TensorRT-8.6.1.6/bin/trtexec --onnx=models/onnx_detect_modify/sam3.onnx --saveEngine=models/onnx_detect_modify/sam3_fp16.engine --verbose --fp16


export LD_LIBRARY_PATH=/home/greatek/zhangx/package/TensorRT-10.8.0.43/lib:$LD_LIBRARY_PATH
/home/greatek/zhangx/package/TensorRT-10.8.0.43/bin/trtexec --onnx=models/onnx_detect_modify/sam3.onnx --saveEngine=models/onnx_detect_modify/sam3_fp16_tensorrt10.engine --verbose --fp16