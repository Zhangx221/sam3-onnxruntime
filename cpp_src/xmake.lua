-- Environment Configuration
-- Adjust these paths according to your server environment
local trt_include_dir   = "/home/greatek/zhangx/package/TensorRT-8.6.1.6/include"
local trt_lib_dir       = "/home/greatek/zhangx/package/TensorRT-8.6.1.6/lib"
local cudnn_include_dir = "/home/greatek/zhangx/package/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/include"
local cudnn_lib_dir     = "/home/greatek/zhangx/package/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib"
local system_lib_dir    = "/usr/lib"

add_rules("mode.debug", "mode.release")

set_project("sam_test")
set_version("0.0.1")

set_languages("c++17")

-- Dependencies
add_requires("nlohmann_json",{system = false})
add_requires("opencv",{system = false})


target("sam_test")
    set_kind("binary")
    set_rundir("$(projectdir)")
    add_rules("cuda")
    
    add_packages("nlohmann_json")
    add_packages("opencv")
    
    -- Sources
    add_files("main.cpp", "clip_bpe.cpp", "utils.cu")
    
    -- CUDA specific flags
    add_cuflags("--extended-lambda")

    -- Jetson system paths for TensorRT and cuDNN
    add_includedirs(trt_include_dir)
    add_includedirs(cudnn_include_dir)
    
    add_linkdirs(trt_lib_dir)
    add_linkdirs(cudnn_lib_dir)
    add_linkdirs(system_lib_dir)
    
    -- TensorRT & cuDNN libs
    add_links("nvinfer", "nvinfer_plugin", "nvparsers", "nvonnxparser", "cudnn")
