
/*
 * @FilePath     : /sam_test/src/utils_2.cu
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2025-12-02 13:07:36
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2025-12-11 08:56:50
 * Copyright (c) 2025 by G, All Rights Reserved.
 */
#include "utils.h"
#include "clip_bpe.h"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>



// Functor must have external linkage (file scope) to be used in device kernels.
struct SigmoidOp
{
    __host__ __device__ float operator()(float x) const
    {
        return 1.f / (1.f + expf(-x));
    }
};


__global__ void preprocess_kernel(const uint8_t* src, float* dst, int width, int height, float mean, float std)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int r_idx = idx * 3;
        int g_idx = idx * 3 + 1;
        int b_idx = idx * 3 + 2;

        float r = (float)src[r_idx] / 255.0f;
        float g = (float)src[g_idx] / 255.0f;
        float b = (float)src[b_idx] / 255.0f;

        dst[idx] = (r - mean) / std;
        dst[total_pixels + idx] = (g - mean) / std;
        dst[2 * total_pixels + idx] = (b - mean) / std;
    }
}

std::shared_ptr<float> sam_preprocess(const cv::Mat& img, int target_width, int target_height, const float mean, float std)
{
    // 1. Resize on CPU (OpenCV is efficient enough for this on Jetson, or use VPI for further optimization)
    cv::Mat canvas;
    cv::cvtColor(img, canvas, cv::COLOR_BGR2RGB);
    cv::resize(canvas, canvas, cv::Size(target_width, target_height), cv::INTER_LINEAR);

    // 2. Allocate host memory for the result
    std::shared_ptr<float> inBlob(new float[3 * target_width * target_height], [](float* s) { delete[] s; });

    // 3. Use CUDA for normalization and layout change (HWC -> CHW)
    float* d_dst;
    uint8_t* d_src;
    size_t src_size = target_width * target_height * 3 * sizeof(uint8_t);
    size_t dst_size = target_width * target_height * 3 * sizeof(float);

    cudaMalloc(&d_src, src_size);
    cudaMalloc(&d_dst, dst_size);

    cudaMemcpy(d_src, canvas.data, src_size, cudaMemcpyHostToDevice);

    int total_pixels = target_width * target_height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    preprocess_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, target_width, target_height, mean, std);

    cudaMemcpy(inBlob.get(), d_dst, dst_size, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);

    return inBlob;
}



std::vector<char> load_engine(const std::string& path)
{
    std::ifstream in_file(path, std::ios::in | std::ios::binary);
    if (!in_file.is_open())
        return {};
    in_file.seekg(0, std::ios::end);
    size_t            length = in_file.tellg();
    std::vector<char> data;
    if (length > 0) {
        in_file.seekg(0, std::ios::beg);
        data.resize(length);
        in_file.read(data.data(), length);
    }
    in_file.close();
    return data;
}
static MyLogger gLogger;

void infer(std::string engine_path, std::string img_path, std::string prompt)
{
    GpuTimer timer;
    timer.start();
    bool didInitPlugins = initLibNvInferPlugins(&gLogger, "");

    /* 1. 反序列化 engine */
    auto                                         engine_buf = load_engine(engine_path);
    std::shared_ptr<nvinfer1::IRuntime>          runtime(nvinfer1::createInferRuntime(gLogger), [](nvinfer1::IRuntime* s) { s->destroy(); });
    std::shared_ptr<nvinfer1::ICudaEngine>       engine(runtime->deserializeCudaEngine(engine_buf.data(), engine_buf.size()), [](nvinfer1::ICudaEngine* s) { s->destroy(); });
    std::shared_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext(), [](nvinfer1::IExecutionContext* s) { s->destroy(); });
    /* 2. 获取维度 */
    nvinfer1::Dims pix_dims    = engine->getBindingDimensions(0);   // 1,3,1008,1008
    nvinfer1::Dims ids_dims    = engine->getBindingDimensions(1);   // 1,32
    nvinfer1::Dims mask_dims   = engine->getBindingDimensions(2);   // 1,32
    nvinfer1::Dims logits_dims = engine->getBindingDimensions(3);   // 1,200,288,288
    nvinfer1::Dims boxes_dims  = engine->getBindingDimensions(4);   // 1,200,4
    nvinfer1::Dims masks_dims  = engine->getBindingDimensions(5);   // 1,200
    const int      img_c       = pix_dims.d[1];                     // pix_dims  3*1080*1008
    const int      img_h       = pix_dims.d[2];
    const int      img_w       = pix_dims.d[3];
    const int      seq_len     = ids_dims.d[1];     // ids_dims  32
    const int      num_inst    = masks_dims.d[1];   // masks_dims 200*288*288
    const int      mask_h      = masks_dims.d[2];
    const int      mask_w      = masks_dims.d[3];

    // std::cout << "---- binding list ----" << std::endl;
    // for (int i = 0; i < engine->getNbBindings(); ++i)
    //     std::cout << i << "  " << engine->getBindingName(i) << "  " << (engine->bindingIsInput(i) ? "IN" : "OUT") << std::endl;

    // 只给输入绑定设维度，输出一律不动
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i))   // 关键判断
        {
            auto dims = engine->getBindingDimensions(i);
            bool ok   = ctx->setBindingDimensions(i, dims);
            if (!ok) {
                std::cerr << "setBindingDimensions failed on input #" << i << " name=" << engine->getBindingName(i) << std::endl;
                return;
            }
        }
    }
    timer.stop();
    std::cout << "初始化引擎时间: " << timer.elapsed_millis() << " ms" << std::endl;

    /* 3. CPU 前处理 → 得到 float RGB [0,1] 已归一化 */
    cv::Mat     img          = cv::imread(img_path);
    std::string input_prompt = prompt;
    std::locale::global(std::locale("en_US.UTF-8"));
    Tokenizer tok("model/vocab.json", "model/merges.txt", 32);
    auto [token_ids, mask] = tok.encode_with_mask(prompt);
    timer.start();
    std::shared_ptr<float> pix_host = sam_preprocess(img, 1008, 1008, 0.5, 0.5);

    /* 4. 用 thrust::device_vector 代替 cudaMalloc/cudaMemcpy --------- */
    thrust::device_vector<float>   dImage(pix_host.get(), pix_host.get() + img_c * img_h * img_w);   // img
    thrust::device_vector<int64_t> dIds(token_ids.begin(), token_ids.end());                         // token ids
    thrust::device_vector<int64_t> dAttn(mask.begin(), mask.end());                                  // attention mask
    thrust::device_vector<float>   dLogits(num_inst);                                                // output
    thrust::device_vector<float>   dBoxes(num_inst * 4);
    thrust::device_vector<float>   dMasks(num_inst * mask_h * mask_w);

    /* 5. 组装 bindings（取原始指针） */
    std::vector<void*> bindings = {thrust::raw_pointer_cast(dImage.data()),
                                   thrust::raw_pointer_cast(dIds.data()),
                                   thrust::raw_pointer_cast(dAttn.data()),
                                   thrust::raw_pointer_cast(dLogits.data()),
                                   thrust::raw_pointer_cast(dBoxes.data()),
                                   thrust::raw_pointer_cast(dMasks.data())};
    timer.stop();
    std::cout << "前处理时间: " << timer.elapsed_millis() << " ms" << std::endl;

    /* 6. 推理 */
    timer.start();
    bool ok = ctx->enqueueV2(bindings.data(), 0, nullptr);
    if (!ok) {
        std::cerr << "enqueueV2 failed!" << std::endl;
        return;
    }
    timer.stop();
    std::cout << "推理时间: " << timer.elapsed_millis() << " ms" << std::endl;


    /* 8. 获取输出 */
    timer.start();
    // std::cout << ">>> post-proc start"
    //         << "  logits=" << dLogits.size() << "  masks=" << dMasks.size() << "  boxes=" << dBoxes.size() << "  mask_h=" << mask_h << "  mask_w=" << mask_w << "  img_wh=" << img.cols << "x" << img.rows << std::endl;


    //后处理
    thrust::transform(dLogits.begin(), dLogits.end(), dLogits.begin(), [] __device__(float x) { return 1.f / (1.f + expf(-x)); });
    /* --- 3. 生成 0..n-1 索引 ------------------------------ */
    const int n = dLogits.size();
    thrust::device_vector<int> d_idx(n);
    thrust::sequence(d_idx.begin(), d_idx.end());
    thrust::sort_by_key(dLogits.begin(), dLogits.end(), d_idx.begin(), thrust::greater<float>()); //dLogits 排序后,顺序已经被改变, d_idx 记录排序前的索引
    //打印排序后的顺序
    thrust::host_vector<int> h_idx = d_idx;
    // std::cout << "d_idx[0..9] = ";
    // for (int i = 0; i < 10; ++i) std::cout << h_idx[i] << ' ';
    // std::cout << std::endl;

    // std::cout << "top-2 sigmoid scores: "
    //       << dLogits[h_idx[0]] << ' ' << dLogits[h_idx[1]] << ' '
    //       << dLogits[0] << ' ' << dLogits[1] << std::endl;

    thrust::device_vector<int> d_keep(n);
    //lamba 表达式需要  add_cuflags("--extended-lambda")
    auto end_it = thrust::copy_if(d_idx.begin(), d_idx.end(), d_keep.begin(), [logits = dLogits.data().get()] __device__(int idx) { return logits[idx] >= 0.5f; });
    int keep_num = end_it - d_keep.begin();
    d_keep.resize(keep_num);

    /* --- 6. 把保留索引拷回 Host（唯一 H2D 拷贝）----------- */
    std::vector<int> h_keep(keep_num);
    thrust::copy(d_keep.begin(), d_keep.end(), h_keep.begin());
    // std::cout << "keep_num=" << keep_num << std::endl;
    // std::cout << "h_keep[0]=" << h_keep[0] << std::endl;
    // std::cout << "h_keep[1]=" << h_keep[1] << std::endl;


    /* --- 7. 检查 dMasks 和 dBoxes 的大小和内容 ------------- */
    // std::cout << "dMasks.size() = " << dMasks.size() << std::endl;
    // std::cout << "num_inst * mask_h * mask_w = " << num_inst * mask_h * mask_w << std::endl;
    // std::cout << "dBoxes  raw ptr = " << thrust::raw_pointer_cast(dBoxes.data()) << std::endl;
    // std::cout << "dMasks  raw ptr = " << thrust::raw_pointer_cast(dMasks.data()) << std::endl;
    // std::cout << "dLogits raw ptr = " << thrust::raw_pointer_cast(dLogits.data()) << std::endl;

    // float *mp = thrust::raw_pointer_cast(dMasks.data());
    // std::cout << "dMasks[0..3] = " << mp[0] << ' ' << mp[1] << ' ' << mp[2] << ' ' << mp[3] << std::endl;

    /* --- 8. Host 端画框 / 掩码 ----------------------------- */
    cv::Mat dis = img.clone();
    thrust::host_vector<float> h_boxes  = dBoxes;
    thrust::host_vector<float> h_masks  = dMasks;
    // float *mp = thrust::raw_pointer_cast(h_masks.data());
    // std::cout << "h_masks[0..3] = " << mp[0] << ' ' << mp[1] << ' ' << mp[2] << ' ' << mp[3] << std::endl;

    for (int k = 0; k < keep_num; ++k) {
         int idx = h_idx[k];
        /* 越界保护 */
        if (idx < 0 || idx >= num_inst) {
            std::cerr << "idx out of range! idx=" << idx << " num_inst=" << num_inst << std::endl;
            continue;
        }

        float* mask_ptr = thrust::raw_pointer_cast(h_masks.data()) + idx * mask_h * mask_w;
        cv::Mat c_mask_mat(mask_h, mask_w, CV_32FC1, mask_ptr);
        cv::resize(c_mask_mat, c_mask_mat, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_NEAREST);
        cv::threshold(c_mask_mat, c_mask_mat, 0.5, 1, cv::THRESH_BINARY);
        c_mask_mat.convertTo(c_mask_mat, CV_8UC1, 255.0);
        float x1 = h_boxes[idx * 4 + 0] * img.cols;
        float y1 = h_boxes[idx * 4 + 1] * img.rows;
        float x2 = h_boxes[idx * 4 + 2] * img.cols;
        float y2 = h_boxes[idx * 4 + 3] * img.rows;
        /* 关键：先保证 x2≥x1, y2≥y1，再 clamp */
        x1 = std::max(0.f, std::min(x1, float(img.cols)));
        x2 = std::max(0.f, std::min(x2, float(img.cols)));
        y1 = std::max(0.f, std::min(y1, float(img.rows)));
        y2 = std::max(0.f, std::min(y2, float(img.rows)));
        /* 如果网络给出反框，交换保证合法 */
        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);
        /* 再收缩到图像范围内 */
        x2 = std::max(0.f, std::min(x2, float(img.cols - 1)));
        y2 = std::max(0.f, std::min(y2, float(img.rows - 1)));

        cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
        // if (k == 0)
        // std::cout << "  1st box raw: "
        //           << dBoxes[idx * 4 + 0] << ',' << dBoxes[idx * 4 + 1] << ','
        //           << dBoxes[idx * 4 + 2] << ',' << dBoxes[idx * 4 + 3]
        //           << "  clamped: " << box << std::endl;
        std::cout << "box: " << box << "\nscore: " << dLogits[k] << std::endl;
        cv::rectangle(dis, box, cv::Scalar(0, 0, 255), 2, 8);
    }
    timer.stop();
    std::cout << "后处理时间: " << timer.elapsed_millis() << " ms" << std::endl;
    std::string filename = "./cpp_res.jpg";
    cv::imwrite(filename, dis);
}
