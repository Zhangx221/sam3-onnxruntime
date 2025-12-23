#include "utils.h"
#include "clip_bpe.h"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

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
    // 1. 在 CPU 上调整大小（OpenCV 在 Jetson 上效率足够高，或者使用 VPI 进行进一步优化）
    cv::Mat canvas;
    cv::cvtColor(img, canvas, cv::COLOR_BGR2RGB);
    cv::resize(canvas, canvas, cv::Size(target_width, target_height), cv::INTER_LINEAR);

    // 2. 为结果分配主机内存
    std::shared_ptr<float> inBlob(new float[3 * target_width * target_height], [](float* s) { delete[] s; });

    // 3. 使用 CUDA 进行归一化和布局更改（HWC -> CHW）
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

// 封装的前处理函数
void perform_preprocess(
    const std::string& img_path,
    std::string prompt,
    int img_h,
    int img_w,
    cv::Mat& img,
    thrust::device_vector<float>& dImage,
    thrust::device_vector<int64_t>& dIds,
    thrust::device_vector<int64_t>& dAttn)
{
    img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "加载图像失败: " << img_path << std::endl;
        return;
    }

    std::locale::global(std::locale("en_US.UTF-8"));
    Tokenizer tok("model/vocab.json", "model/merges.txt", 32);
    auto [token_ids, mask] = tok.encode_with_mask(prompt);

    std::shared_ptr<float> pix_host = sam_preprocess(img, img_w, img_h, 0.5, 0.5);

    // 初始化设备向量
    dImage = thrust::device_vector<float>(pix_host.get(), pix_host.get() + 3 * img_h * img_w);
    dIds   = thrust::device_vector<int64_t>(token_ids.begin(), token_ids.end());
    dAttn  = thrust::device_vector<int64_t>(mask.begin(), mask.end());
}

// 封装的后处理函数
void perform_postprocess(
    const cv::Mat& img,
    thrust::device_vector<float>& dLogits,
    thrust::device_vector<float>& dBoxes,
    thrust::device_vector<float>& dMasks,
    int num_inst,
    int mask_h,
    int mask_w)
{
    // 应用 sigmoid
    thrust::transform(dLogits.begin(), dLogits.end(), dLogits.begin(), SigmoidOp());

    // 生成索引 0..n-1
    const int n = dLogits.size();
    thrust::device_vector<int> d_idx(n);
    thrust::sequence(d_idx.begin(), d_idx.end());

    // 对 logits 进行降序排序并重新排列索引
    thrust::sort_by_key(dLogits.begin(), dLogits.end(), d_idx.begin(), thrust::greater<float>());
    
    thrust::host_vector<int> h_idx = d_idx;

    // 按阈值 0.5 过滤
    thrust::device_vector<int> d_keep(n);
    auto end_it = thrust::copy_if(d_idx.begin(), d_idx.end(), d_keep.begin(), 
        [logits = dLogits.data().get()] __device__(int idx) { return logits[idx] >= 0.5f; });
    
    int keep_num = end_it - d_keep.begin();
    d_keep.resize(keep_num);

    // 将保留的索引复制到主机
    std::vector<int> h_keep(keep_num);
    thrust::copy(d_keep.begin(), d_keep.end(), h_keep.begin());

    // 主机端可视化
    cv::Mat dis = img.clone();
    thrust::host_vector<float> h_boxes = dBoxes;
    thrust::host_vector<float> h_masks = dMasks;

    for (int k = 0; k < keep_num; ++k) {
        int idx = h_idx[k];
        if (idx < 0 || idx >= num_inst) {
            std::cerr << "索引超出范围! idx=" << idx << " num_inst=" << num_inst << std::endl;
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

        // 截断并确保框有效
        x1 = std::max(0.f, std::min(x1, float(img.cols)));
        x2 = std::max(0.f, std::min(x2, float(img.cols)));
        y1 = std::max(0.f, std::min(y1, float(img.rows)));
        y2 = std::max(0.f, std::min(y2, float(img.rows)));

        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);

        x2 = std::max(0.f, std::min(x2, float(img.cols - 1)));
        y2 = std::max(0.f, std::min(y2, float(img.rows - 1)));

        cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
        std::cout << "box: " << box << "\nscore: " << dLogits[k] << std::endl;
        cv::rectangle(dis, box, cv::Scalar(0, 0, 255), 2, 8);
    }

    std::string filename = "./cpp_res.jpg";
    cv::imwrite(filename, dis);
}

void infer(std::string engine_path, std::string img_path, std::string prompt)
{
    GpuTimer timer;
    timer.start();
    bool didInitPlugins = initLibNvInferPlugins(&gLogger, "");

    /* 1. 反序列化引擎 */
    auto                                         engine_buf = load_engine(engine_path);
    std::shared_ptr<nvinfer1::IRuntime>          runtime(nvinfer1::createInferRuntime(gLogger), [](nvinfer1::IRuntime* s) { s->destroy(); });
    std::shared_ptr<nvinfer1::ICudaEngine>       engine(runtime->deserializeCudaEngine(engine_buf.data(), engine_buf.size()), [](nvinfer1::ICudaEngine* s) { s->destroy(); });
    std::shared_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext(), [](nvinfer1::IExecutionContext* s) { s->destroy(); });

    /* 2. 获取维度 */
    nvinfer1::Dims pix_dims    = engine->getBindingDimensions(0);   // 1,3,1008,1008
    // nvinfer1::Dims ids_dims    = engine->getBindingDimensions(1);   // 1,32
    // nvinfer1::Dims mask_dims   = engine->getBindingDimensions(2);   // 1,32
    // nvinfer1::Dims logits_dims = engine->getBindingDimensions(3);   // 1,200,288,288
    // nvinfer1::Dims boxes_dims  = engine->getBindingDimensions(4);   // 1,200,4
    nvinfer1::Dims masks_dims  = engine->getBindingDimensions(5);   // 1,200,288,288
    
    const int      img_h       = pix_dims.d[2];
    const int      img_w       = pix_dims.d[3];
    const int      num_inst    = masks_dims.d[1];   // 200
    const int      mask_h      = masks_dims.d[2];
    const int      mask_w      = masks_dims.d[3];

    // 仅为输入绑定设置维度
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i)) {
            auto dims = engine->getBindingDimensions(i);
            if (!ctx->setBindingDimensions(i, dims)) {
                std::cerr << "设置绑定维度失败，输入 #" << i << " 名称=" << engine->getBindingName(i) << std::endl;
                return;
            }
        }
    }
    timer.stop();
    std::cout << "引擎初始化时间: " << timer.elapsed_millis() << " ms" << std::endl;

    /* 3. 前处理 */
    timer.start();
    cv::Mat img;
    thrust::device_vector<float>   dImage;
    thrust::device_vector<int64_t> dIds;
    thrust::device_vector<int64_t> dAttn;

    perform_preprocess(img_path, prompt, img_h, img_w, img, dImage, dIds, dAttn);
    
    // 分配输出缓冲区
    thrust::device_vector<float> dLogits(num_inst);
    thrust::device_vector<float> dBoxes(num_inst * 4);
    thrust::device_vector<float> dMasks(num_inst * mask_h * mask_w);

    /* 4. 绑定 */
    std::vector<void*> bindings = {thrust::raw_pointer_cast(dImage.data()),
                                   thrust::raw_pointer_cast(dIds.data()),
                                   thrust::raw_pointer_cast(dAttn.data()),
                                   thrust::raw_pointer_cast(dLogits.data()),
                                   thrust::raw_pointer_cast(dBoxes.data()),
                                   thrust::raw_pointer_cast(dMasks.data())};
    timer.stop();
    std::cout << "前处理时间: " << timer.elapsed_millis() << " ms" << std::endl;

    /* 5. 推理 */
    timer.start();
    if (!ctx->enqueueV2(bindings.data(), 0, nullptr)) {
        std::cerr << "enqueueV2 失败!" << std::endl;
        return;
    }
    timer.stop();
    std::cout << "推理时间: " << timer.elapsed_millis() << " ms" << std::endl;

    /* 6. 后处理 */
    timer.start();
    perform_postprocess(img, dLogits, dBoxes, dMasks, num_inst, mask_h, mask_w);
    timer.stop();
    std::cout << "后处理时间: " << timer.elapsed_millis() << " ms" << std::endl;
}
