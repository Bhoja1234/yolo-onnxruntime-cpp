#include <iostream>
#include <iomanip>
#include "RunOnnx.h"
#include <filesystem>
#include <fstream>
#include <random>

bool isGPU = false;

void Classifier(DCSP_CORE*& p)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images_cls_orig";
    std::cout << "imgs_path: %s" << imgs_path << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
        {
            std::string img_path = i.path().string();
            //std::cout << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            // 找到res中最大的confidence
            float maxConfidence = -1.0f;
            int predCls = -1;
            for (int i = 0; i < res.size(); i++) {
                if (res.at(i).confidence > maxConfidence) {
                    maxConfidence = res.at(i).confidence;
                    predCls = res.at(i).classId;
                }
            }
            // 输出结果
            std::cout << "predCls: " << predCls << std::endl;
            std::cout << "predConfidence: " << maxConfidence << std::endl;


            float positionY = 50;
            for (int i = 0; i < res.size(); i++)
            {
                int r = dis(gen);
                int g = dis(gen);
                int b = dis(gen);
                cv::putText(img, p->classes[i] + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                positionY += 50;
            }

            cv::imshow("TEST_CLS", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

    }
}

void ClsTest()
{
    try
    {
        DCSP_CORE* yoloDetector = new DCSP_CORE;
        std::string model_path = "./models/yolov8-cls.onnx"; // PCBA模型
        yoloDetector->classes = { "NG", "OK" };
        DL_INIT_PARAM params{ model_path, YOLO_CLS, {640, 640} };
        yoloDetector->CreateSession(params);
        Classifier(yoloDetector);
    }
    catch (const std::exception&)
    {

    }

}

void Detector(DCSP_CORE*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images_dec";

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );

            }
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

void DetectTest()
{
    DCSP_CORE* yoloDetector = new DCSP_CORE;
    //ReadCocoYaml(yoloDetector);
    yoloDetector->classes = { "person" };
    DL_INIT_PARAM params;
    params.ModelPath = "./models/yolov8n.onnx";
    params.rectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;
    params.imgSize = { 640, 640 };
    params.modelType = YOLO_DETECT;

    if (isGPU) {
        // GPU FP32 inference
        params.cudaEnable = true;
    }
    else {
        // CPU inference
        params.cudaEnable = false;
    }

    yoloDetector->CreateSession(params);
    Detector(yoloDetector);
}

void Segment(DCSP_CORE*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images_dec";

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            cv::Mat mask = img.clone();
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );

                mask(re.box).setTo(color, re.boxMask);

            }
            cv::addWeighted(img, 0.5, mask, 0.5, 0, img);
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}


void SegmentTest()
{
    DCSP_CORE* yoloDetector = new DCSP_CORE;
    //ReadCocoYaml(yoloDetector);
    yoloDetector->classes = { "person" };
    DL_INIT_PARAM params;
    params.ModelPath = "./models/yolov8n-seg.onnx";
    params.rectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;
    params.imgSize = { 640, 640 };
    params.modelType = YOLO_SEGMENT;

    if (isGPU) {
        // GPU FP32 inference
        params.cudaEnable = true;
    }
    else {
        // CPU inference
        params.cudaEnable = false;
    }
    yoloDetector->CreateSession(params);
    Segment(yoloDetector);
}

void Keypoint(DCSP_CORE*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images_dec";

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".bmp")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );

                // 绘制点
                for (PoseKeyPoint kp : re.keyPoints) {
                    float conf = kp.confidence;
                    if (conf < 0.7) {
                        continue;
                    }
                    cv::Point2f point(kp.x, kp.y);
                    cv::circle(img, point, 5, cv::Scalar(0, 255, 120), -1);
                }

            }
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

void KeypointTest() {
    DCSP_CORE* yoloPose = new DCSP_CORE;
    //ReadCocoYaml(yoloDetector);
    yoloPose->classes = { "person" };
    DL_INIT_PARAM params;
    params.ModelPath = "./models/yolov8n-pose.onnx";

    params.rectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;
    params.imgSize = { 640, 640 };
    params.modelType = YOLO_POSE;

    if (isGPU) {
        // GPU FP32 inference
        params.cudaEnable = true;
    }
    else {
        // CPU inference
        params.cudaEnable = false;
    }

    yoloPose->CreateSession(params);
    Keypoint(yoloPose);
}

void Unsupervied(DCSP_CORE*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "temp" / "images";

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            std::cout << "img_path: " << img_path << std::endl;
            cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                // 热力图
                cv::Mat result_img = re.boxMask;    
                cv::resize(result_img, result_img, cv::Size(640, 640));
                cv::imshow("Defect Contour", result_img);
                cv::waitKey(0);

            }
            std::cout << "Press any key to exit" << std::endl;
            cv::destroyAllWindows();
        }
    }
}

void UnsuperviedTest() {
    DCSP_CORE* anomalib = new DCSP_CORE;

    DL_INIT_PARAM params;
    params.ModelPath = "./models/wujiandu.onnx";
    params.pixel_threshold = 9.178058624267578; // metadata.json中的信息
    params.min_val = 2.297870054235318e-07; // anomaly_maps.min
    params.max_val = 29.42519187927246; // anomaly_maps.max
    params.imgSize = { 512, 512 };
    params.modelType = ANOMALIB;    

    if (isGPU) {
        // GPU FP32 inference
        params.cudaEnable = true;
    }
    else {
        // CPU inference
        params.cudaEnable = false;
    }

    anomalib->CreateSession(params);
    Unsupervied(anomalib);
}

int main()
{
    // 获取所有的支持设备列表CPU/GPU
    Ort::Env env;
    Ort::Session session(nullptr); // 创建一个空会话
    Ort::SessionOptions sessionOptions{ nullptr };//创建会话配置
    //获取所有的支持设备列表CPU/GPU
    auto providers = Ort::GetAvailableProviders();
    //for (auto& provider : providers) {
    //    std::cout << provider << std::endl;
    //}
    auto cudaAvailable = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
    if (cudaAvailable == providers.end())//没有找到cuda列表
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (cudaAvailable != providers.end())//找到cuda列表
    {
        std::cout << "Inference device: GPU" << std::endl;
        isGPU = true;
    }
    else // 什么也没找到，默认使用CPU
    {
        std::cout << "Inference device: CPU" << std::endl;
    }
    
    //DetectTest();
    //ClsTest();
    SegmentTest();
    //KeypointTest();
    //UnsuperviedTest();

    return 0;
}
