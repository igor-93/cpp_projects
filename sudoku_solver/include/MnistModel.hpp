#ifndef MNISTMODEL_HPP
#define MNISTMODEL_HPP

#include <iostream>
#include <functional>
#include <queue>
#include <vector>

#include <torch/torch.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

class MnistModel{

    public:
        MnistModel();
        void testLibTorch();
        void trainModel();
        std::vector<std::pair<int, float> > inferClass(const cv::Mat& digit);

        static cv::Mat convertImg(torch::Tensor input);
        static torch::Tensor convertImg(cv::Mat input);

        static const std::string dataPath;
        static const std::string modelPath;
        static const int trainBatchSize;
        static const int testBatchSize;
        static const int numberOfEpochs;
        static const int logInterval;
        static const int nClasses;

        static const float dataMean;
        static const float dataStd;

        static const float acceptanceThreshold;

    private:
        torch::Device device = torch::Device(c10::DeviceType::CPU);
        torch::nn::Sequential net;
        bool readyForInference;

        torch::nn::Sequential getModel();

        template <typename DataLoader>
        void trainEpoch(int32_t epoch, torch::nn::Sequential& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size);
        template <typename DataLoader>
        void test(torch::nn::Sequential& model, DataLoader& data_loader, size_t dataset_size);

};

#endif