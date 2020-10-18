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
        static MnistModel& getInstance(){
            static MnistModel instance; // Guaranteed to be destroyed and instantiated on first use.
            return instance;
        }

        MnistModel(MnistModel const&) = delete;
        void operator=(MnistModel const&) = delete;

        void testLibTorch();
        void trainModel();
        std::vector<std::pair<int, float> > inferClass(const cv::Mat& digit);

        static cv::Mat convertImg(torch::Tensor input);
        static torch::Tensor convertImg(const cv::Mat& input);

        constexpr static const float acceptanceThreshold = 0.8f;

    private:
        MnistModel();

        const int trainBatchSize = 64;
        const int testBatchSize = 512;
        const int numberOfEpochs = 10;
        const int logInterval= 10;
        const int nClasses = 10;

        const float dataMean = 0.1307f;
        const float dataStd = 0.3081f;

        const std::string dataPath = "./mnist";
        const std::string modelPath = "./model.pt";
        torch::Device device = torch::Device(c10::DeviceType::CPU);
        torch::nn::Sequential net;
        bool readyForInference;

        torch::nn::Sequential getModel();

        template <typename DataLoader>
        void trainEpoch(int32_t epoch, torch::nn::Sequential& model, DataLoader& data_loader,
                        torch::optim::Optimizer& optimizer, size_t dataset_size);
        template <typename DataLoader>
        void test(torch::nn::Sequential& model, DataLoader& data_loader, size_t dataset_size);

};

#endif