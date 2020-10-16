#include "MnistModel.hpp"

using namespace std;
using namespace torch;

const string MnistModel::dataPath = "./mnist";
const string MnistModel::modelPath = "./model.pt";
const int MnistModel::trainBatchSize = 64;
const int MnistModel::testBatchSize = 512;
const int MnistModel::numberOfEpochs = 2; // 50
const int MnistModel::logInterval = 10;
const int MnistModel::nClasses = 10;
const float MnistModel::dataMean = 0.1307;
const float MnistModel::dataStd = 0.3081;
const float MnistModel::acceptanceThreshold = 0.8;

MnistModel::MnistModel() {
    if (torch::cuda::is_available()) {
        cout << "CUDA is available! Training on GPU." << endl;
        device = Device(c10::DeviceType::CUDA);
    }

    net = getModel();

    readyForInference = false;
}

nn::Sequential MnistModel::getModel(){
    nn::Sequential discriminator(
        // input is 28x28
        nn::Conv2d(
            nn::Conv2dOptions(1, 64, 5).stride(1).padding(0).bias(false)
        ),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::MaxPool2d(nn::MaxPool2dOptions({2, 2}).stride({2, 2})),
        // Layer 2
        nn::Conv2d(
            nn::Conv2dOptions(64, 128, 3).stride(1).padding(0).bias(false)
        ),
        // nn::BatchNorm2d(128),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::MaxPool2d(nn::MaxPool2dOptions({2, 2}).stride({2, 2})),  // output is 5x5
        // Layer 3
        nn::Flatten(),
        nn::Dropout(nn::DropoutOptions().p(0.5)),
        nn::Linear(25 * 128, 256),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Linear(256, this->nClasses),
        nn::LogSoftmax(nn::LogSoftmaxOptions(1))
    );  
    discriminator->to(device);

    return discriminator;
}

template <typename DataLoader>
void MnistModel::trainEpoch(int32_t epoch, nn::Sequential& model, DataLoader& data_loader, optim::Optimizer& optimizer,
    size_t dataset_size){
    model->train();
    size_t batch_idx = 0;
    for (auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        optimizer.zero_grad();
        Tensor output = model->forward(data);
        Tensor loss = torch::nll_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        if (batch_idx++ % logInterval == 0) {
            printf(
                "\rTrain Epoch: %d [%5ld/%5ld] Loss: %.4f",
                epoch,
                batch_idx * batch.data.size(0),
                dataset_size,
                loss.template item<float>()
            );
        }
    }
}

template <typename DataLoader>
void MnistModel::test(nn::Sequential& model, DataLoader& data_loader, size_t dataset_size) {
    NoGradGuard no_grad;
    model->eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        Tensor output = model->forward(data);
        test_loss += nll_loss(
            output,
            targets,
            /*weight=*/{},
            Reduction::Sum
        ).template item<float>();
        Tensor pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    printf(
        "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
        test_loss,
        static_cast<double>(correct) / dataset_size);
}


void MnistModel::trainModel(){
    float learning_rate = 1e-3;
    float l2_loss = 1e-5;
    optim::Adam optimizer(net->parameters(), optim::AdamOptions(learning_rate).weight_decay(l2_loss));

    auto train_dataset = data::datasets::MNIST(dataPath)
                            .map(data::transforms::Normalize<>(dataMean, dataStd))
                            .map(data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader = data::make_data_loader<data::samplers::SequentialSampler>(
            std::move(train_dataset), trainBatchSize);

    auto test_dataset = data::datasets::MNIST(dataPath, data::datasets::MNIST::Mode::kTest)
                            .map(data::transforms::Normalize<>(dataMean, dataStd))
                            .map(data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
        torch::data::make_data_loader(std::move(test_dataset), testBatchSize);

    for (size_t epoch = 1; epoch <= numberOfEpochs; ++epoch) {
        cout << "Epoch: " << epoch << endl;
        trainEpoch(epoch, net, *train_loader, optimizer, train_dataset_size);
        test(net, *test_loader, test_dataset_size);
    }

    readyForInference = true;
    save(net, modelPath);
    cout << "Saved the model at " << modelPath << endl;
}

cv::Mat MnistModel::convertImg(Tensor input){
    input = input.to(torch::kCPU).squeeze();
    // cout << "sizes: " << input.sizes() << endl;
    // cout << "type: " << input.scalar_type() << endl;
    
    int height = input.sizes()[0];
    int width = input.sizes()[1];

    float* temp_arr = input.data_ptr<float>();
	
    cv::Mat resultImg(height, width, CV_32F);
    memcpy((void *) resultImg.data, temp_arr, sizeof(float) * input.numel());
    return resultImg;
}

Tensor MnistModel::convertImg(cv::Mat input){
    if(input.channels() != 1){
        cout << "image has more than 1 channels: " << input.channels() << endl;
        throw -1;
    }
    // cv::Mat imgFloat;
    // input.convertTo(imgFloat, CV_32FC1, 1.0f / 255.0f);
    int height = input.size().height;
    int width = input.size().width;
    // number of channels, number of images, height, width
    Tensor tensor = torch::from_blob(input.data, {1, 1, height, width});

    // cout << "sizes: " << tensor.sizes() << endl;
    // cout << "type: " << tensor.scalar_type() << endl;
    return tensor;
}


vector<pair<int, float> > MnistModel::inferClass(const cv::Mat& digit){
    if(!readyForInference){
        cout << "Loading the saved model from " << modelPath << " ...";
        load(net, modelPath);
        cout << " done." << endl;
        readyForInference = true;
    }
    net->eval();

    // reshape
    cv::Mat resized;
    cv::resize(digit, resized, cv::Size(28, 28));

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC1); 
    floatImg = floatImg / 255.0;
 
    // normalize
    floatImg = (floatImg - dataMean) / dataStd;

    Tensor netInput = convertImg(floatImg).to(device);;
    Tensor output = torch::exp(net->forward(netInput)).to(kCPU);
    float* probs = output.data_ptr<float>();

    // get top 3 most likely digits
    auto cmp = [](pair<int, float> left, pair<int, float> right) { return left.second < right.second; };
    priority_queue<pair<int, float>, vector<pair<int, float> >, decltype(cmp)> classWithProb(cmp);
    for(int i=0; i<nClasses; i++){
        classWithProb.push(make_pair(i, probs[i]));
    }

    pair<int, float> top_class = classWithProb.top();
    classWithProb.pop();
    pair<int, float> snd_class = classWithProb.top();
    classWithProb.pop();
    pair<int, float> trd_class = classWithProb.top();

    vector<pair<int, float> > res = {top_class, snd_class, trd_class}; 

    return res;
} 


void MnistModel::testLibTorch(){
    auto train_dataset = data::datasets::MNIST(dataPath);

    for(int i=0; i<1; i++){
        data::Example<> example = train_dataset.get(i);
        Tensor sampleImg = example.data;
        int sampleLabel = example.target.template item<int>();

        cv::Mat cvImg = convertImg(sampleImg);
        Tensor tensorImg = convertImg(cvImg);
        cout << tensorImg << endl;
        std::stringstream sstm;
        sstm << i << " Label: " << sampleLabel;

        cv::namedWindow(sstm.str(), cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
        cv::imshow(sstm.str(), cvImg);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}

