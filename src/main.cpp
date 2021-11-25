#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<vector>
#include<algorithm>
#include "process.hpp"

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;

    std::string filename = "output.png";
    std::string mode = "GPU";

    CLI::App app{"code"};
    app.add_option("-o", filename, "Output image");
    app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");
    CLI11_PARSE(app, argc, argv);

    // Rendering
    if (mode == "CPU")
    {
        printf("CPU\n");
    }
    else if (mode == "GPU")
    {
        printf("GPU\n");
    }
    
    cv::Mat img = cv::imread("../train/PXL_20211101_175643604.jpg", 0);

    std::vector<uint8_t> img_vec;
    if (!img.isContinuous()) {
        std::cout << "Could not onvert img to array" << std::endl;
    }
    img_vec.assign(img.data, img.data + img.total()*img.channels());

    unsigned char *buffer = img_vec.data();


    float *sobel_x = (float*)calloc(img.rows * img.cols, sizeof(float));

    sobel_filter(buffer, sobel_x, img.cols, img.rows, img.cols);

    std::vector<float> v(sobel_x, sobel_x + img.rows * img.cols);
    float max = *std::max_element(v.begin(), v.end());

    transform(v.begin(), v.end(), v.begin(), [max](float &c){ return 255*c/max; });

    cv::Mat output = cv::Mat(img.rows, img.cols, CV_32FC1, v.data());
    cv::imwrite("../output_gpu/test.jpg", output);
}