#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
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

    sobel_filter(buffer, img.rows, img.cols, img.rows);

    cv::Mat output = cv::Mat(img.rows, img.cols, CV_8U, buffer);
    cv::imwrite("../output_gpu/test.jpg", output);
}