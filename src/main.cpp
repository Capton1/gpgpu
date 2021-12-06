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

    int orig_width = img.size().width;
    int orig_height = img.size().height;


    int output_width = orig_width/POOLSIZE;
    int output_height = orig_height/POOLSIZE;

    std::vector<uint8_t> img_vec;
    if (!img.isContinuous()) {
        std::cout << "Could not convert img to array" << std::endl;
    }
    img_vec.assign(img.data, img.data + img.total()*img.channels());

    unsigned char *buffer = img_vec.data();


    uint8_t *output = (uint8_t*)calloc(orig_width * orig_height, sizeof(uint8_t));
    process_image(buffer, output, orig_width, orig_height);
    cv::Mat output_img = cv::Mat(orig_height, orig_width, CV_8U, output);
    cv::imwrite("../output_gpu/output.jpg", output_img);
    free(output);

    return 0;
}