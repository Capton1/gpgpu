#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<vector>
#include<algorithm>
#include "process.hpp"


void average_pooling(cv::Mat src, cv::Mat &dst, int pool_size) {
    cv::Size s = src.size();
    int patchs_y = s.height/pool_size;
    int patchs_x = s.width/pool_size;
    dst.create(cv::Size(patchs_x, patchs_y), CV_8U);

    for (int ii = 0; ii < patchs_y; ii++) {
        for (int jj = 0; jj < patchs_x; jj++) {
            unsigned int sum = 0;
            int index_y = ii * pool_size;
            int index_x = jj * pool_size;
            for (int i = index_y; i < index_y+pool_size; i++) {
                for (int j = index_x; j < index_x+pool_size; j++) {
                    sum += src.at<uint8_t>(i, j);
                }
            }
            dst.at<uint8_t>(ii, jj) = sum/(pool_size*pool_size);
        }
    }
}

void post_processing(cv::Mat src, cv::Mat &dst, int postproc_size) {
    cv::Mat se(cv::Size(postproc_size, postproc_size), CV_8U, cv::Scalar(0));
    for (int i = postproc_size/2 - 1; i < postproc_size/2 + 2; i++) {
        for (int j = 0; j < postproc_size; j++) {
            se.at<uint8_t>(i, j) = 1;
        }
    }

    morphologyEx(src, dst, cv::MORPH_CLOSE, se);
}

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

    int width = img.size().width;
    int height = img.size().height;
    int stride = width;

    std::vector<uint8_t> img_vec;
    if (!img.isContinuous()) {
        std::cout << "Could not onvert img to array" << std::endl;
    }
    img_vec.assign(img.data, img.data + img.total()*img.channels());

    unsigned char *buffer = img_vec.data();

    uint8_t *sobel_x = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    sobel_filter(buffer, sobel_x, width, height, stride * sizeof(uint8_t), 'x');
    cv::Mat sobelx = cv::Mat(height, width, CV_8U, sobel_x);
    cv::imwrite("../output_gpu/sobelx.jpg", sobelx);

    uint8_t *sobel_y = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    sobel_filter(buffer, sobel_y, width, height, stride * sizeof(uint8_t), 'y');
    cv::Mat sobely = cv::Mat(height, width, CV_8U, sobel_y);
    cv::imwrite("../output_gpu/sobely.jpg", sobely);

    int pool_size = 31;
    int patchs_y = height/pool_size;
    int patchs_x = width/pool_size;
    std::cout << patchs_x << " : " << patchs_y << std::endl;
    uint8_t *resp = (uint8_t*)calloc(patchs_x * patchs_y, sizeof(uint8_t));
    average_pooling(sobel_x, sobel_y, resp, width, height, stride * sizeof(uint8_t), pool_size);
    cv::Mat resp_out = cv::Mat(patchs_y, patchs_x, CV_8U, resp);
    cv::imwrite("../output_gpu/response.jpg", resp_out);


    int postproc_size = 5;
    cv::Mat resp_postproc;
    post_processing(resp_out, resp_postproc, postproc_size);
    cv::imwrite("../output_gpu/resp_postproc.jpg", resp_postproc);
}