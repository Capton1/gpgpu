#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<vector>
#include<algorithm>
#include "process.hpp"


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
    
    cv::Mat img = cv::imread("../test.png", 0);

    int width = img.size().width;
    int height = img.size().height;
    int stride = width;

    std::vector<uint8_t> img_vec;
    if (!img.isContinuous()) {
        std::cout << "Could not convert img to array" << std::endl;
    }
    img_vec.assign(img.data, img.data + img.total()*img.channels());

    unsigned char *buffer = img_vec.data();

    /*uint8_t *sobel_x = (uint8_t*)calloc(width * height, sizeof(uint8_t));
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
    uint8_t *resp = (uint8_t*)calloc(patchs_x * patchs_y, sizeof(uint8_t));
    average_pooling(sobel_x, sobel_y, resp, width, height, stride * sizeof(uint8_t), pool_size);
    cv::Mat resp_out = cv::Mat(patchs_y, patchs_x, CV_8U, resp);
    cv::imwrite("../output_gpu/response.jpg", resp_out);*/


    //int postproc_size = 5;
    //cv::Mat resp_postproc;
    //post_processing(resp_out, resp_postproc, postproc_size);
    int patchs_x = width;
    int patchs_y = height;
    uint8_t *post_proc = (uint8_t*)calloc(patchs_x * patchs_y, sizeof(uint8_t));
    morph_closure(buffer, post_proc, patchs_x, patchs_y, patchs_x * sizeof(uint8_t));
    cv::Mat resp_postproc = cv::Mat(patchs_y, patchs_x, CV_8U, post_proc);
    cv::imwrite("../output_gpu/resp_postproc.jpg", resp_postproc);

    /*std::vector<uint8_t> resp_vec;
    if (!resp_postproc.isContinuous()) {
        std::cout << "Could not convert img to array" << std::endl;
    }
    resp_vec.assign(resp_postproc.data, resp_postproc.data + resp_postproc.total()*resp_postproc.channels());
    buffer = resp_vec.data();

    uint8_t *output = (uint8_t*)calloc(patchs_x * patchs_y, sizeof(uint8_t));
    threshold(buffer, output, patchs_x, patchs_y, patchs_x * sizeof(uint8_t));
    cv::Mat output_img = cv::Mat(patchs_y, patchs_x, CV_8U, output);
    cv::imwrite("../output_gpu/output.jpg", output_img);*/

    /*free(sobel_x);
    free(sobel_y);
    free(resp);*/
    free(post_proc);
    //free(output);
}