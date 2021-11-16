#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>


void gpool(cv::Mat src, cv::Mat &dst, int pool_size) {
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

    cv::Mat img_grayscale = cv::imread("../train/PXL_20211101_175643604.jpg", 0);

    // Check for failure
    if (img_grayscale.empty()) 
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::imwrite("../output/grayscale.jpg", img_grayscale);


    // SobelX
    cv::Mat sobx1;
    cv::Sobel(img_grayscale, sobx1, -1, 1, 0);
    cv::imwrite("../output/sobelx.jpg", sobx1);

    // SobelY
    cv::Mat soby1;
    cv::Sobel(img_grayscale, soby1, -1, 0, 1);
    cv::imwrite("../output/sobely.jpg", soby1);

    int pool_size = 31;
    cv::Mat pool_sobx1;
    gpool(sobx1, pool_sobx1, pool_size);
    cv::imwrite("../output/pool_sobx1.jpg", pool_sobx1);

    cv::Mat pool_soby1;
    gpool(soby1, pool_soby1, pool_size);
    cv::imwrite("../output/pool_soby1.jpg", pool_soby1);

    // since all pixels are unsigned, values < 0 are replaced by 0
    cv::Mat resp1 = pool_sobx1 - pool_soby1;
    cv::imwrite("../output/resp1.jpg", resp1);

    int postproc_size = 5;
    cv::Mat se(cv::Size(postproc_size, postproc_size), CV_8U, cv::Scalar(0));
    for (int i = postproc_size/2 - 1; i < postproc_size/2 + 2; i++) {
        for (int j = 0; j < postproc_size; j++) {
            se.at<uint8_t>(i, j) = 1;
        }
    }

    cv::Mat resp1_postproc;
    morphologyEx(resp1, resp1_postproc, cv::MORPH_CLOSE, se);
    cv::imwrite("../output/resp1_postproc.jpg", resp1_postproc);

    double minVal;
    double maxVal;
    cv::minMaxLoc(resp1_postproc, &minVal, &maxVal);
    std::cout << 0.5*maxVal <<std::endl;
    cv::Mat output;
    cv::threshold(resp1_postproc, output, maxVal/2, 255, cv::THRESH_BINARY);
    cv::imwrite("../output/output.jpg", output);
}
