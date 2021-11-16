#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>


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

    //cv::Mat abs_grad_x, abs_grad_y;
    //cv::convertScaleAbs(sobx1, abs_grad_x);
    //cv::convertScaleAbs(soby1, abs_grad_y);

    //printf("%d %d\n", sobx1.at<uint8_t>(0,0), sobx1.at<uint8_t>(1478, 2116));

    int pool_size = 31;
    cv::Size s = sobx1.size();
    int patchs_y = s.height/pool_size;
    int patchs_x = s.width/pool_size;
    std::cout << patchs_y << " " << patchs_x << std::endl;
    cv::Mat response(cv::Size(patchs_y, patchs_x), CV_64FC1, cv::Scalar(0));

    

    // crop = grad_img[:pool_size*patchs_y, :pool_size*patchs_x]
    // features_patchs = view_as_blocks(crop, block_shape=(pool_size, pool_size))
    for (int ii = 0; ii < patchs_y; ii++) {
        for (int jj = 0; jj < patchs_x; jj++) {
            unsigned int sum = 0;
            int index_y = ii * pool_size;
            int index_x = jj * pool_size;
            for (int i = index_y; i < index_y+pool_size; i++) {
                for (int j = index_x; j < index_x+pool_size; j++) {
                    sum += sobx1.at<uint8_t>(i, j);
                }
            }
            float avg = (1.0*sum)/(pool_size*pool_size);
            //std::cout << avg << std::endl;
            response.at<float>(ii, jj) = avg;
        }
    }
    
    cv::imwrite("../output/pool_sobx1.jpg", response);
}
