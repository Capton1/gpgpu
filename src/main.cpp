#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>



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
}
