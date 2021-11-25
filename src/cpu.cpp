#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>


cv::Mat get_grayscale(const std::string img_path) {
    cv::Mat img_grayscale = cv::imread(img_path, 0);

    // Check for failure
    if (img_grayscale.empty()) {
        throw std::runtime_error("Could not open or find the image: " + img_path);
    }

    return img_grayscale;
}

void get_sobel(cv::Mat img, char type, cv::Mat &dst) {
    if (type == 'x') {
        cv::Sobel(img, dst, -1, 1, 0);
    }
    else if (type == 'y') {
        cv::Sobel(img, dst, -1, 0, 1);
    }
    else {
        throw std::runtime_error("Invalid type");
    }
}

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

void threshold(cv::Mat src, cv::Mat &dst) {
    double maxVal;
    cv::minMaxLoc(src, nullptr, &maxVal);
    cv::threshold(src, dst, maxVal/2, 255, cv::THRESH_BINARY);
}

void process_cpu(const std::string img_path) {
    cv::Mat img_grayscale = get_grayscale(img_path);
    cv::imwrite("../output1/grayscale.jpg", img_grayscale);

    // SobelX
    cv::Mat sobx1;
    get_sobel(img_grayscale, 'x', sobx1);
    cv::imwrite("../output1/sobelx.jpg", sobx1);

    // SobelY
    cv::Mat soby1;
    get_sobel(img_grayscale, 'y', soby1);
    cv::imwrite("../output1/sobely.jpg", soby1);

    int pool_size = 31;

    cv::Mat pool_sobx1;
    average_pooling(sobx1, pool_sobx1, pool_size);
    cv::imwrite("../output1/pool_sobx1.jpg", pool_sobx1);

    cv::Mat pool_soby1;
    average_pooling(soby1, pool_soby1, pool_size);
    cv::imwrite("../output1/pool_soby1.jpg", pool_soby1);

    // since all pixels are unsigned, values < 0 are replaced by 0
    cv::Mat resp1 = pool_sobx1 - pool_soby1;
    cv::imwrite("../output1/resp1.jpg", resp1);

    int postproc_size = 5;
    cv::Mat resp1_postproc;
    post_processing(resp1, resp1_postproc, postproc_size);
    cv::imwrite("../output1/resp1_postproc.jpg", resp1_postproc);

    cv::Mat output;
    threshold(resp1_postproc, output);
    cv::imwrite("../output1/output.jpg", output);
}