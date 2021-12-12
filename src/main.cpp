#include <CLI/CLI.hpp>
#include <iostream>
#include <stdio.h>
//#include "cpu.hpp"
#include <FreeImagePlus.h>

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;

    std::string filename = "../collective_database/test.png";

    
    std::string mode = "CPU";

    CLI::App app{"code"};
    app.add_option("-i", filename, "Input image");
    app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");
    CLI11_PARSE(app, argc, argv);

    const char *cstr = filename.c_str();

    // Rendering
    if (mode == "CPU") {
        printf("CPU\n");

        fipImage img;
        img.load(cstr);
        std::cout << img.getWidth() << " " << img.getHeight() << std::endl;
        std::cout << img.isGrayscale() << std::endl;
        std::cout << img.getBitsPerPixel() << std::endl;
        //process_cpu(cstr);
    }
    else if (mode == "GPU") {
        printf("GPU\n");
    }

    
    // Evaluate
    int pool_size = 31;

    //Image* output = read_png("../collective_database/output.png");
    //Image* scaled_output = image_scaler(output, pool_size+1);
    //write_png(scaled_output, "../collective_database/scaled_output.png");

    //Image* gt = read_png("../collective_database/test-GT.png");
    
    //float iou = compute_IoU(gt, scaled_output);
    //printf("Metrics: IoU: %f\n", iou);
}
