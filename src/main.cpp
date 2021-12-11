#include <CLI/CLI.hpp>
#include <iostream>
#include <stdio.h>
#include "cpu.hpp"


int main(int argc, char** argv) {
    (void) argc;
    (void) argv;

    std::string filename = "output.png";
    std::string str = "../collective_database/test.png";
    const char *cstr = str.c_str();
    
    std::string mode = "CPU";

    CLI::App app{"code"};
    app.add_option("-o", filename, "Output image");
    app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");
    CLI11_PARSE(app, argc, argv);

    // Rendering
    if (mode == "CPU") {
        printf("CPU\n");
        process_cpu(cstr);
    }
    else if (mode == "GPU") {
        printf("GPU\n");
    }
    
    // Evaluate
    int pool_size = 31;

    Image* output = read_png("../collective_database/output.png");
    Image* scaled_output = image_scaler(output, pool_size+1);
    write_png(scaled_output, "../collective_database/scaled_output.png");

    Image* gt = read_png("../collective_database/test-GT.png");
    
    float iou = compute_IoU(gt, scaled_output);
    printf("Metrics: IoU: %f\n", iou);
}
