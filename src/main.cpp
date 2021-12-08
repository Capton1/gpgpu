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
}
