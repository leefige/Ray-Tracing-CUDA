#include <iostream>
#ifdef OLD_CXX
#include <experimental/filesystem>    /* C++17 is required */
#else
#include <filesystem>    /* C++17 is required */
#endif

#include <omp.h>

#include "raytracer.h"
#include "render.h"

#ifdef OLD_CXX
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

using namespace cg;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << std::endl << argv[0] << " <input_file.txt> <output_file.bmp>" << std::endl;
        exit(1);
    }

    const char* const inputFile = argv[1];
    const char* const outputFile = argv[2];

    // check file exist
    fs::path inputPath = inputFile;
    if (!(fs::exists(inputPath) && fs::is_regular_file(inputPath))) {
        std::cerr << "Error: input file does not exist: '" << fs::absolute(inputPath) << "'." << std::endl;
        exit(-1);
    }

    // check dir exist
    fs::path outputPath = outputFile;
    auto parentPath = outputPath.parent_path();
    if (!(fs::exists(parentPath) && fs::is_directory(parentPath))) {
        if (!fs::create_directories(parentPath)) {
            std::cerr << "Error: cannot create output directory '" << fs::absolute(parentPath) << "'." << std::endl;
            exit(-2);
        }
    }

    int tx = 8;
    int ty = 8;

    Raytracer* raytracer = new Raytracer;
    raytracer->SetInput(inputFile);
    raytracer->SetOutput(outputFile);

    raytracer->CreateAll();
    int H = raytracer->GetH(), W = raytracer->GetW();

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef _OPENMP
    omp_set_num_threads(raytracer->GetH());
#pragma omp parallel for
#endif
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            Render(*raytracer, i, j);
        }
    }

    raytracer->Write();
    std::cout << "Output file saved at '" << fs::absolute(outputPath) << "'." << std::endl;
    return 0;
}
