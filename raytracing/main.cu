#include <iostream>
#ifdef OLD_CXX
#include <experimental/filesystem>    /* C++17 is required */
#else
#include <filesystem>    /* C++17 is required */
#endif

#include <omp.h>

// #include "raytracer.h"
#include "camera.h"
#include "color.h"
#include "bmp.h"
#include "render.h"

#include "cutils.h"

#ifdef OLD_CXX
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

using namespace cg;

constexpr int TX = 8;
constexpr int TY = 8;

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

#ifndef TESTING
    Raytracer* raytracer = new Raytracer;
    raytracer->SetInput(inputFile);
    raytracer->SetOutput(outputFile);

    raytracer->CreateAll();
    int H = raytracer->GetH(), W = raytracer->GetW();
#else
    int H = STD_IMAGE_HEIGHT, W = STD_IMAGE_WIDTH;
#endif

    // Render our buffer
    int blockX = int(ceil(float(H) / TX));
    int blockY = int(ceil(float(W) / TY));
    dim3 blocks(blockX, blockY);
    dim3 threads(TX, TY);

    std::cout << "blocks: (" << blockX << "," << blockY << ")" << std::endl;

#ifndef TESTING
    Render<<<blocks, threads>>>(*raytracer, H, W);
#else
    Camera* camera;
    checkCudaErrors(cudaMallocManaged((void **)&camera, sizeof(Camera)));
    camera->HostInit();

    Color* fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, sizeof(Color) * H * W));

    camera->DeviceInit(fb);

    Render<<<blocks, threads>>>(camera, H, W);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#endif


// #ifdef _OPENMP
//     omp_set_num_threads(raytracer->GetH());
// #pragma omp parallel for
// #endif
//     for (int i = 0; i < H; i++) {
//         for (int j = 0; j < W; j++) {
//             Render(*raytracer, i, j);
//         }
//     }

#ifndef TESTING
    raytracer->Write();
#else
    Bmp bmp;
    bmp.Initialize(H , W);
    (camera)->Output(&bmp);
    bmp.Output(outputFile);
#endif

    std::cout << "Output file saved at '" << fs::absolute(outputPath) << "'." << std::endl;
    return 0;
}
