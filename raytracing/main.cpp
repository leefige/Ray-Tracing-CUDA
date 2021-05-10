#include <iostream>
#include <omp.h>

#ifdef OLD_CXX
#include <experimental/filesystem>    /* C++17 is required */
#else
#include <filesystem>    /* C++17 is required */
#endif

#include"raytracer.h"

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

    Raytracer* raytracer = new Raytracer;
    raytracer->SetInput(inputFile);
    raytracer->SetOutput(outputFile);

#ifdef _OPENMP
    //omp_set_num_threads(12);
    omp_set_num_threads(raytracer->GetH());
#endif

    //raytracer->Run();
    raytracer->MultiThreadRun();
    //raytracer->DebugRun(740,760,410,430);

    std::cout << "Output file saved at '" << fs::absolute(outputPath) << "'." << std::endl;
    return 0;
}
