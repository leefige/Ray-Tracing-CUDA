#ifndef CG_DEFS_H_
#define CG_DEFS_H_

#include <cstdlib>

namespace cg
{

/* random double in [0, 1) */
inline double preciseRan()
{
    return double( rand() % RAND_MAX ) / RAND_MAX;
}

inline double ran()
{
    return double( rand() % 32768 ) / 32768;
}

constexpr double EPS = 1e-6;
constexpr double PI = 3.1415926535897932384626;
constexpr double BIG_DIST = 1e100;

constexpr double STD_LENS_WIDTH = 0.88; //the width of lens in the scene
constexpr double STD_LENS_HEIGHT = 0.88;

constexpr int STD_IMAGE_WIDTH = 420;
constexpr int STD_IMAGE_HEIGHT = 420;
constexpr int STD_SHADE_QUALITY = 4;    //caln shade :: how many times will be run (*16)
constexpr int STD_DREFL_QUALITY = 4;    //caln drefl :: how many times will be run (*16)
constexpr int STD_MAX_PHOTONS = 2000000;
constexpr int STD_EMIT_PHOTONS = 1000000;
constexpr int STD_SAMPLE_PHOTONS = 100;
constexpr double STD_SAMPLE_DIST = 1;

} /* namespace cg */

#endif /* CG_DEFS_H_ */
