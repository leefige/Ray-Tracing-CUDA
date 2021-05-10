#ifndef CG_CAMERA_H_
#define CG_CAMERA_H_

#include <cstdio>

#include <string>
#include <sstream>
#include <iostream>

#include "defs.h"
#include "vector3.h"
#include "color.h"
#include "bmp.h"

#include "cutils.h"

namespace cg
{

class Camera
{
    Vector3 O , N , Dx , Dy;
    float lens_W , lens_H;
    int W , H;
    Color** data;
    float shade_quality;
    float drefl_quality;
    int max_photons;
    int emit_photons;
    int sample_photons;
    float sample_dist;

public:
    Camera();
    ~Camera();

    Vector3 GetO() { return O; }
    int GetW() { return W; }
    int GetH() { return H; }
    void SetColor( int i , int j , Color color ) { data[i][j] = color; }
    float GetShadeQuality() { return shade_quality; }
    float GetDreflQuality() { return drefl_quality; }
    int GetMaxPhotons() { return max_photons; }
    int GetEmitPhotons() { return emit_photons; }
    int GetSamplePhotons() { return sample_photons; }
    float GetSampleDist() { return sample_dist; }

    Vector3 Emit( float i , float j );
    void Initialize();
    void Input( std::string var , std::stringstream& fin );
    void Output( Bmp* );
};

// ===================================================

Camera::Camera()
{
    O = Vector3( 0 , 0 , 0 );
    N = Vector3( 0 , 1 , 0 );
    lens_W = STD_LENS_WIDTH;
    lens_H = STD_LENS_HEIGHT;
    W = STD_IMAGE_WIDTH;
    H = STD_IMAGE_HEIGHT;
    shade_quality = STD_SHADE_QUALITY;
    drefl_quality = STD_DREFL_QUALITY;
    max_photons = STD_MAX_PHOTONS;
    emit_photons = STD_EMIT_PHOTONS;
    sample_photons = STD_SAMPLE_PHOTONS;
    sample_dist = STD_SAMPLE_DIST;
    data = nullptr;
}

Camera::~Camera()
{
    if (data != nullptr) {
        for (int i = 0; i < H; i++) {
            delete[] data[i];
        }
        delete[] data;
    }
}

void Camera::Initialize()
{
    N = N.GetUnitVector();
    Dx = N.GetAnVerticalVector();
    Dy = Dx * N;
    Dx = Dx * lens_W / 2;
    Dy = Dy * lens_H / 2;

    data = new Color*[H];
    for (int i = 0; i < H; i++) {
        data[i] = new Color[W];
    }
}

Vector3 Camera::Emit( float i , float j )
{
    return N + Dy * ( 2 * i / H - 1 ) + Dx * ( 2 * j / W - 1 );
}

void Camera::Input( std::string var , std::stringstream& fin )
{
    if ( var == "O=" ) O.Input( fin );
    if ( var == "N=" ) {
        N.Input(fin);
        N = N.GetUnitVector();
    }
    if ( var == "lens_W=" ) fin >> lens_W;
    if ( var == "lens_H=" ) fin >> lens_H;
    if ( var == "image_W=" ) fin >> W;
    if ( var == "image_H=" ) fin >> H;
    if ( var == "shade_quality=" ) fin >> shade_quality;
    if ( var == "drefl_quality=" ) fin >> drefl_quality;
    if ( var == "max_photons=" ) fin >> max_photons;
    if ( var == "emit_photons=" ) fin >> emit_photons;
    if ( var == "sample_photons=" ) fin >> sample_photons;
    if ( var == "sample_dist=" ) fin >> sample_dist;
}

void Camera::Output( Bmp* bmp )
{
    bmp->Initialize( H , W );

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            bmp->SetColor(i, j, data[i][j]);
        }
    }
}

} /* namespace cg */

#endif /* CG_CAMERA_H_ */
