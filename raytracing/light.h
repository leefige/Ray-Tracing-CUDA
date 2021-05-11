#ifndef CG_LIGHT_H_
#define CG_LIGHT_H_

#include <cmath>
#include <cstdlib>

#include <sstream>
#include <string>

#include "vector3.h"
#include "color.h"
#include "primitive.h"

namespace cg
{

class Light
{
protected:
    int sample;
    Color color;
    Light* next;
    Primitive* lightPrimitive;

public:

    Light();
    virtual ~Light() {}

    __device__ int GetSample() { return sample; }
    __device__ Color GetColor() { return color; }
    __device__ Light* GetNext() { return next; }
    __device__ void SetNext( Light* light ) { next = light; }

    __device__ virtual bool IsPointLight() = 0;
    virtual void Input( std::string , std::stringstream& );
    __device__ virtual Vector3 GetO() = 0;
    __device__ virtual float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) = 0;
    __device__ virtual Primitive* CreateLightPrimitive() = 0;

protected:
    __device__ virtual Vector3 GetRandPointLight(const Vector3& crashPoint) = 0;
};

class PointLight : public Light
{
    Vector3 O;
public:
    PointLight() : Light() {}
    ~PointLight() {}

    __device__ bool IsPointLight() { return true; }
    __device__ Vector3 GetO() { return O; }
    void Input( std::string , std::stringstream& );
    __device__ float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality );
    __device__ Primitive* CreateLightPrimitive() { return nullptr; }

protected:
    __device__ virtual Vector3 GetRandPointLight(const Vector3& crashPoint) { return O; }
};

class SquareLight : public Light
{
    Vector3 O;
    Vector3 Dx, Dy;
public:
    SquareLight() : Light() {}
    ~SquareLight() {}

    __device__ bool IsPointLight() { return false; }
    __device__ Vector3 GetO() { return O; }
    void Input( std::string , std::stringstream& );
    __device__ float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality );
    __device__ Primitive* CreateLightPrimitive();

protected:
    __device__ virtual Vector3 GetRandPointLight(const Vector3& crashPoint);
};

class SphereLight : public Light
{
    Vector3 O;
    float R;
public:
    SphereLight() : Light(), R(0) {}
    ~SphereLight() {}

    __device__ bool IsPointLight() { return false; }
    __device__ Vector3 GetO() { return O; }
    void Input( std::string , std::stringstream& );
    __device__ float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality );
    __device__ Primitive* CreateLightPrimitive();

protected:
    __device__ __device__ virtual Vector3 GetRandPointLight(const Vector3& crashPoint);
};

// =======================================================

Light::Light() {
    sample = rand();
    next = nullptr;
    lightPrimitive = nullptr;
}

void Light::Input( std::string var , std::stringstream& fin ) {
    if ( var == "color=" ) color.Input( fin );
}

// ==========================================

void PointLight::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    Light::Input( var , fin );
}

__device__ float PointLight::CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) {
    /* For point light, shade_quality is of no use: we don't need to sample. */

    // light ray from diffuse point to light source
    Vector3 V = O - C;
    float dist = V.Module();

    // if light ray collide any object, light source produce no shade to diffuse light
    for (Primitive* now = primitive_head ; now != NULL ; now = now->GetNext()) {
        CollidePrimitive tmp = now->Collide(C, V);
        if (dist - tmp.dist > EPS) {
            return 0;
        }
    }

    return 1;
}

// ==========================================

void SquareLight::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    if ( var == "Dx=" ) Dx.Input( fin );
    if ( var == "Dy=" ) Dy.Input( fin );
    Light::Input( var , fin );
}

__device__ float SquareLight::CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) {
    int shade = 0;

    // TODO: NEED TO IMPLEMENT
    for (int i = 0; i < shade_quality; i++) {
        // sample a point light from light primitive
        Vector3 randO = GetRandPointLight(C);

        // light ray from diffuse point to point light
        Vector3 V = randO - C;
        float dist = V.Module();

        int addShade = 1;
        // if light ray collide any object before reaching the light, point light produce no shade to diffuse light
        for (Primitive* now = primitive_head; now != NULL; now = now->GetNext()) {
            // don't collide with myself!
            // this shouldn't have been a trouble, but there is COMPUTATIONAL ERROR!
            if (now == lightPrimitive) {
                continue;
            }
            CollidePrimitive tmp = now->Collide(C, V);
            if (dist - tmp.dist > EPS) {
                addShade = 0;
                break;
            }
        }
        shade += addShade;
    }
    return float(shade) / shade_quality;
}

__device__ Primitive* SquareLight::CreateLightPrimitive()
{
    PlaneAreaLightPrimitive* res = new PlaneAreaLightPrimitive(O, Dx, Dy, color);
    lightPrimitive = res;
    return res;
}

__device__ Vector3 SquareLight::GetRandPointLight(const Vector3& crashPoint)
{
    return O + Dx * (2 * ran() - 1) + Dy * (2 * ran() - 1);
}

void SphereLight::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    if ( var == "R=" ) fin>>R;
    Light::Input( var , fin );
}

// ==========================================

__device__ float SphereLight::CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) {
    int shade = 0;

    // TODO: NEED TO IMPLEMENT
    for (int i = 0; i < shade_quality; i++) {
        // sample a point light from light primitive
        Vector3 randO = GetRandPointLight(C);

        // light ray from diffuse point to point light
        Vector3 V = randO - C;
        float dist = V.Module();

        int addShade = 1;
        // if light ray collide any object before reaching the light, point light produce no shade to diffuse light
        for (Primitive* now = primitive_head; now != NULL; now = now->GetNext()) {
            // don't collide with myself!
            // this shouldn't have been a trouble, but there is COMPUTATIONAL ERROR!
            // so sometimes a light blocks itself, especially when the radius is small...
            if (now == lightPrimitive) {
                continue;
            }
            CollidePrimitive tmp = now->Collide(C, V);
            if (dist - tmp.dist > EPS) {
                addShade = 0;
                break;
            }
        }
        shade += addShade;
    }
    return float(shade) / shade_quality;
}

__device__ Primitive* SphereLight::CreateLightPrimitive()
{
    SphereLightPrimitive* res = new SphereLightPrimitive(O, R, color);
    lightPrimitive = res;
    return res;
}

__device__ Vector3 SphereLight::GetRandPointLight(const Vector3& crashPoint)
{
    Vector3 toCrash = crashPoint - O;
    Vector3 radiusV = toCrash.GetUnitVector() * R;
    Vector3 axisV = radiusV.GetAnVerticalVector();

    float maxTheta = acos(R / toCrash.Module());
    // theta in [-PI/2, PI/2)
    float theta = (ran() * 2 - 1) * PI / 2;
    theta *= maxTheta / (PI / 2);

    // phi in [0, 2*PI)
    float phi = ran() * 2 * PI;
    Vector3 ret = radiusV.Rotate(axisV, theta).Rotate(radiusV, phi);
    return O + ret;
}

} /* namespace cg */

#endif /* CG_LIGHT_H_ */
