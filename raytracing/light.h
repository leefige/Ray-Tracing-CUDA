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

    int GetSample() { return sample; }
    Color GetColor() { return color; }
    Light* GetNext() { return next; }
    void SetNext( Light* light ) { next = light; }

    virtual bool IsPointLight() = 0;
    virtual void Input( std::string , std::stringstream& );
    virtual Vector3 GetO() = 0;
    virtual float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) = 0;
    virtual Primitive* CreateLightPrimitive() = 0;

protected:
    virtual Vector3 GetRandPointLight(const Vector3& crashPoint) = 0;
};

class PointLight : public Light
{
    Vector3 O;
public:
    PointLight() : Light() {}
    ~PointLight() {}

    bool IsPointLight() { return true; }
    Vector3 GetO() { return O; }
    void Input( std::string , std::stringstream& );
    float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality );
    Primitive* CreateLightPrimitive() { return nullptr; }

protected:
    virtual Vector3 GetRandPointLight(const Vector3& crashPoint) { return O; }
};

class SquareLight : public Light
{
    Vector3 O;
    Vector3 Dx, Dy;
public:
    SquareLight() : Light() {}
    ~SquareLight() {}

    bool IsPointLight() { return false; }
    Vector3 GetO() { return O; }
    void Input( std::string , std::stringstream& );
    float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality );
    Primitive* CreateLightPrimitive();

protected:
    virtual Vector3 GetRandPointLight(const Vector3& crashPoint);
};

class SphereLight : public Light
{
    Vector3 O;
    float R;
public:
    SphereLight() : Light(), R(0) {}
    ~SphereLight() {}

    bool IsPointLight() { return false; }
    Vector3 GetO() { return O; }
    void Input( std::string , std::stringstream& );
    float CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality );
    Primitive* CreateLightPrimitive();

protected:
    virtual Vector3 GetRandPointLight(const Vector3& crashPoint);
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

float PointLight::CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) {
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

float SquareLight::CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) {
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

Primitive* SquareLight::CreateLightPrimitive()
{
    PlaneAreaLightPrimitive* res = new PlaneAreaLightPrimitive(O, Dx, Dy, color);
    lightPrimitive = res;
    return res;
}

Vector3 SquareLight::GetRandPointLight(const Vector3& crashPoint)
{
    return O + Dx * (2 * ran() - 1) + Dy * (2 * ran() - 1);
}

void SphereLight::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    if ( var == "R=" ) fin>>R;
    Light::Input( var , fin );
}

// ==========================================

float SphereLight::CalnShade( Vector3 C , Primitive* primitive_head , int shade_quality ) {
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

Primitive* SphereLight::CreateLightPrimitive()
{
    SphereLightPrimitive* res = new SphereLightPrimitive(O, R, color);
    lightPrimitive = res;
    return res;
}

Vector3 SphereLight::GetRandPointLight(const Vector3& crashPoint)
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
