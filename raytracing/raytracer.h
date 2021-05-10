#ifndef CG_RAYTRACER_H_
#define CG_RAYTRACER_H_

#include <cstdlib>

#include <iostream>
#include <thread>
#include <string>
#include <vector>

#include <omp.h>

#include "ray.h"
#include "scene.h"
#include "bmp.h"

namespace cg
{

constexpr double SPEC_POWER = 20;
constexpr int MAX_DREFL_DEP = 2;
constexpr int MAX_RAYTRACING_DEP = 10;
constexpr int HASH_FAC = 7;
constexpr int HASH_MOD = 10000007;
constexpr int NUM_RESAMPLE = 3;

constexpr int TRACER_REFLE_BIT = 1;
constexpr int TRACER_REFRA_BIT = 2;

class Raytracer {
    std::string input , output;
    Scene scene;
    Light* light_head;
    Color background_color_top;
    Color background_color_bottom;
    Camera* camera;

    Color CalnDiffusion(const CollidePrimitive& collide_primitive);
    bool CalnReflection(const CollidePrimitive& collide_primitive, const Vector3& in_V, Ray& ray);
    bool CalnRefraction(const CollidePrimitive& collide_primitive, const Vector3& in_V, Ray& ray);

    Color GetBackgroundColor(const Vector3& V) const
    {
        auto direction = V.GetUnitVector();
        double t = 0.5 * (direction.z + 1.0);
        return background_color_bottom * (1.0 - t) + background_color_top * t;
    }

public:
    Raytracer();
    ~Raytracer() {}

    int TraceRay(Ray& in, Ray& refl, Ray& refr);

    void SetInput( std::string file ) { input = file; }
    void SetOutput( std::string file ) { output = file; }

    void CreateAll();
    Primitive* CreateAndLinkLightPrimitive(Primitive* primitive_head);

    int GetH() const { return camera->GetH(); }
    int GetW() const { return camera->GetW(); }

    Camera* GetCamera() { return camera; }

    void Write() const;
};

// =========================================================

Raytracer::Raytracer() {
    light_head = NULL;
    background_color_top = Color();
    background_color_bottom = Color();
    camera = new Camera;
}

int Raytracer::TraceRay(Ray& in, Ray& refl, Ray& refr)
{
    in.visited = true;
    if (in.depth > MAX_RAYTRACING_DEP) {
        return 0;
    }

    int ret = 0;

    CollidePrimitive collide_primitive = scene.FindNearestPrimitiveGetCollide(in.O , in.V);

    if (collide_primitive.isCollide) {
        Primitive* primitive = collide_primitive.collide_primitive;
        // light: end of tracing
        if (primitive->IsLightPrimitive()) {
            in.myColor += primitive->GetMaterial()->color;
            //std::cout<<primitive->GetMaterial()->color.r;
        }
        // more rays!
        else {
            if (primitive->GetMaterial()->diff > EPS || primitive->GetMaterial()->spec > EPS) {
                in.myColor += CalnDiffusion(collide_primitive);
            }
            if (primitive->GetMaterial()->refl > EPS) {
                if (CalnReflection(collide_primitive, in.V, refl)) {
                    ret |= TRACER_REFLE_BIT;
                }
            }
            if (primitive->GetMaterial()->refr > EPS) {
                if (CalnRefraction(collide_primitive, in.V, refr)) {
                    ret |= TRACER_REFRA_BIT;
                }
            }
        }
    }
    // nothing there, just background
    else {
        in.myColor += GetBackgroundColor(in.V);
    }

    return ret;
}

Color Raytracer::CalnDiffusion(const CollidePrimitive& collide_primitive)
{
    Primitive* primitive = collide_primitive.collide_primitive;
    Color color = primitive->GetMaterial()->color;

    // use texture if any
    if (primitive->GetMaterial()->texture != NULL) {
        color = color * collide_primitive.GetTexture();
    }

    // TODO: is this necessary???
    //Color ret = color * background_color * primitive->GetMaterial()->diff;
    Color ret;

    for ( Light* light = light_head ; light != NULL ; light = light->GetNext() ) {
        double shade = light->CalnShade(collide_primitive.C, scene.GetPrimitiveHead(), int(16 * camera->GetShadeQuality()));
        if (shade < EPS) {
            continue;
        }

        // now the light produces some shade on this diffuse point

        // R: light ray from diffuse point to the light
        Vector3 R = ( light->GetO() - collide_primitive.C ).GetUnitVector();
        double dot = R.Dot( collide_primitive.N );
        if (dot > EPS) {
            // diffuse light
            if ( primitive->GetMaterial()->diff > EPS ) {
                double diff = primitive->GetMaterial()->diff * dot * shade;
                ret += color * light->GetColor() * diff;
            }
            // specular light
            if ( primitive->GetMaterial()->spec > EPS ) {
                double spec = primitive->GetMaterial()->spec * pow( dot , SPEC_POWER ) * shade;
                ret += color * light->GetColor() * spec;
            }
        }
    }

    return ret;
}

bool Raytracer::CalnReflection(const CollidePrimitive& collide_primitive, const Vector3& in_V, Ray& ray)
{
    Vector3 ray_V = in_V.Reflect( collide_primitive.N );
    Primitive* primitive = collide_primitive.collide_primitive;

    // only reflection
    if (primitive->GetMaterial()->drefl < EPS || ray.depth > MAX_DREFL_DEP) {
        ray.O = collide_primitive.C;
        ray.V = ray_V;
        ray.attenuation = primitive->GetMaterial()->color * primitive->GetMaterial()->refl;
        return true;
    }
    // diffuse reflection (fuzzy reflection)
    else {
        // TODO: NEED TO IMPLEMENT

        // Unit *circle* perpendicular to ray_V.
        // This is different from sampling from a unit sphere -- when projecting the sphere
        // to this circle the points are not uniformly distributed.
        // However, considering the ExpBlur, this approximation may be justified.
        auto baseX = ray_V.GetAnVerticalVector();
        auto baseY = (ray_V * baseX).GetUnitVector();

        // scale the circle according to drefl (fuzzy) value
        baseX *= primitive->GetMaterial()->drefl;
        baseY *= primitive->GetMaterial()->drefl;

        // Color diffReflected;
        // int numSamples = int(floor(16 * camera->GetDreflQuality()));
        // for (int i = 0; i < numSamples; i++) {
        //     // ADD BLUR
        //     auto xy = primitive->GetMaterial()->blur->GetXY();
        //     auto& x = xy.first;
        //     auto& y = xy.second;
        //     auto fuzzy_V = ray_V + baseX * x + baseY * y;
        //     diffReflected += RayTracing(collide_primitive.C, fuzzy_V, dep + 1, hash);
        // }

        // ADD BLUR
        auto xy = primitive->GetMaterial()->blur->GetXY();
        auto& x = xy.first;
        auto& y = xy.second;
        auto fuzzy_V = ray_V + baseX * x + baseY * y;

        ray.O = collide_primitive.C;
        ray.V = fuzzy_V;
        ray.attenuation = primitive->GetMaterial()->color * primitive->GetMaterial()->refl;
        return true;
    }
}

bool Raytracer::CalnRefraction(const CollidePrimitive& collide_primitive, const Vector3& in_V, Ray& ray)
{
    Primitive* primitive = collide_primitive.collide_primitive;
    double n = primitive->GetMaterial()->rindex;
    if (collide_primitive.front) {
        n = 1 / n;
    }

    Vector3 ray_V = in_V.Refract( collide_primitive.N , n );

    if (collide_primitive.front) {
        ray.attenuation = Color(1.0, 1.0, 1.0) * primitive->GetMaterial()->refr;
    } else {
        Color absor = primitive->GetMaterial()->absor * -collide_primitive.dist;
        Color trans = Color( exp( absor.r ) , exp( absor.g ) , exp( absor.b ) );
        ray.attenuation = trans * primitive->GetMaterial()->refr;
    }

    ray.O = collide_primitive.C;
    ray.V = ray_V;

    return true;
}

Primitive* Raytracer::CreateAndLinkLightPrimitive(Primitive* primitive_head)
{
    Light* light_iter = light_head;
    while(light_iter != NULL)
    {
        Primitive* new_primitive = light_iter->CreateLightPrimitive();
        if ( new_primitive != NULL ) {
            new_primitive->SetNext( primitive_head );
            primitive_head = new_primitive;
        }
        light_iter = light_iter->GetNext();
    }
    return primitive_head;
}

void Raytracer::CreateAll()
{
    srand( 1995 - 05 - 12 );
    std::cout << input << std::endl;
    std::ifstream fin( input );

    std::string obj;
    Primitive* primitive_head = nullptr;
    while ( fin >> obj ) {
        if (obj[0] == '#') {
            continue;
        }

        // create a primitive if necessary
        Primitive* new_primitive = nullptr;
        Light* new_light = nullptr;
        if ( obj == "primitive" ) {
            std::string type; fin >> type;
            if ( type == "sphere" ) new_primitive = new Sphere;
            if ( type == "plane" ) new_primitive = new Plane;
            if ( type == "square" ) new_primitive = new Square;
            if ( type == "cylinder" ) new_primitive = new Cylinder;
            if ( type == "bezier" ) new_primitive = new Bezier;

            if ( new_primitive != nullptr ) {
                new_primitive->SetNext( primitive_head );
                primitive_head = new_primitive;
            }
        }
        else if ( obj == "light" ) {
            std::string type; fin >> type;
            if ( type == "point" ) new_light = new PointLight;
            if ( type == "square" ) new_light = new SquareLight;
            if ( type == "sphere" ) new_light = new SphereLight;

            if ( new_light != nullptr ) {
                new_light->SetNext( light_head );
                light_head = new_light;
            }
        }
        else if ( obj != "background" && obj != "camera" ) continue;

        fin.ignore( 1024 , '\n' );

        // read config for this primitive
        std::string order;
        while ( getline( fin , order , '\n' ) ) {
            std::stringstream fin2( order );
            std::string var; fin2 >> var;

            if (var[0] == '#') {
                continue;
            }

            if (var == "end") {
                break;
            }

            // parse configs
            if (obj == "background") {
                if (var == "color=") {
                    background_color_top.Input(fin2);
                    background_color_bottom = background_color_top;
                } else if (var == "top=") {
                    background_color_top.Input(fin2);
                    background_color_bottom = Color(1.0, 1.0, 1.0);
                } else if (var == "bottom=") {
                    background_color_bottom.Input(fin2);
                }
            }
            if ( obj == "primitive" && new_primitive != nullptr) new_primitive->Input( var , fin2 );
            if ( obj == "light" && new_light != nullptr) new_light->Input( var , fin2 );
            if ( obj == "camera" ) camera->Input( var , fin2 );
        }
    }

    scene.CreateScene(CreateAndLinkLightPrimitive(primitive_head));
    camera->Initialize();
}

void Raytracer::Write() const
{
    int H = camera->GetH() , W = camera->GetW();
    Bmp* bmp = new Bmp( H , W );
    camera->Output( bmp );
    bmp->Output( output );
    delete bmp;
}


} /* namespace cg */

#endif /* CG_RAYTRACER_H_ */
