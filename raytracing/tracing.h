#ifndef CG_TRACING_H_
#define CG_TRACING_H_

#include <vector>

#include "color.h"
#include "ray.h"
#include "vector3.h"
#include "raytracer.h"

namespace cg
{

constexpr int REFLE_BIT = 1;
constexpr int REFRA_BIT = 2;


int TraceRay(Raytracer& tracer, Ray& in, Ray& refl, Ray& refr)
{
    in.visited = true;
    if (in.depth > MAX_RAYTRACING_DEP) {
        return 0;
    }

    int ret = 0;

    CollidePrimitive collide_primitive = tracer.GetScene().FindNearestPrimitiveGetCollide(in.O , in.V);

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
                in.myColor += tracer.CalnDiffusion(collide_primitive);
            }
            /*if (primitive->GetMaterial()->refl > EPS) {
                if (tracer.CalnReflection(collide_primitive, in.V, refl)) {
                    ret |= REFLE_BIT;
                }
            }
            if (primitive->GetMaterial()->refr > EPS) {
                if (tracer.CalnRefraction(collide_primitive, in.V, refr)) {
                    ret |= REFRA_BIT;
                }
            }*/
        }
    }
    // nothing there, just background
    else {
        auto direction = in.V.GetUnitVector();
        double t = 0.5 * (direction.z + 1.0);
        in.myColor += tracer.GetBackgroundColor(t);
    }

    return ret;
}


void Render(Raytracer& tracer, int i, int j)
{
    std::vector<Ray*> stack;

    Camera* camera = tracer.GetCamera();
    Vector3 ray_O = camera->GetO();
    Color pixel;

    for (int r = -NUM_RESAMPLE; r <= NUM_RESAMPLE; r++) {
        for (int c = -NUM_RESAMPLE; c <= NUM_RESAMPLE; c++) {
            Vector3 ray_V = camera->Emit(i + (double)r / (NUM_RESAMPLE * 2 + 1), j + (double)c / (NUM_RESAMPLE * 2 + 1));
            Ray res(Vector3(), Vector3(), Color(1.0, 1.0, 1.0));
            Ray* origin = new Ray(ray_O, ray_V, Color(1.0, 1.0, 1.0), &res);
            stack.push_back(origin);
            //std::cout << stack.size() << std::endl;
            while (stack.size() > 0) {
                Ray* rayIn = stack.back();

                // not traced yet
                if (!rayIn->visited) {
                    Ray* rayRefl = rayIn->Generate();
                    Ray* rayRefr = rayIn->Generate();
                    int res = TraceRay(tracer, *rayIn, *rayRefl, *rayRefr);
                    // go on tracing
                    if (res != 0) {
                        if (res | REFLE_BIT) {
                            stack.push_back(rayRefl);
                        }
                        if (res | REFRA_BIT) {
                            stack.push_back(rayRefr);
                        }
                    }
                    // end of tracing
                    else {
                        rayIn->Finish();
                        stack.pop_back();
                        delete rayIn;
                    }
                }
                // children rays have returned color back already
                else {
                    rayIn->Finish();
                    stack.pop_back();
                    delete rayIn;
                }
            } /* while */

            // now all rays have finished
            pixel += res.myColor / pow((NUM_RESAMPLE * 2 + 1), 2);
        }
    } /* for */
    camera->SetColor(i, j, pixel);
}


} /* CG_TRACING_H_ */

#endif /* CG_TRACING_H_ */
