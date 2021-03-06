#ifndef CG_RENDER_H_
#define CG_RENDER_H_

#include <vector>

#include "color.h"
#include "ray.h"
#include "vector3.h"
#include "raytracer.h"

namespace cg
{

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
                    int res = tracer.TraceRay(*rayIn, *rayRefl, *rayRefr);
                    // go on tracing
                    if (res != 0) {
                        if (res & TRACER_REFLE_BIT) {
                            stack.push_back(rayRefl);
                        }
                        if (res & TRACER_REFRA_BIT) {
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


} /* CG_RENDER_H_ */

#endif /* CG_RENDER_H_ */
