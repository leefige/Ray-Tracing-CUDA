#ifndef CG_RAY_H_
#define CG_RAY_H_

#include "color.h"
#include "vector3.h"

namespace cg
{

class Ray
{
public:
    Vector3 O;
    Vector3 V;

    Color myColor;
    Color attenuation;

    Ray* parent;

    const int depth;
    bool visited;

    Ray(const Vector3& o, const Vector3& v, const Color& attenuation, Ray* parent = nullptr, int depth = 0) :
        O(o), V(v), attenuation(attenuation),
        parent(parent), depth(depth), visited(false)
    { }

    Ray* Generate() { return new Ray(Vector3(), Vector3(), Color(), this, depth + 1); }

    void Finish()
    {
        myColor.Confine();
        parent->myColor += myColor * attenuation;
    }

};


} /* namespace cg */

#endif /* CG_RAY_H_ */
