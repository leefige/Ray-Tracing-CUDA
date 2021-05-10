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
    const int depth;
    bool visited;

    Ray(const Vector3& o, const Vector3& v, Color& color, Color attenuation, int depth = 0) :
        O(o), V(v), attenuation(attenuation),
        parentColor(color), depth(depth), visited(false)
    { }

    Ray Generate()
    {
        return Ray(Vector3(), Vector3(), myColor, Color(), depth + 1);
    }

    void Finish()
    {
        myColor.Confine();
        parentColor += myColor * attenuation;
    }

private:
    Color& parentColor;
};


} /* namespace cg */

#endif /* CG_RAY_H_ */
