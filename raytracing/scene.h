#ifndef CG_SCENE_H_
#define CG_SCENE_H_

#include<string>
#include<fstream>
#include<sstream>

#include"color.h"
#include"vector3.h"
#include"primitive.h"
#include"light.h"
#include"camera.h"

namespace cg
{

class Scene
{
    Primitive* primitive_head;

public:
    Scene();
    ~Scene();

    Primitive* GetPrimitiveHead() { return primitive_head; }

    void CreateScene(Primitive* primitive_head_p);
    CollidePrimitive FindNearestPrimitiveGetCollide( Vector3 ray_O , Vector3 ray_V ) const;
};

// ===============================================

Scene::Scene()
{
    primitive_head = NULL;
}

Scene::~Scene()
{
    while ( primitive_head != NULL ) {
        Primitive* next_head = primitive_head->GetNext();
        if ( primitive_head->GetMaterial()->texture != NULL )
            delete primitive_head->GetMaterial()->texture;
        delete primitive_head;
        primitive_head = next_head;
    }
}

void Scene::CreateScene(Primitive* primitive_head_p)
{
    primitive_head = primitive_head_p;
}

CollidePrimitive Scene::FindNearestPrimitiveGetCollide( Vector3 ray_O , Vector3 ray_V ) const
{
    CollidePrimitive ret;

    for ( Primitive* now = primitive_head ; now != NULL ; now = now->GetNext() ) {
        CollidePrimitive tmp = now->Collide(ray_O, ray_V);
        if(tmp.dist < ret.dist) {
            ret = tmp;
        }
    }

    return ret;
}

} /* namespace cg */

#endif /* CG_SCENE_H_ */
