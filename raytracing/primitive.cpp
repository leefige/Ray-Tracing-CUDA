#include"primitive.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "solver.h"

#define ran() ( double( rand() % 32768 ) / 32768 )

const int BEZIER_MAX_DEGREE = 5;
const int Combination[BEZIER_MAX_DEGREE + 1][BEZIER_MAX_DEGREE + 1] =
{	0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0,
    1, 2, 1, 0, 0, 0,
    1, 3, 3, 1, 0, 0,
    1, 4, 6, 4, 1, 0,
    1, 5, 10,10,5, 1
};

const int MAX_COLLIDE_TIMES = 10;
const int MAX_COLLIDE_RANDS = 10;

std::pair<double, double> ExpBlur::GetXY()
{
    double x,y;
    x = ran();
    // x in [0, 1), but with a higher prob to be a small value
    x = pow(2, x)-1;
    y = rand();
    return std::pair<double, double>(x*cos(y),x*sin(y));
}

// ====================================================

Material::Material() {
    color = absor = Color();
    refl = refr = 0;
    diff = spec = 0;
    rindex = 0;
    drefl = 0;
    texture = NULL;
    blur = new ExpBlur();
}

void Material::Input( std::string var , std::stringstream& fin ) {
    if ( var == "color=" ) color.Input( fin );
    if ( var == "absor=" ) absor.Input( fin );
    if ( var == "refl=" ) fin >> refl;
    if ( var == "refr=" ) fin >> refr;
    if ( var == "diff=" ) fin >> diff;
    if ( var == "spec=" ) fin >> spec;
    if ( var == "drefl=" ) fin >> drefl;
    if ( var == "rindex=" ) fin >> rindex;
    if ( var == "texture=" ) {
        std::string file; fin >> file;
        texture = new Bmp;
        texture->Input( file );
    }
    if ( var == "blur=" ) {
        std::string blurname; fin >> blurname;
        if(blurname == "exp")
            blur = new ExpBlur();
    }
}

// ====================================================

Primitive::Primitive() {
    sample = rand();
    material = new Material;
    next = NULL;
}

Primitive::Primitive( const Primitive& primitive ) {
    *this = primitive;
    material = new Material;
    *material = *primitive.material;
}

Primitive::~Primitive() {
    delete material;
}

void Primitive::Input( std::string var , std::stringstream& fin ) {
    material->Input( var , fin );
}

// -----------------------------------------------

Sphere::Sphere() : Primitive() {
    De = Vector3( 0 , 0 , 1 );
    Dc = Vector3( 0 , 1 , 0 );
}

void Sphere::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    if ( var == "R=" ) fin >> R;
    if ( var == "De=" ) De.Input( fin );
    if ( var == "Dc=" ) Dc.Input( fin );
    Primitive::Input( var , fin );
}

CollidePrimitive Sphere::Collide( Vector3 ray_O , Vector3 ray_V ) {
    ray_V = ray_V.GetUnitVector();
    Vector3 P = ray_O - O;
    double b = -P.Dot( ray_V );
    double det = b * b - P.Module2() + R * R;
    CollidePrimitive ret;

    if ( det > EPS ) {
        det = sqrt( det );
        double x1 = b - det  , x2 = b + det;

        if ( x2 < EPS ) return ret;
        if ( x1 > EPS ) {
            ret.dist = x1;
            ret.front = true;
        } else {
            ret.dist = x2;
            ret.front = false;
        }
    } else {
        return ret;
    }

    ret.C = ray_O + ray_V * ret.dist;
    ret.N = ( ret.C - O ).GetUnitVector();
    if ( ret.front == false ) ret.N = -ret.N;
    ret.isCollide = true;
    ret.collide_primitive = this;
    return ret;
}

Color Sphere::GetTexture(Vector3 crash_C) {
    Vector3 I = ( crash_C - O ).GetUnitVector();
    double a = acos( -I.Dot( De ) );
    double b = acos( std::min( std::max( I.Dot( Dc ) / sin( a ) , -1.0 ) , 1.0 ) );
    double u = a / PI , v = b / 2 / PI;
    if ( I.Dot( Dc * De ) < 0 ) v = 1 - v;
    return material->texture->GetSmoothColor( u , v );
}

// -----------------------------------------------

void Plane::Input( std::string var , std::stringstream& fin ) {
    if ( var == "N=" ) N.Input( fin );
    if ( var == "R=" ) fin >> R;
    if ( var == "Dx=" ) Dx.Input( fin );
    if ( var == "Dy=" ) Dy.Input( fin );
    Primitive::Input( var , fin );
}

CollidePrimitive Plane::Collide( Vector3 ray_O , Vector3 ray_V ) {
    ray_V = ray_V.GetUnitVector();
    N = N.GetUnitVector();
    double d = N.Dot( ray_V );
    CollidePrimitive ret;
    if ( fabs( d ) < EPS ) return ret;
    double l = ( N * R - ray_O ).Dot( N ) / d;
    if ( l < EPS ) return ret;

    ret.dist = l;
    ret.front = ( d < 0 );
    ret.C = ray_O + ray_V * ret.dist;
    ret.N = ( ret.front ) ? N : -N;
    ret.isCollide = true;
    ret.collide_primitive = this;
    return ret;
}

Color Plane::GetTexture(Vector3 crash_C) {
    double u = crash_C.Dot( Dx ) / Dx.Module2();
    double v = crash_C.Dot( Dy ) / Dy.Module2();
    return material->texture->GetSmoothColor( u , v );
}

// -----------------------------------------------

void Square::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    if ( var == "Dx=" ) Dx.Input( fin );
    if ( var == "Dy=" ) Dy.Input( fin );
    Primitive::Input( var , fin );
}

CollidePrimitive Square::Collide( Vector3 ray_O , Vector3 ray_V ) {
    CollidePrimitive ret;

    // TODO: NEED TO IMPLEMENT
    ray_V = ray_V.GetUnitVector();
    auto N = (Dx * Dy).GetUnitVector();
    double d = N.Dot(ray_V);

    if (fabs(d) < EPS) {
        return ret;
    }

    // solve equation
    double t = (O - ray_O).Dot(N) / d;
    if (t < EPS) {
        return ret;
    }
    auto P = ray_O + ray_V * t;

    // check whether inside square
    double DxLen2 = Dx.Module2();
    double DyLen2 = Dy.Module2();

    double x2 = abs((P - O).Dot(Dx));
    double y2 = abs((P - O).Dot(Dy));
    if (x2 > DxLen2 || y2 > DyLen2) {
        return ret;
    }

    ret.dist = t;
    ret.front = (d < 0);
    ret.C = P;
    ret.N = (ret.front) ? N : -N;
    ret.isCollide = true;
    ret.collide_primitive = this;
    return ret;
}

Color Square::GetTexture(Vector3 crash_C) {
    double u = (crash_C - O).Dot( Dx ) / Dx.Module2() / 2 + 0.5;
    double v = (crash_C - O).Dot( Dy ) / Dy.Module2() / 2 + 0.5;
    return material->texture->GetSmoothColor( u , v );
}

// -----------------------------------------------

void Cylinder::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O1=" ) O1.Input( fin );
    if ( var == "O2=" ) O2.Input( fin );
    if ( var == "R=" ) fin>>R;
    Primitive::Input( var , fin );
}

CollidePrimitive Cylinder::Collide( Vector3 ray_O , Vector3 ray_V ) {
    CollidePrimitive ret;

    // TODO: NEED TO IMPLEMENT
    ray_V = ray_V.GetUnitVector();
    auto N = O1 - O2;
    double lenSide = N.Module();

    N = N.GetUnitVector();

    // check circles
    bool interO1 = false;
    double t1 = FLT_MAX;
    Vector3 P1;

    bool interO2 = false;
    double t2 = FLT_MAX;
    Vector3 P2;

    double d = N.Dot(ray_V);
    if (fabs(d) > EPS) {
        // circle O1
        auto t = (O1 - ray_O).Dot(N) / d;
        if (t > EPS) {
            P1 = ray_O + ray_V * t;
            // check whether inside circle
            if ((P1 - O1).Module() <= R) {
                t1 = t;
                interO1 = true;
            }
        }
        
        // circle O2
        t = (O2 - ray_O).Dot(N) / d;
        if (t > EPS) {
            P2 = ray_O + ray_V * t;
            // check whether inside circle
            if ((P2 - O2).Module() <= R) {
                t2 = t;
                interO2 = true;
            }
        }
    }

    // check side face
    Vector3 O3 = ray_O - O2;
    Vector3 A = O3 - N * O3.Dot(N);
    Vector3 B = ray_V - N * ray_V.Dot(N);

    // (A + B*t) * (A + B*t) = R^2
    double a = B.Module2();
    double b = 2 * A.Dot(B);
    double c = A.Module2() - R * R;

    double det = b * b - 4 * a * c;

    bool interSide = false;
    double tSide = FLT_MAX;
    Vector3 PSide;

    if (det > EPS) {
        det = sqrt(det);
        double x1 = (-b - det) / 2 / a, x2 = (-b + det) / 2 / a;

        // check x1
        if (x1 > EPS) {
            PSide = ray_O + ray_V * x1;
            auto len = (PSide - O2).Dot(N);
            if (len >= 0 && len <= lenSide) {
                tSide = x1;
                interSide = true;
            }
        }

        // check x2
        if (!interSide && x2 > EPS) {
            PSide = ray_O + ray_V * x2;
            auto len = (PSide - O2).Dot(N);
            if (len >= 0 && len <= lenSide) {
                tSide = x2;
                interSide = true;
            }
        }
    }

    // decide which intersect to return
    if (!interO1 && !interO2 && !interSide) {
        return ret;
    }

    double minT = FLT_MAX;
    Vector3 minP;
    Vector3 colliN;
    bool front = false;
    if (t1 < t2) {
        minT = t1;
        minP = P1;

        front = (minP - ray_O).Dot(N) < 0;
        colliN = front ? N : -N;
    } else {
        minT = t2;
        minP = P2;

        front = (minP - ray_O).Dot(N) > 0;
        colliN = front ? -N : N;
    }

    if (tSide < minT) {
        minT = tSide;
        minP = PSide;

        auto P_prim = minP - O2;
        colliN = (P_prim - N * P_prim.Dot(N)).GetUnitVector();
        front = (minP - ray_O).Dot(colliN) < 0;
    }

    ret.dist = minT;
    ret.C = minP;
    ret.N = colliN;
    ret.front = front;
    ret.isCollide = true;
    ret.collide_primitive = this;
    return ret;
}

Color Cylinder::GetTexture(Vector3 crash_C) {
    double u = 0.5 ,v = 0.5;

    // TODO: NEED TO IMPLEMENT
    Vector3 N = O1 - O2;
    double sideLen = N.Module();
    N = N.GetUnitVector();

    Vector3 P = crash_C - O2;
    double height = P.Dot(N);
    Vector3 R_prim = P - N * height;

    // circle O1 or O2
    if (abs((crash_C - O2).Dot(N)) < EPS || abs((crash_C - O1).Dot(N)) < EPS) {
        // this will always return the same Vec for a fixed N
        Vector3 axisX = N.GetAnVerticalVector();
        Vector3 axisY = N * axisX;
        u = R_prim.Dot(axisX) / R / 2 + 0.5;
        v = R_prim.Dot(axisY) / R / 2 + 0.5;
    }
    // side
    else {
        Vector3 axisX = N.GetAnVerticalVector();
        v = height / sideLen;
        double phi = acos(R_prim.Dot(axisX) / R_prim.Module());
        if ((axisX * R_prim).Dot(N) < 0) {
            phi = 2 * PI - phi;
        }
        u = phi / 2 / PI;
    }

    return material->texture->GetSmoothColor( u , v );
}

// -----------------------------------------------

void Bezier::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O1=" ) O1.Input( fin );
    if ( var == "O2=" ) O2.Input( fin );
    if ( var == "P=" ) {
        degree++;
        double newR, newZ;
        fin>>newZ>>newR;
        R.push_back(newR);
        Z.push_back(newZ);
    }
    if ( var == "Cylinder" ) {
        double maxR = 0;
        for (int i = 0; i < R.size(); i++) {
            if (R[i] > maxR) {
                maxR = R[i];
            }
        }
        boundingCylinder = new Cylinder(O1, O2, maxR);
        N = (O1 - O2).GetUnitVector();
        Nx = N.GetAnVerticalVector();
        Ny = N * Nx;
    }
    Primitive::Input( var , fin );
}

CollidePrimitive Bezier::Collide( Vector3 ray_O , Vector3 ray_V ) {
    CollidePrimitive ret;

    // TODO: NEED TO IMPLEMENT
    ray_V = ray_V.GetUnitVector();

    if (!boundingCylinder->Collide(ray_O, ray_V).isCollide) {
        return ret;
    }

    Vector3 O_prim = ray_O - O2;

    // P' dot N = O_prim_N + V_N * t
    double O_prim_N = O_prim.Dot(N);
    double V_N = ray_V.Dot(N);

    // R' = A + t * B
    Vector3 A = O_prim - N * O_prim_N;
    Vector3 B = ray_V - N * V_N;

    double l = (O1 - O2).Module();

    // h' = h0 + h1 * t
    double h0 = O_prim_N / l;
    double h1 = V_N / l;

    // r^2 = r0 + r1 * t + r2 * t^2
    double r0 = A.Module2() / l / l;
    double r1 = 2 * A.Dot(B) / l / l;
    double r2 = B.Module2() / l / l;

    const auto& x = Z;
    const auto& y = R;

    double t0 = (x[0] - h0) / h1;
    double t1 = 2 * (x[1] - x[0]) / h1;
    double t2 = (x[0] + x[2] - 2 * x[1]) / h1;

    // quartic
    double p[5]{0};
    p[4] = pow(y[0], 2) + 4 * pow(y[1], 2) + pow(y[2], 2)
        - 4 * y[0] * y[1] + 2 * y[1] * y[2] - 4 * y[1] * y[2] - r2 * pow(t2, 2);
    p[3] = 12 * y[0] * y[1] - 4 * y[0] * y[2] + 4 * y[1] * y[2] - 8 * pow(y[1], 2) - 2 * r2 * t1 * t2;
    p[2] = 4 * pow(y[1], 2) - 2 * pow(y[0], 2) - 12 * y[0] * y[1] + 2 * y[0] * y[2] - r1 * t2 - r2 * pow(t1, 2) + 2 * r2 * t0 * t2;
    p[1] = 4 * y[0] * y[1] - r1 * t1 - 2 * r2 * t0 * t1;
    p[0] = pow(y[0], 2) - r0 - r1 * t0 - r2 * pow(t0, 2);

    double roots[4]{0};
    int rootCount = 0;

    auto err = solve_quartic_equation(p, roots, &rootCount);
    if (err != 0 && rootCount <= 0) {
        return ret;
    }

    // check roots
    double minT = FLT_MAX;
    double minU = -1;
    for (int i = 0; i < rootCount; i++) {
        const auto& u = roots[i];
        if (u < 0 || u > 1) {
            continue;
        }

        double t = t2 * pow(u, 2) + t1 * u + t0;
        // we want a positive min t
        if (t > EPS && t < minT) {
            minT = t;
            minU = u;
        }
    }

    if (minU < 0) {
        return ret;
    }

    Vector3 P = ray_O + ray_V * minT;
    Vector3 P_prim = P - O2;

    // tangent
    std::vector<double> subX;
    std::vector<double> subY;
    const int degree = x.size() - 1;
    for (int i = 0; i < degree; i++) {
        subX.push_back(degree * (x[i + 1] - x[i]));
        subY.push_back(degree * (y[i + 1] - y[i]));
    }
    auto tangentPair = valueAt(minU, subX, subY);
    // Tangent = (tangentPair.second, 0, tangentPair.first)
    Vector3 Norm(1 / tangentPair.second, 0, -1 / tangentPair.first);  // in corrd O2-Nx-Ny-N
    Norm = Norm.GetUnitVector();
    Norm = Nx * Norm.x + Ny * Norm.y + N * Norm.z;  // in corrd O-X-Y-Z
    Norm = Norm.GetUnitVector();

    Vector3 R_prim = (P_prim - N * P_prim.Dot(N)).GetUnitVector();
    double phi = acos(R_prim.Dot(Nx) / R_prim.Module());
    if ((Nx * R_prim).Dot(N) < 0) {
        phi = 2 * PI - phi;
    }
    Norm = Norm.Rotate(N, phi);
    if (Norm.Dot(R_prim) < 0) {
        Norm = -Norm;
    }

    //Vector3 colliN = (P_prim - N * P_prim.Dot(N)).GetUnitVector();
    //Norm = R_prim;
    bool front = (P - ray_O).Dot(R_prim) < 0;

    ret.isCollide = true;
    ret.collide_primitive = this;
    ret.C = P;
    ret.dist = minT;
    ret.front = front;
    ret.N = Norm;
    return ret;

}

Color Bezier::GetTexture(Vector3 crash_C) {
    double u = 0.5 ,v = 0.5;

    // TODO: NEED TO IMPLEMENT
    double length = (O1 - O2).Module();

    Vector3 P = crash_C - O2;
    double crashLen = P.Dot(N);
    v = 1 - crashLen / length;

    Vector3 R_prim = P - N * crashLen;
    double phi = acos(R_prim.Dot(Nx) / R_prim.Module());
    if ((Nx * R_prim).Dot(N) < 0) {
        phi = 2 * PI - phi;
    }
    u = (2 * PI - phi) / 2 / PI;
    return material->texture->GetSmoothColor( u , v );
}

std::pair<double, double> Bezier::valueAt(double u)
{
    return valueAt(u, Z, R);
}


std::pair<double, double> Bezier::valueAt(double u, const std::vector<double>& xs, const std::vector<double>& ys)
{
    const int degree = xs.size() - 1;
    double x = 0;
    double y = 0;
    for (int i = 0; i <= degree; i++) {
        double factor = double(Combination[degree][i]) * pow(u, i) * pow(1 - u, degree - i);
        x += factor * xs[i];
        y += factor * ys[i];
    }
    return std::make_pair(x, y);
}

