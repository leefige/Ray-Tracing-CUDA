#ifndef CG_VECTOR3_H_
#define CG_VECTOR3_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "defs.h"

#include "cutils.h"
namespace cg
{

class Vector3
{
public:
    float x , y , z;

    __host__ __device__ Vector3() : x(0), y(0), z(0) {}
    __host__ __device__ explicit Vector3( float X , float Y , float Z ) : x( X ) , y( Y ) , z( Z ) {}
    __host__ __device__ virtual ~Vector3() {}

    friend __host__ __device__ Vector3 operator + ( const Vector3& , const Vector3& );
    friend __host__ __device__ Vector3 operator - ( const Vector3& , const Vector3& );
    friend __host__ __device__ Vector3 operator * ( const Vector3& , const float& );
    friend __host__ __device__ Vector3 operator * ( const float& , const Vector3& );
    friend __host__ __device__ Vector3 operator / ( const Vector3& , const float& );
    friend __host__ __device__ Vector3 operator * ( const Vector3& , const Vector3& ); //cross product
    friend __host__ __device__ Vector3& operator += ( Vector3& , const Vector3& );
    friend __host__ __device__ Vector3& operator -= ( Vector3& , const Vector3& );
    friend __host__ __device__ Vector3& operator *= ( Vector3& , const float& );
    friend __host__ __device__ Vector3& operator /= ( Vector3& , const float& );
    friend __host__ __device__ Vector3& operator *= ( Vector3& , const Vector3& ); //cross product
    friend __host__ __device__ Vector3 operator - ( const Vector3& );

    __host__ __device__ float Dot( const Vector3& ) const;
    __host__ __device__ float Module2() const;
    __host__ __device__ float Module() const;
    __host__ __device__ float Distance2( Vector3& ) const;
    __host__ __device__ float Distance( Vector3& ) const;
    __host__ __device__ Vector3 Ortho( Vector3 ) const;
    __host__ __device__ Vector3 GetUnitVector() const;
    /* Generate a unit vector perpendicular to this vector. */
    __host__ __device__ Vector3 GetAnVerticalVector() const;
    __host__ __device__ bool IsZeroVector() const;
    __host__ __device__ Vector3 Reflect( Vector3 N ) const;
    __host__ __device__ Vector3 Refract( Vector3 N , float n ) const;
    /* Generate a random vector with the same length above the tangent plaine. */
    __host__ __device__ Vector3 Diffuse() const;
    __host__ __device__ Vector3 Rotate( Vector3 axis , float theta ) const;

    __host__ __device__ float& GetCoord(int axis);

    void Input(std::stringstream& fin);
    __host__ __device__ void AssRandomVector();
};

// ===============================================================

__host__ __device__ Vector3 operator + ( const Vector3& A , const Vector3& B )
{
    return Vector3( A.x + B.x , A.y + B.y , A.z + B.z );
}

__host__ __device__ Vector3 operator - ( const Vector3& A , const Vector3& B )
{
    return Vector3( A.x - B.x , A.y - B.y , A.z - B.z );
}

__host__ __device__ Vector3 operator * ( const Vector3& A , const float& k )
{
    return Vector3( A.x * k , A.y * k , A.z * k );
}

__host__ __device__ Vector3 operator * ( const float& k, const Vector3& A )
{
    return Vector3( A.x * k , A.y * k , A.z * k );
}

__host__ __device__ Vector3 operator / ( const Vector3& A , const float& k )
{
    return Vector3( A.x / k , A.y / k , A.z / k );
}

__host__ __device__ Vector3 operator * ( const Vector3& A , const Vector3& B )
{
    return Vector3( A.y * B.z - A.z * B.y , A.z * B.x - A.x * B.z , A.x * B.y - A.y * B.x );
}

__host__ __device__ Vector3& operator += ( Vector3& A , const Vector3& B )
{
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    return A;
}

__host__ __device__ Vector3& operator -= ( Vector3& A , const Vector3& B )
{
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    return A;
}

__host__ __device__ Vector3& operator *= ( Vector3& A , const float& k )
{
    A.x *= k;
    A.y *= k;
    A.z *= k;
    return A;
}

__host__ __device__ Vector3& operator /= ( Vector3& A, const float& k )
{
    A.x /= k;
    A.y /= k;
    A.z /= k;
    return A;
}

__host__ __device__ Vector3& operator *= ( Vector3& A , const Vector3& B )
{
    auto x = A.y * B.z - A.z * B.y;
    auto y = A.z * B.x - A.x * B.z;
    auto z = A.x * B.y - A.y * B.x;
    A.x = x;
    A.y = y;
    A.z = z;
    return A;
}

__host__ __device__ Vector3 operator - ( const Vector3& A )
{
    return Vector3( -A.x , -A.y , -A.z );
}

__host__ __device__ float Vector3::Dot( const Vector3& term ) const
{
    return x * term.x + y * term.y + z * term.z;
}

__host__ __device__ float Vector3::Module2() const
{
    return x * x + y * y + z * z;
}

__host__ __device__ float Vector3::Module() const
{
    return sqrt( x * x + y * y + z * z );
}

__host__ __device__ float Vector3::Distance2( Vector3& term ) const
{
    return ( term - *this ).Module2();
}

__host__ __device__ float Vector3::Distance( Vector3& term ) const
{
    return ( term - *this ).Module();
}

__host__ __device__ Vector3 Vector3::Ortho( Vector3 term ) const
{
    return *this - term * this->Dot(term);
}

__host__ __device__ float& Vector3::GetCoord( int axis )
{
    if (axis == 0) {
        return x;
    } else if (axis == 1) {
        return y;
    } else {
        return z;
    }
}

__host__ __device__ Vector3 Vector3::GetUnitVector() const
{
    return *this / Module();
}

__host__ __device__ void Vector3::AssRandomVector()
{
    do {
        x = 2 * preciseRan() - 1;
        y = 2 * preciseRan() - 1;
        z = 2 * preciseRan() - 1;
    } while ( x * x + y * y + z * z > 1 || x * x + y * y + z * z < EPS );
    *this = GetUnitVector();
}

__host__ __device__ Vector3 Vector3::GetAnVerticalVector() const
{
    Vector3 ret = *this * Vector3( 0 , 0 , 1 );
    if (ret.IsZeroVector()) {
        ret = Vector3(1, 0, 0);
    } else {
        ret = ret.GetUnitVector();
    }
    return ret;
}

__host__ __device__ bool Vector3::IsZeroVector() const
{
    return fabs( x ) < EPS && fabs( y ) < EPS && fabs( z ) < EPS;
}

void Vector3::Input( std::stringstream& fin )
{
    fin >> x >> y >> z;
}

__host__ __device__ Vector3 Vector3::Reflect( Vector3 N ) const
{
    return *this - N * ( 2 * Dot( N ) );
}

__host__ __device__ Vector3 Vector3::Refract( Vector3 N , float n ) const
{
    Vector3 V = GetUnitVector();
    float cosI = -N.Dot( V ) , cosT2 = 1 - ( n * n ) * ( 1 - cosI * cosI );
    if ( cosT2 > EPS ) return V * n + N * ( n * cosI - sqrt( cosT2 ) );
    return V.Reflect( N );
}

__host__ __device__ __host__ __device__ Vector3 Vector3::Diffuse() const
{
    Vector3 Vert = GetAnVerticalVector();
    // sqrt to avoid too small value
    // theta in [0, pi/2), more likely to be a larger value
    float theta = acos( sqrt( preciseRan() ) );
    float phi = preciseRan() * 2 * PI;
    // rotate this vector around a perpendicular vector by theta,
    // then rotate the result vector around the original vector by phi.
    return Rotate( Vert , theta ).Rotate( *this , phi );
}

__host__ __device__ Vector3 Vector3::Rotate( Vector3 axis , float theta ) const
{
    Vector3 ret;
    float cost = cos( theta );
    float sint = sin( theta );
    axis = axis.GetUnitVector();

    ret.x += x * ( axis.x * axis.x + ( 1 - axis.x * axis.x ) * cost );
    ret.x += y * ( axis.x * axis.y * ( 1 - cost ) - axis.z * sint );
    ret.x += z * ( axis.x * axis.z * ( 1 - cost ) + axis.y * sint );

    ret.y += x * ( axis.y * axis.x * ( 1 - cost ) + axis.z * sint );
    ret.y += y * ( axis.y * axis.y + ( 1 - axis.y * axis.y ) * cost );
    ret.y += z * ( axis.y * axis.z * ( 1 - cost ) - axis.x * sint );

    ret.z += x * ( axis.z * axis.x * ( 1 - cost ) - axis.y * sint );
    ret.z += y * ( axis.z * axis.y * ( 1 - cost ) + axis.x * sint );
    ret.z += z * ( axis.z * axis.z + ( 1 - axis.z * axis.z ) * cost );
    return ret;
}

} /* namespace cg */

#endif /* CG_VECTOR3_H_ */
