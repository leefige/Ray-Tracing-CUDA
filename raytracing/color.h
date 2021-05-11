#ifndef CG_COLOR_H_
#define CG_COLOR_H_

#include <sstream>

namespace cg
{

class Color
{
public:
    float r , g , b;

    __host__ __device__ explicit Color( float R = 0 , float G = 0 , float B = 0 ) : r( R ) , g( G ) , b( B ) {}
    __host__ __device__ virtual ~Color() {}

    friend __host__ __device__ Color operator + ( const Color& , const Color& );
    friend __host__ __device__ Color operator - ( const Color& , const Color& );
    friend __host__ __device__ Color operator * ( const Color& , const Color& );
    friend __host__ __device__ Color operator * ( const Color& , const float& );
    friend __host__ __device__ Color operator / ( const Color& , const float& );
    friend __host__ __device__ Color& operator += ( Color& , const Color& );
    friend __host__ __device__ Color& operator -= ( Color& , const Color& );
    friend __host__ __device__ Color& operator *= ( Color& , const float& );
    friend __host__ __device__ Color& operator /= ( Color& , const float& );
    void __host__ __device__ Confine(); //luminance must be less than or equal to 1
    void Input( std::stringstream& );
};

// ===============================================

__host__ __device__ Color operator + ( const Color& A , const Color& B )
{
    return Color( A.r + B.r , A.g + B.g , A.b + B.b );
}

__host__ __device__ Color operator - ( const Color& A , const Color& B )
{
    return Color( A.r - B.r , A.g - B.g , A.b - B.b );
}

__host__ __device__ Color operator * ( const Color& A , const Color& B )
{
    return Color( A.r * B.r , A.g * B.g , A.b * B.b );
}

__host__ __device__ Color operator * ( const Color& A , const float& k )
{
    return Color( A.r * k , A.g * k , A.b * k );
}

__host__ __device__ Color operator / ( const Color& A , const float& k )
{
    return Color( A.r / k , A.g / k , A.b / k );
}

__host__ __device__ Color& operator += ( Color& A , const Color& B )
{
    A.r += B.r;
    A.g += B.g;
    A.b += B.b;
    return A;
}

__host__ __device__ Color& operator -= ( Color& A , const Color& B )
{
    A.r -= B.r;
    A.g -= B.g;
    A.b -= B.b;
    return A;
}

__host__ __device__ Color& operator *= ( Color& A , const float& k )
{
    A.r *= k;
    A.g *= k;
    A.b *= k;
    return A;
}

__host__ __device__ Color& operator /= ( Color& A , const float& k )
{
    A.r /= k;
    A.g /= k;
    A.b /= k;
    return A;
}

__host__ __device__ void Color::Confine()
{
    if ( r > 1 ) r = 1;
    if ( g > 1 ) g = 1;
    if ( b > 1 ) b = 1;
}

void Color::Input( std::stringstream& fin )
{
    fin >> r >> g >> b;
}

} /* namespace cg */

#endif /* CG_COLOR_H_ */
