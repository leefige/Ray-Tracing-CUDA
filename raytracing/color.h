#ifndef CG_COLOR_H_
#define CG_COLOR_H_

#include <sstream>

namespace cg
{

class Color
{
public:
    double r , g , b;

    explicit Color( double R = 0 , double G = 0 , double B = 0 ) : r( R ) , g( G ) , b( B ) {}
    ~Color() {}

    friend Color operator + ( const Color& , const Color& );
    friend Color operator - ( const Color& , const Color& );
    friend Color operator * ( const Color& , const Color& );
    friend Color operator * ( const Color& , const double& );
    friend Color operator / ( const Color& , const double& );
    friend Color& operator += ( Color& , const Color& );
    friend Color& operator -= ( Color& , const Color& );
    friend Color& operator *= ( Color& , const double& );
    friend Color& operator /= ( Color& , const double& );
    void Confine(); //luminance must be less than or equal to 1
    void Input( std::stringstream& );
};

// ===============================================

Color operator + ( const Color& A , const Color& B )
{
    return Color( A.r + B.r , A.g + B.g , A.b + B.b );
}

Color operator - ( const Color& A , const Color& B )
{
    return Color( A.r - B.r , A.g - B.g , A.b - B.b );
}

Color operator * ( const Color& A , const Color& B )
{
    return Color( A.r * B.r , A.g * B.g , A.b * B.b );
}

Color operator * ( const Color& A , const double& k )
{
    return Color( A.r * k , A.g * k , A.b * k );
}

Color operator / ( const Color& A , const double& k )
{
    return Color( A.r / k , A.g / k , A.b / k );
}

Color& operator += ( Color& A , const Color& B )
{
    A.r += B.r;
    A.g += B.g;
    A.b += B.b;
    return A;
}

Color& operator -= ( Color& A , const Color& B )
{
    A.r -= B.r;
    A.g -= B.g;
    A.b -= B.b;
    return A;
}

Color& operator *= ( Color& A , const double& k )
{
    A.r *= k;
    A.g *= k;
    A.b *= k;
    return A;
}

Color& operator /= ( Color& A , const double& k )
{
    A.r /= k;
    A.g /= k;
    A.b /= k;
    return A;
}

void Color::Confine()
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
