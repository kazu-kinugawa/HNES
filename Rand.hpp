#pragma once

#include "Matrix.hpp"
#include <vector>
#include <iostream>
#include <limits> 

// for std::numeric_limits
/* for initiallization */

/*
const unsigned int bits = std::numeric_limits<unsigned long>::digits;

unsigned long lRot(unsigned long x, unsigned int n)
{
  n %= bits;
  return ( x << n ) | ( x >> ( bits - n ) );
}

unsigned long rRot(unsigned long x, unsigned int n)
{
  n %= bits;
  return ( x >> n ) | ( x << ( bits - n ) );
}
*/

class Rand{
public:
  static Rand r_;

  Rand():
    x(123456789), y(362436069), z(521288629), w(88675123) {};

  /*
  Rand(unsigned long w_ = 88675123):
    x(123456789), y(362436069), z(521288629), w(w_) {};
  */

  //Rand(unsigned long seed);
  void init(unsigned long seed);
  //void init2(unsigned long seed);

  unsigned long next();
  Real zero2one();
  void uniform(MatD& mat, const Real scale = 1.0);
  void uniform(VecD& vec, const Real scale = 1.0);
  void uniform(Real& param, const Real scale = 1.0);
  Real gauss(Real sigma, Real mu = 0.0);
  void gauss(MatD& mat, Real sigma, Real mu = 0.0);
  void setMask(VecD& mask, const Real p = 0.5);
  template <typename T> void shuffle(std::vector<T>& data);
  void print(){ std::cout << "x = "<< this->x << ", y = " << this->y << ", z = " << this->z << ", w = " << this->w << std::endl; };

private:
  unsigned long x;
  unsigned long y;
  unsigned long z;
  unsigned long w;
  unsigned long t; //tmp
};

/* for initiallization */
/*
Rand::Rand(unsigned long seed):
x(123456789), y(362436069), z(521288629), w(88675123){
  this->x^=seed;
  this->y^=lRot(seed,17);
  this->z^=lRot(seed,31);
  this->w^=lRot(seed,18);
  // seed = (unsigned long)time(NULL)
  // "^=" means bit xor
  // unsigned long is 4 bytes
}

inline void Rand::init2(unsigned long seed){
  this->x^=seed;
  this->y^=lRot(seed,17);
  this->z^=lRot(seed,31);
  this->w^=lRot(seed,18);
}
*/
inline void Rand::init(unsigned long seed){
  this->x = seed = 1812433253U * (seed ^ (seed >> 30)) + 1;
  this->y = seed = 1812433253U * (seed ^ (seed >> 30)) + 2;
  this->z = seed = 1812433253U * (seed ^ (seed >> 30)) + 3;
  this->w = seed = 1812433253U * (seed ^ (seed >> 30)) + 4;
}
/* for initiallization */

inline unsigned long Rand::next(){
  this->t=(this->x^(this->x<<11));
  this->x=this->y;
  this->y=this->z;
  this->z=this->w;
  return (this->w=(this->w^(this->w>>19))^(this->t^(this->t>>8)));
}

inline Real Rand::zero2one(){
  return ((this->next()&0xFFFF)+1)/65536.0;
}

inline void Rand::uniform(MatD& mat, const Real scale){
  for (int i = 0, i_end = (int)mat.rows(); i < i_end; ++i){
    for (int j = 0, j_end = (int)mat.cols(); j < j_end; ++j){
      mat.coeffRef(i, j) = 2.0*this->zero2one()-1.0;
    }
  }

  mat *= scale;
}

inline void Rand::uniform(VecD& vec, const Real scale){
  for (int i = 0, i_end = (int)vec.rows(); i < i_end; ++i){
    vec.coeffRef(i, 0) = 2.0*this->zero2one()-1.0;
  }

  vec *= scale;
}

inline void Rand::uniform(Real& param, const Real scale){
  param = scale * (2.0*this->zero2one() - 1.0);
}

inline Real Rand::gauss(Real sigma, Real mu){
  /*
  return
    mu+
    sigma*
    sqrt(-2.0*log(this->zero2one()))*
    sin(2.0*M_PI*this->zero2one());
  */
  return mu + sigma*sqrt(-2.0*log(this->zero2one()))*sin(2.0*3.14159265358979323846*this->zero2one());
}

inline void Rand::gauss(MatD& mat, Real sigma, Real mu){
  for (int i = 0, i_end = (int)mat.rows(); i < i_end; ++i){
    for (int j = 0, j_end = (int)mat.cols(); j < j_end; ++j){
      mat.coeffRef(i, j) = this->gauss(sigma, mu);
    }
  }
}

inline void Rand::setMask(VecD& mask, const Real p){
  for (int i = 0, i_end = (int)mask.rows(); i < i_end; ++i){
    mask.coeffRef(i, 0) = (this->zero2one() < p ? 1.0 : 0.0);
  }
}

template <typename T> inline void Rand::shuffle(std::vector<T>& data){
  T tmp;

  for (int i = data.size(), a, b; i > 1; --i){
    a = i-1;
    b = this->next()%i;
    tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
  }
}
