#pragma once

#include "Matrix.hpp"

class Adam{
public:
  class HyperParam;
  class MatGrad;
  class VecGrad;
  class SclGrad;
  static void adam(const MatD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::MatGrad& gradHist, MatD& param);
  static void adam(const VecD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::VecGrad& gradHist, VecD& param);
  static void adam(const Real& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::SclGrad& gradHist, Real& param);

  // template<class type> class Grad{};
  template<class type> class Grad;
  static void adam(const MatD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::Grad<MatD>& gradHist, MatD& param);
  static void adam(const VecD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::Grad<VecD>& gradHist, VecD& param);
  static void adam(const Real& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::Grad<Real>& gradHist, Real& param);
};

class Adam::HyperParam{
public:
  Real alpha;
  Real beta1;
  Real beta2;
  Real eps;
  unsigned int t;
  HyperParam():
    alpha(1.0e-03), beta1(0.9), beta2(0.999), eps(1.0e-08), t(0)
  {}
  HyperParam(const Real _alpha, const Real _beta1, const Real _beta2, const Real _eps):
    alpha(_alpha), beta1(_beta1), beta2(_beta2), eps(_eps), t(0)
  {}
  void operator = (const Adam::HyperParam& hp);
  Real lr();
};

class Adam::MatGrad{
public:
  MatD m;
  MatD v;
  MatD grad;
  MatGrad(){};
  MatGrad(const MatD temp);
  MatGrad(const int row, const int col);
  void getGrad(const MatD& grad, const Adam::HyperParam& hp);
  virtual void clear();
  ~MatGrad(){ this->clear(); };
};

class Adam::VecGrad{
public:
  VecD m;
  VecD v;
  VecD grad;
  VecGrad(){};
  VecGrad(const VecD temp);
  VecGrad(const int row);
  void getGrad(const VecD& grad, const Adam::HyperParam& hp);
  virtual void clear();
  ~VecGrad(){ this->clear(); };
};

class Adam::SclGrad{
public:
  Real m;
  Real v;
  Real grad;
  SclGrad():m(0.0), v(0.0), grad(0.0){}
  void getGrad(const Real grad, const Adam::HyperParam& hp);
};

template<>
class Adam::Grad<MatD>{
public:
  MatD m;
  MatD v;
  MatD grad;
  Grad(){};
  Grad(const MatD& temp){
    this->m = MatD::Zero(temp.rows(), temp.cols());
    this->v = MatD::Zero(temp.rows(), temp.cols());
    this->grad = MatD::Zero(temp.rows(), temp.cols());
  }
  Grad(const int row, const int col){
    this->m = MatD::Zero(row, col);
    this->v = MatD::Zero(row, col);
    this->grad = MatD::Zero(row, col);
  }
  void getGrad(const MatD& input, const Adam::HyperParam& hp){
    this->m += (1-hp.beta1)*(input-this->m);
    this->v.array() += (1-hp.beta2)*(input.array()*input.array()-this->v.array());
    this->grad = this->m.array() / (this->v.array().sqrt() + hp.eps);
  }
  void clear(){
    this->m = MatD();
    this->v = MatD();
    this->grad = MatD();
  }
  ~Grad(){ this->clear(); }
};

template<>
class Adam::Grad<VecD>{
public:
  VecD m;
  VecD v;
  VecD grad;
  Grad(){}
  Grad(const VecD& temp){
    this->m = VecD::Zero(temp.rows());
    this->v = VecD::Zero(temp.rows());
    this->grad = VecD::Zero(temp.rows());
  }
  Grad(const int row){
    this->m = VecD::Zero(row);
    this->v = VecD::Zero(row);
    this->grad = VecD::Zero(row);
  }
  void getGrad(const VecD& input, const Adam::HyperParam& hp){
    this->m += (1-hp.beta1)*(input-this->m);
    this->v.array() += (1-hp.beta2)*(input.array()*input.array()-this->v.array());
    this->grad = this->m.array() / (this->v.array().sqrt() + hp.eps);
  }
  void clear(){
    this->m = VecD();
    this->v = VecD();
    this->grad = VecD();
  }
  ~Grad(){ this->clear(); }
};

template<>
class Adam::Grad<Real>{
public:
  Real m;
  Real v;
  Real grad;
  Grad():m(0.0), v(0.0), grad(0.0){}
  Grad(const Real temp):m(0.0), v(0.0), grad(0.0){}
  void getGrad(const Real input, const Adam::HyperParam& hp){
    this->m += (1-hp.beta1)*(input-this->m);
    this->v += (1-hp.beta2)*(input*input-this->v);
    this->grad = this->m / (std::sqrt(this->v) + hp.eps);
  }
};
