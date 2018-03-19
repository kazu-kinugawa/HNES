#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include "Adam.hpp"
#include "GradChecker.hpp"

class Convolution{
public:
  class State;
  class Grad;
  class AdamGrad;

  enum PARAM{
    KERNEL, BIAS,
  };

  MatD W;
//  VecD b;
  MatD b;

  Convolution(){}
  Convolution(const int patchSize, const int kernelNum);

  void init(Rand& rnd, const Real scale);
  void forward(MatD& x, Convolution::State* cur);
  void backward1(MatD& delx, const Convolution::State* cur);
  void backward1(MatD& delx, const Convolution::State* cur, Convolution::Grad& grad);
  template<Convolution::PARAM> void backward2(const Convolution::State* cur, Convolution::Grad& grad);

  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class Convolution::State{
public:
  MatD* x;
  MatD y;
  MatD dely;
  void clear(){
    this->x = NULL;
    this->y = MatD();
    this->dely = MatD();
  }
  ~State(){
    this->clear();
  }
  void setZero(){
    this->dely.setZero();
  }
};

class Convolution::Grad{
public:
  Convolution::Grad* gradHist;
  Convolution::AdamGrad* adamGradHist;

  MatD W;
//  VecD b;
  MatD b;

  Grad():gradHist(0) {}
  Grad(const Convolution& conv);

  void init();
  Real norm();
  void l2reg(const Real lambda, const Convolution& conv);
  void l2reg(const Real lambda, const Convolution& conv, const Convolution& target);
  void sgd(const Real lr, Convolution& conv);
  void adagrad(const Real lr, Convolution& conv, const Real initVal = 1.0);
  void momentum(const Real lr, const Real m, Convolution& conv);
  void adam(const Real lr, const Adam::HyperParam& adam, Convolution& conv);//add by kinu
  void fill(const Real initVal);// add by kinu
  void saveHist(std::ofstream& ofs);
  void loadHist(std::ifstream& ifs);

  void operator += (const Convolution::Grad& grad);
  void operator /= (const Real val);
  void clear();
};

class Convolution::AdamGrad{
public:
  Adam::Grad<MatD> W;
  // Adam::Grad<VecD> b;
  Adam::Grad<MatD> b;

  AdamGrad(const Convolution& conv);
};

template<> inline void Convolution::backward2<Convolution::KERNEL>(const Convolution::State* cur, Convolution::Grad& grad){
  grad.W.noalias() += cur->x->transpose()*cur->dely;
}
template<> inline void Convolution::backward2<Convolution::BIAS>(const Convolution::State* cur, Convolution::Grad& grad){
  grad.b += cur->dely.colwise().sum();
}

class ConvolutionGradChecker : public GradChecker{
public:
    Convolution conv;

    Convolution::Grad grad;

    Convolution::State* cur;

    MatD x;
    MatD delx;

    MatD g;

    ConvolutionGradChecker(const int patchNum, const int patchSize, const int kernelNum);

    Real calcLoss();
    void calcGrad();
    // static void test();
};
