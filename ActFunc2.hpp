#pragma once

#include "LossFunc.hpp"
#include <algorithm>
#include "Rand.hpp"

// MAYBE OK

class ActFunc{
public:
  class State;
  class Sigmoid;
  class Tanh;
  class Relu;
  static void gradCheck();
};

class ActFunc::State{
public:
  /*
  VecD x;
  VecD delx;
  */

  VecD y;
  VecD dely;

  State(){}
  State(const int outputDim){
    this->dely = VecD::Zero(outputDim);
  }
  void clear(){
    /*
    this->x = VecD();
    this->delx = VecD();
    */
    this->y = VecD();
    this->dely = VecD();
  }
  ~State(){ this->clear(); }
  void setZero(){
    this->dely.setZero();
  }
};

class ActFunc::Sigmoid{
public:
  /*
  Real sigmoid(const Real x){
    return 1.0/(1.0+::exp(-x));
  }
  void forward(const VecD& x, ActFunc::State* cur){
    cur->y = x.unaryExpr(this->sigmoid);
  }
  */
  static Real sigmoid(const Real x){
    return 1.0/(1.0+::exp(-x));
  }
  void forward(const VecD& x, ActFunc::State* cur){
    cur->y = x.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::Sigmoid::sigmoid));

    cur->dely = VecD::Zero(cur->y.rows());// for backprop
  }
  void backward(VecD& delx, const ActFunc::State* cur){
    delx.array() += cur->y.array()*(1.0-cur->y.array())*cur->dely.array();
  }
  void forward(VecD& x){
    // for lstm
    x = x.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::Sigmoid::sigmoid));
  }
  VecD backward(const VecD& x){
    // for lstm
    return x.array()*(1.0-x.array());
  }
  void forward(Real& x){
    // for lstm
    x = 1.0/(1.0+::exp(-x));
  }
  Real backward(const Real x){
    // for lstm
    return x*(1.0-x);
  }
};

class ActFunc::Tanh{
public:
  void forward(const VecD& x, ActFunc::State* cur){
    cur->y = x.unaryExpr(std::ptr_fun(::tanh));

    cur->dely = VecD::Zero(cur->y.rows());// for backprop
  }
  void backward(VecD& delx, const ActFunc::State* cur){
    delx.array() += (1.0-cur->y.array().square()).array()*cur->dely.array();
  }
  void forward(VecD& x){
    // for lstm
    x = x.unaryExpr(std::ptr_fun(::tanh));
  }
  VecD backward(const VecD& x){
    // for lstm
    return 1.0-x.array().square();
  }
};

class ActFunc::Relu{
public:
  static Real relu(const Real x){
    return std::max<Real>(0, x);
  }
  static Real bRelu(const Real x){
    return (x <= 0) ? 0 : 1;
  }
  void forward(const VecD& x, ActFunc::State* cur){
    cur->y = x.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::Relu::relu));

    cur->dely = VecD::Zero(cur->y.rows());// for backprop
  }
  void backward(VecD& delx, const ActFunc::State* cur){
    delx.array() += cur->y.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::Relu::bRelu)).array()*cur->dely.array();
  }

  void forward(VecD& x){
    x = x.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::Relu::relu));
  }
  VecD backward(const VecD& x){
    VecD res = x;
    res = x.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::Relu::bRelu));
    return res;
  }
  /*
  void forward(const VecD& x, ActFunc::State* cur){
    for (unsigned int i = 0, i_end = x.rows(); i < i_end; ++i){
      cur->y.coeffRef(i,0) = std::max<Real>(0.0, x.coeffRef(i,0));
    }
  }
  void backward(ActFunc::State* cur){
    cur->delx = cur->y;
    for (unsigned int i = 0, i_end = cur->delx.rows(); i < i_end; ++i){
      cur->delx.coeffRef(i,0) = (cur->delx.coeffRef(i,0) <= 0.0) ? 0.0 : 1.0;
    }
    cur->delx.array() *= cur->dely.array();
  }
  */
};

/*
void ActFunc::gradCheck(){

  Rand rnd;
  const Real scale = 0.05;

  LossFunc<BinaryCrossEntropy> lf;

  // ActFunc::Tanh af;
  ActFunc::Relu af;

  ActFunc::State* cur = new ActFunc::State;

  VecD x = VecD(20);
  rnd.uniform(x, scale);

  VecD goldOutput = VecD(20);
  rnd.uniform(goldOutput, scale);

  std::cout << "calcGrad" << std::endl;

  af.forward(x, cur);
  cur->dely = lf.backward(cur->y, goldOutput);
  af.backward(cur);

  std::cout << "x" << std::endl;
  std::cout << x << std::endl;
  std::cout << "y" << std::endl;
  std::cout << cur->y << std::endl;

  VecD delx = cur->delx;

  std::cout << "Grad Check Start" << std::endl;

  const Real EPS = 1.0e-04;
  Real val;
  Real objFuncPlus, objFuncMinus;
  Real diff;

  VecD& param = x;
  VecD& grad = delx;

  std::cout << "Input" << std::endl;

  for (int i = 0, i_end = x.rows(); i < i_end; ++i) {
    val = param.coeff(i, 0);

    param.coeffRef(i, 0) = val + EPS;

    af.forward(x, cur);
    objFuncPlus =  lf.forward(cur->y, goldOutput);

    param.coeffRef(i, 0) = val - EPS;

    af.forward(x, cur);
    objFuncMinus =  lf.forward(cur->y, goldOutput);

    param.coeffRef(i, 0) = val;

    diff = std::abs( grad.coeff(i, 0) - (objFuncPlus-objFuncMinus)/(2.0*EPS) );
    if(diff >= 1.0e-6){
      std::cout << "SOMETHING WRONG!\t\t" << diff << std::endl;
    }
  }
}
*/
