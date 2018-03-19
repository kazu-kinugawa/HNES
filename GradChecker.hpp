#pragma once

#include <iostream>
#include "Matrix.hpp"

class GradChecker{
public:
  const Real EPS = 1.0e-04;
  const Real threshold = 1.0e-06;

  virtual void calcGrad() = 0;
  virtual Real calcLoss() = 0;

  void gradCheck(const Real& grad, Real& param){
    const Real val = param;
    param = val + this->EPS;
    const Real objFuncPlus = this->calcLoss();
    param = val - this->EPS;
    const Real objFuncMinus = this->calcLoss();
    param = val;

    const Real diff = std::abs( grad - (objFuncPlus-objFuncMinus)/(2.0*this->EPS) );

    if(diff >= this->threshold){
      std::cout << "SOMETHING WRONG!\t\t" << diff << std::endl;
    }
  }
  void gradCheck(const VecD& grad, VecD& param){
    for (int i = 0, i_end = param.rows(); i < i_end; ++i) {
      this->gradCheck(grad.coeffRef(i,0), param.coeffRef(i,0));
    }
  }
  void gradCheck(const MatD& grad, MatD& param){
    for (int i = 0, i_end = param.rows(); i < i_end; ++i) {
      for (int j = 0, j_end = param.cols(); j < j_end; ++j) {
        this->gradCheck(grad.coeffRef(i,j), param.coeffRef(i,j));
      }
    }
  }
};
