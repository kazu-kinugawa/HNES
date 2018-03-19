#pragma once

#include "Matrix.hpp"
#include <iostream>

class MaxPooling{
public:
  class State;

  enum GRAD_CHECK{
    CALC_GRAD, CALC_LOSS,
  };

  MaxPooling(){}

  /*
  void forward(const VecD& x, MaxPooling::State* cur);
  void backward(VecD& delx, const MaxPooling::State* cur);
  */
  void forward(const MatD& x, MaxPooling::State* cur);
  void backward(MatD& delx, const MaxPooling::State* cur);

  // for grad check
  void forward(const MatD& x, MaxPooling::State* cur, const MaxPooling::GRAD_CHECK flag);

//  static void test();
};

class MaxPooling::State{
public:
  std::vector<VecD::Index> maxIndex;
  VecD y;
  VecD dely;

  State(const int outputDim):
  maxIndex(outputDim)
  {
    this->y = VecD::Zero(outputDim);
    this->dely = VecD::Zero(outputDim);
  }
  void setZero(){
    this->dely.setZero();
  }
  void clear(){
    this->y = VecD();
    this->dely = VecD();
  }
  ~State(){
    this->clear();
  }
};

inline void MaxPooling::forward(const MatD& x, MaxPooling::State* cur){
  // cur->dely = VecD::Zero(x.cols());
  // cur->y = cur->dely;
  for(int i = 0, i_end = x.cols(); i < i_end; ++i){
    cur->y.coeffRef(i,0) = x.col(i).maxCoeff(&(cur->maxIndex[i]));
  }
  // cur->dely.setZero();// for backprop
}
inline void MaxPooling::backward(MatD& delx, const MaxPooling::State* cur){
  for(int i = 0, i_end = cur->dely.rows(); i < i_end; ++i){
    delx.coeffRef(cur->maxIndex[i], i) += cur->dely.coeffRef(i,0);
  }
}

inline void MaxPooling::forward(const MatD& x, MaxPooling::State* cur, const MaxPooling::GRAD_CHECK flag){
  // cur->dely = VecD::Zero(x.cols());
  // cur->y = cur->dely;
  if(flag == MaxPooling::CALC_GRAD){
    for(int i = 0, i_end = x.cols(); i < i_end; ++i){
      cur->y.coeffRef(i,0) = x.col(i).maxCoeff(&(cur->maxIndex[i]));
    }
  }
  else if(flag == MaxPooling::CALC_LOSS){
    for(int i = 0, i_end = x.cols(); i < i_end; ++i){
      cur->y.coeffRef(i,0) = x.coeffRef(cur->maxIndex[i], i);
    }
  }
  // cur->dely.setZero();// for backprop
}

namespace MaxPoolingGradChecker{
  inline void test(){

    const int x_size = 4;
    const int y_size = 4;

    MaxPooling mp;

    MatD x(x_size, y_size);
    x << 1,5,12,13,
    4,7,9,16,
    3,6,11,15,
    2,8,10,14;

    MatD delx = MatD::Zero(x_size, y_size);

    MaxPooling::State cur(y_size);

    std::cout << "x" << std::endl;
    std::cout << x << std::endl;

    mp.forward(x, &cur);

    std::cout << "cur.y" << std::endl;
    std::cout << cur.y << std::endl;

    cur.dely << 1,2,3,4;
    std::cout << "cur.dely" << std::endl;
    std::cout << cur.dely << std::endl;

    mp.backward(delx, &cur);

    std::cout << "delx" << std::endl;
    std::cout << delx << std::endl;
  }
};
