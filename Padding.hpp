#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"

class Padding{
public:
  class State;

  int left;
  int right;

  Padding(){}
  Padding(const int left_, const int right_):left(left_), right(right_){}

  void forward(const MatD& x, Padding::State* cur);
  void backward(MatD& delx, const Padding::State* cur);

//  static void test();
};

class Padding::State{
public:
  MatD y;
  MatD dely;

  State(){
    // std::cout << "constrocutr" << std::endl;
    this->y = MatD(1,1);
    this->dely = MatD(1,1);
  }
  ~State(){
    this->clear();
  }
  void clear(){
    this->y = MatD();
    this->dely = MatD();
  }
  void setZero(){
    this->dely.setZero();
  }
};

inline void Padding::forward(const MatD& x, Padding::State* cur){

  cur->y.resize(x.rows(), x.cols()+this->left+this->right);
  cur->y.setZero();
  cur->dely.resize(x.rows(), x.cols()+this->left+this->right);
  // cur->dely.setZero();

  cur->dely = cur->y;

  cur->y.block(0, this->left, x.rows(), x.cols()) = x;
}
inline void Padding::backward(MatD& delx, const Padding::State* cur){
  delx += cur->dely.block(0, this->left, delx.rows(), delx.cols());
}

namespace PaddingGradChecker{
  inline void test(){

    Rand rnd;
    const Real scale = 0.05;

    // set parameter
    const int left = 3;
    const int right = 4;

    Padding pad(left, right);

    // set state
    Padding::State cur;

    // set input
    MatD x(2,4);
    x << 1,2,3,4,
    5,6,7,8;

    MatD delx = MatD::Zero(2,4);

    MatD dely(2,4+left+right);
    rnd.uniform(dely, scale);

    // calc
    std::cout << "x" << std::endl;
    std::cout << x << std::endl;

    pad.forward(x, &cur);

    std::cout << "cur.y" << std::endl;
    std::cout << cur.y << std::endl;

    std::cout << "dely" << std::endl;
    std::cout << dely << std::endl;

    cur.dely += dely;
    pad.backward(delx, &cur);

    std::cout << "delx" << std::endl;
    std::cout << delx << std::endl;
  }
};
