#pragma once

#include "Matrix.hpp"
#include <iostream>

// only for NLP

class Im2Col{
public:
  class State;

  int dim;
  int colBlock;
  int patchSize;

  Im2Col(){}
  Im2Col(const int dim_, const int colBlock_):dim(dim_), colBlock(colBlock_), patchSize(dim_*colBlock_){}

  void forward(const MatD& x, Im2Col::State* cur);
  void backward(MatD& delx, const Im2Col::State* cur);

//  static void test();
};

class Im2Col::State{
public:
  MatD y;
  MatD dely;
  int patchNum;
  State(){}
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

inline void Im2Col::forward(const MatD& x, Im2Col::State* cur){

  cur->patchNum = x.cols() - this->colBlock + 1;

  cur->y = MatD(cur->patchNum, this->patchSize);
  // cur->dely = MatD::Zero(cur->patchNum, this->patchSize);
  cur->dely.resize(cur->patchNum, this->patchSize);

  for(int i = 0, i_end = cur->patchNum; i < i_end; ++i){
    for(int j = i, j_end = i + this->colBlock, stride = 0; j < j_end; ++j, stride += dim){
      cur->y.row(i).segment(stride, this->dim) = x.col(j).transpose();
    }
  }
}
inline void Im2Col::backward(MatD& delx, const Im2Col::State* cur){
  for(int i = 0, i_end = cur->patchNum; i < i_end; ++i){
    for(int j = i, j_end = i + this->colBlock, stride = 0; j < j_end; ++j, stride += dim){
      delx.col(j) += cur->dely.row(i).segment(stride, this->dim).transpose();
    }
  }
}

/*
class Im2ColGradChcker{
public:
  static void test();
};
*/

namespace Im2ColGradChecker{
  inline void test(){
    const int dim = 4;
    const int col = 4;
    const int colBlock = 2;

    Im2Col im2col(dim, colBlock);

    Im2Col::State cur;

    MatD x(dim, col);
    MatD delx(dim, col);
    delx.setZero();

    int counter = 1;
    for(int i = 0; i < dim; ++i){
      for(int j = 0; j < col; ++j){
        x.coeffRef(i,j) = counter++;
      }
    }

    std::cout << "x" << std::endl;
    std::cout << x << std::endl;

    im2col.forward(x, &cur);

    std::cout << "cur.y" << std::endl;
    std::cout << cur.y << std::endl;

    cur.dely = cur.y;

    im2col.backward(delx, &cur);

    std::cout << "delx" << std::endl;
    std::cout << delx << std::endl;
  }
};
