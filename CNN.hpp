#pragma once

#include "Padding.hpp"
#include "MaxPooling.hpp"
#include "Im2Col.hpp"
#include "Convolution.hpp"
#include "ActFunc2.hpp"
#include "GradChecker.hpp"

class CNN{
public:
  class State;
  class Grad;

  Padding pad;
  Im2Col im2col;
  Convolution conv;
  ActFunc::Tanh af;
  MaxPooling mp;

  CNN(){}
  CNN(const int inputDim, const int left, const int right, const int colBlock, const int kernelNum);
  void init(Rand& rnd, const Real scale);
  void forward(const MatD& x, CNN::State* cur);
  void backward1(MatD& delx, CNN::State* cur, CNN::Grad& grad);
  void backward2(CNN::State* cur, CNN::Grad& grad);

  // for grad check
  void forward(const MatD& x, CNN::State* cur, const MaxPooling::GRAD_CHECK flag);

  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class CNN::State{
public:
  Padding::State* pad;
  Im2Col::State* im2col;
  Convolution::State* conv;
  ActFunc::State* af;
  MaxPooling::State* mp;

  VecD* y;
  VecD* dely;

  State(){}
  State(const CNN& cnn){
    this->pad = new Padding::State;
    this->im2col = new Im2Col::State;
    this->conv = new Convolution::State;
    this->af = new ActFunc::State;
    this->mp = new MaxPooling::State(cnn.conv.W.cols());

    this->y = &this->mp->y;
    this->dely = &this->mp->dely;
  }
  void clear(){
    this->y = NULL;
    this->dely = NULL;

    delete this->pad;
    delete this->im2col;
    delete this->conv;
    delete this->af;
    delete this->mp;
  }
  ~State(){
    this->clear();
  }
  void setZero(){
    this->pad->setZero();
    this->im2col->setZero();
    this->conv->setZero();
    this->af->setZero();
    this->mp->setZero();
    this->dely->setZero();
  }
};

class CNN::Grad{
public:
  Convolution::Grad conv;

  Grad(){}
  Grad(CNN& cnn);
  void init();
  Real norm();
  void operator += (const CNN::Grad& grad);
  void l2reg(const Real lambda, const CNN& cnn);
  void l2reg(const Real lambda, const CNN& cnn, const CNN& target);
  void sgd(const Real lr, CNN& cnn);
  void adam(const Real lr, const Adam::HyperParam& hp, CNN& cnn);
  void clear(){
    this->conv.clear();
  }
};

class CNNGradChecker : public GradChecker{
public:

  CNN cnn;

  CNN::Grad grad;

  CNN::State* cur;

  MaxPooling::GRAD_CHECK flag;

  MatD x;
  MatD delx;
  VecD g;

  CNNGradChecker(const int inputDim, const int len, const int left, const int right, const int colBlock, const int kernelNum);
  Real calcLoss();
  void calcGrad();

  static void test1();
  static void test2();
};
