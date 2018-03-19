#pragma once

#include "CNN.hpp"
#include "GradChecker.hpp"
#include "LossFunc.hpp"
#include <cassert>

template<int N>
class CNNS{
public:
  class State;
  class Grad;

  CNN cnn[N];

  // Concat<N> concat;

  std::pair<int,int> len[N];
  // int dims[N];

  CNNS(){}
  CNNS(const int inputDim, const int hiddenDim){

    const int flag = hiddenDim % N;
    if( flag != 0){
      std::cout << "Indivisible @ CNN" << std::endl;
      assert(flag == 0);
    }

    const int kernelNum = hiddenDim / N;

    for(int i = 1, j = 0; i <= N; ++i, j += kernelNum){
      int left;
      int right;
      if(i % 2 != 0){// odd number
        left = right = (i-1)/2;
      }
      else{// even number
        right = i/2;
        left = right - 1;
      }
      this->cnn[i-1] = CNN(inputDim, left, right, i, kernelNum);

      this->len[i-1] = std::make_pair(j, kernelNum);
      // this->dims[i-1] = kernelNum;
    }

    // this->concat = Concat<N>(dims);
  }

  void init(Rand& rnd, const Real scale);

  void forward(const MatD& x, CNNS<N>::State* cur);
  void backward1(MatD& delx, CNNS<N>::State* cur, CNNS<N>::Grad& grad);
  void backward2(const CNNS<N>::State* cur, CNNS<N>::Grad& grad);
  void backward2(const CNNS<N>::State* cur, const int i, CNNS<N>::Grad& grad);

  void load(std::ifstream& ifs);
  void save(std::ofstream& ofs);

  // for grad check
  void forward(const MatD& x, CNNS<N>::State* cur, const MaxPooling::GRAD_CHECK flag);
};

template<int N>
class CNNS<N>::State{
public:

  CNN::State* cnn[N];

  /*
  typename Concat<N>::State* concat;

  VecD* y;
  VecD* dely;
 */
 VecD y;
 VecD dely;

  State(const CNNS<N>& cnns_){

    for(int i = 0; i < N; ++i){
      this->cnn[i] = new CNN::State(cnns_.cnn[i]);
    }
    // this->concat = new Concat<N>::State(cnns_.concat);

    this->y = VecD::Zero(cnns_.len[N-1].first + cnns_.len[N-1].second);
    this->dely = this->y;

    // std::cout << "this->y.size() = " << this->y.size() << std::endl;
  }
  void clear(){
    /*
    this->y = NULL;
    this->dely = NULL;
    */
    this->y = VecD();
    this->dely = VecD();

    for(int i = 0; i < N; ++i){
      delete this->cnn[i];
    }
    // delete this->concat;
  }
  ~State(){
    this->clear();
  }
  void setZero(){
    this->dely.setZero();
    for(int i = 0; i < N; ++i){
      this->cnn[i]->setZero();
    }
  }
};

template<int N>
class CNNS<N>::Grad{
public:
  CNN::Grad cnn[N];

  Grad(){}
  Grad(CNNS<N>& cnns_);
  void init();
  Real norm();
  void operator += (const CNNS<N>::Grad& grad);
  void l2reg(const Real lambda, const CNNS<N>& cnns_);
  void l2reg(const Real lambda, const CNNS<N>& cnns_, const CNNS<N>& target);
  void sgd(const Real lr, CNNS<N>& cnns_);
  void adam(const Real lr, const Adam::HyperParam& hp, CNNS<N>& cnns_);
  void clear(){
    for(int i = 0; i < N; ++i){
      this->cnn[i].clear();
    }
  }
};

template<int N>
void CNNS<N>::init(Rand& rnd, const Real scale){
  for(int i = 0; i < N; ++i){
    this->cnn[i].init(rnd, scale);
  }
}

template<int N>
void CNNS<N>::forward(const MatD& x, CNNS<N>::State* cur){
  for(int i = 0; i < N; ++i){
    this->cnn[i].forward(x, cur->cnn[i]);
    // this->concat.forward(cur->cnn[i]->y, i, cur->concat);

    cur->y.segment(this->len[i].first, this->len[i].second) = *cur->cnn[i]->y;
  }
  cur->dely.setZero();// for backprop
}

template<int N>
void CNNS<N>::backward1(MatD& delx, CNNS<N>::State* cur, CNNS<N>::Grad& grad){
  for(int i = 0; i < N; ++i){
    // this->concat.backward(cur->cnn[i]->dely, i, cur->concat);
    *cur->cnn[i]->dely += cur->dely.segment(this->len[i].first, this->len[i].second);

    this->cnn[i].backward1(delx, cur->cnn[i], grad.cnn[i]);
  }
}

template<int N>
void CNNS<N>::backward2(const CNNS<N>::State* cur, CNNS<N>::Grad& grad){
  for(int i = 0; i < N; ++i){
    this->cnn[i].backward2(cur->cnn[i], grad.cnn[i]);
  }
}

template<int N>
void CNNS<N>::backward2(const CNNS<N>::State* cur, const int i, CNNS<N>::Grad& grad){
    this->cnn[i].backward2(cur->cnn[i], grad.cnn[i]);
}

template<int N>
void CNNS<N>::load(std::ifstream& ifs){
  for(int i = 0; i < N; ++i){
    this->cnn[i].load(ifs);
  }
}

template<int N>
void CNNS<N>::save(std::ofstream& ofs){
  for(int i = 0; i < N; ++i){
    this->cnn[i].save(ofs);
  }
}

template<int N>
void CNNS<N>::forward(const MatD& x, CNNS<N>::State* cur, const MaxPooling::GRAD_CHECK flag){
  for(int i = 0; i < N; ++i){
    // std::cout << "i = " << i << std::endl;

    this->cnn[i].forward(x, cur->cnn[i], flag);
    // this->concat.forward(cur->cnn[i]->y, i, cur->concat);

    // std::cout << "this->cnn[i].forward OK" << std::endl;

    cur->y.segment(this->len[i].first, this->len[i].second) = *cur->cnn[i]->y;

    // std::cout << "concat OK" << std::endl;
  }
  cur->dely.setZero();// for backprop
}

template<int N>
CNNS<N>::Grad::Grad(CNNS<N>& cnns_){
  for(int i = 0; i < N; ++i){
    this->cnn[i] = CNN::Grad(cnns_.cnn[i]);
  }
  this->init();
}

template<int N>
void CNNS<N>::Grad::init(){
  for(int i = 0; i < N; ++i){
    this->cnn[i].init();
  }
}

template<int N>
Real CNNS<N>::Grad::norm(){
  Real res = 0;
  for(int i = 0; i < N; ++i){
    res += this->cnn[i].norm();
  }
  return res;
}

template<int N>
void CNNS<N>::Grad::operator += (const CNNS<N>::Grad& grad){
  for(int i = 0; i < N; ++i){
    this->cnn[i] += grad.cnn[i];
  }
}

template<int N>
void CNNS<N>::Grad::l2reg(const Real lambda, const CNNS<N>& cnns_){
  for(int i = 0; i < N; ++i){
    this->cnn[i].l2reg(lambda, cnns_.cnn[i]);
  }
}
template<int N>
void CNNS<N>::Grad::l2reg(const Real lambda, const CNNS<N>& cnns_, const CNNS<N>& target){
  for(int i = 0; i < N; ++i){
    this->cnn[i].l2reg(lambda, cnns_.cnn[i], target.cnn[i]);
  }
}

template<int N>
void CNNS<N>::Grad::sgd(const Real lr, CNNS<N>& cnns_){
  for(int i = 0; i < N; ++i){
    this->cnn[i].sgd(lr, cnns_.cnn[i]);
  }
}

template<int N>
void CNNS<N>::Grad::adam(const Real lr, const Adam::HyperParam& hp, CNNS<N>& cnns_){
  for(int i = 0; i < N; ++i){
    this->cnn[i].adam(lr, hp, cnns_.cnn[i]);
  }
}

class CNNSGradChecker : public GradChecker{
public:
  CNNS<6> cnns;
  LossFunc::MeanSquaredError lf;

  CNNS<6>::Grad grad;

  CNNS<6>::State* cur;
  LossFunc::State* e;

  MaxPooling::GRAD_CHECK flag;

  MatD x;
  MatD delx;

  VecD g;

  CNNSGradChecker(const int len, const int inputDim, const int hiddenDim){
    Rand rnd;
    const Real scale = 0.05;

    this->cnns = CNNS<6>(inputDim, hiddenDim);
    this->cnns.init(rnd, scale);

    this->grad = CNNS<6>::Grad(this->cnns);

    this->cur = new CNNS<6>::State(this->cnns);
    this->e = new LossFunc::State;

    this->flag = MaxPooling::CALC_GRAD;

    this->x = MatD(inputDim, len);
    rnd.uniform(this->x, scale);

    this->delx = MatD::Zero(inputDim, len);

    this->g = VecD(hiddenDim);
    rnd.uniform(this->g, scale);
  }

  Real calcLoss(){
    this->cnns.forward(this->x, this->cur, this->flag);
    /*
    std::cout << "this->cnns.forward OK" << std::endl;
    std::cout << "this->cur->y.size() = " << this->cur->y.size() << std::endl;
    std::cout << "this->g.size() = " << this->g.size() << std::endl;
    */
    const Real loss = this->lf.forward(this->cur->y, this->g, this->e);

    // std::cout << "this->lf.forward OK" << std::endl;

    return loss;
  }
  void calcGrad(){
    const Real loss = this->calcLoss();
    std::cout << "Loss = " << loss << std::endl;

    this->lf.backward(this->cur->dely, this->g, this->e);

    this->cnns.backward1(this->delx, this->cur, this->grad);
    this->cnns.backward2(this->cur, this->grad);
  }
  static void test(){
    const int len = 10;
    const int inputDim = 50;
//    const int hiddenDim = 60;
  const int hiddenDim = 70;

    CNNSGradChecker gc(len, inputDim, hiddenDim);

    std::cout << "calcGrad" << std::endl;
    gc.flag = MaxPooling::CALC_GRAD;
    gc.calcGrad();

    /* gradient checking */
    gc.flag = MaxPooling::CALC_LOSS;

    std::cout << "x" << std::endl;
    gc.gradCheck(gc.delx, gc.x);

    for(int i = 0; i < 6; ++i){
      std::cout << "cnn[" << i << "]" << std::endl;
      std::cout << "conv.W" << std::endl;
      gc.gradCheck(gc.grad.cnn[i].conv.W, gc.cnns.cnn[i].conv.W);
      std::cout << "conv.b" << std::endl;
      gc.gradCheck(gc.grad.cnn[i]. conv.b, gc.cnns.cnn[i].conv.b);
    }
  }
};
