#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <fstream>
#include "Optimizer.hpp"
#include "Adam.hpp"//add by kinu
#include "ActFunc2.hpp"
#include "LossFunc.hpp"

#include "GradChecker.hpp"

// edit by kinugawa, 2018/01/06

class LSTM{
public:
  LSTM(){};
  LSTM(const int inputDim, const int hiddenDim, const Real dropoutRateX_);

  class State;
  class Grad;
  class AdamGrad;//add by kinu

  enum PARAM{
    WXI, WHI,
    WXF, WHF,
    WXO, WHO,
    WXU, WHU,
    BI, BF, BO, BU,
  };

  Real dropoutRateX;
  Real orgDropoutRateX;

  MatD Wxi, Whi; VecD bi; //for the input gate
  MatD Wxf, Whf; VecD bf; //for the forget gate
  MatD Wxo, Who; VecD bo; //for the output gate
  MatD Wxu, Whu; VecD bu; //for the memory cell

  // activation fuction
  ActFunc::Sigmoid sigmoid;
  ActFunc::Tanh tanh;

  void init(Rand& rnd, const Real scale = 1.0);

  void forward(const VecD& xt, LSTM::State* cur);
  void forward(const VecD& xt, VecD& prevC, VecD& prevH, LSTM::State* cur);

  void backward1(VecD& delx, LSTM::State* cur, LSTM::Grad& grad);
  void backward1(VecD& delx, VecD& delPrevC, VecD& delPrevH, LSTM::State* cur, LSTM::Grad& grad);

  template<LSTM::PARAM> void backward2(const LSTM::State* cur, LSTM::Grad& grad);

  void sgd(const LSTM::Grad& grad, const Real lr);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  void dropout(const MODE mode);
  // void dr4test();//add by kinu
  // void dr4train(const Real dropoutRateX, const Real dropoutRateH);//add by kinu
  void operator += (const LSTM& lstm);
  void operator /= (const Real val);
};

class LSTM::State{
public:
  virtual ~State() {this->clear();};

  VecD maskXt;

  VecD xt;
  VecD* prevC;
  VecD* prevH;

  VecD h, c, u, i, f, o;
  VecD cTanh;

  VecD delh, delc;//for backprop
  // VecD delx; //for backprop

  VecD delo, deli, delu, delf; //for temp backprop

  virtual void clear();

  State(const LSTM& lstm):
  prevC(0), prevH(0)
  {
    this->maskXt = VecD::Zero(lstm.Wxi.cols());
    this->delh = VecD::Zero(lstm.bi.rows());
    this->delc = VecD::Zero(lstm.bi.rows());
  }
  void setZero(){
    this->delh.setZero();
    this->delc.setZero();
  }
};

class LSTM::Grad{
public:
  Grad(): gradHist(0), adamGradHist(0){}
  Grad(const LSTM& lstm);

  LSTM::Grad* gradHist;
  LSTM::AdamGrad* adamGradHist;

  MatD Wxi, Whi; VecD bi;
  MatD Wxf, Whf; VecD bf;
  MatD Wxo, Who; VecD bo;
  MatD Wxu, Whu; VecD bu;

  void init();
  Real norm();
  void l2reg(const Real lambda, const LSTM& lstm);
  void l2reg(const Real lambda, const LSTM& lstm, const LSTM& target);
  void sgd(const Real lr, LSTM& lstm);
  void adagrad(const Real lr, LSTM& lstm, const Real initVal = 1.0);
  void momentum(const Real lr, const Real m, LSTM& lstm);
  void adam(const Real lr, const Adam::HyperParam& hp, LSTM& lstm);//add by kinu
  void fill(const Real initVal);// add by kinu
  void saveHist(std::ofstream& ofs);
  void loadHist(std::ifstream& ifs);

  void operator += (const LSTM::Grad& grad);
  void operator /= (const Real val);

  void clear();
};

class LSTM::AdamGrad{
public:
  Adam::Grad<MatD> Wxi, Whi; Adam::Grad<VecD> bi;
  Adam::Grad<MatD> Wxf, Whf; Adam::Grad<VecD> bf;
  Adam::Grad<MatD> Wxo, Who; Adam::Grad<VecD> bo;
  Adam::Grad<MatD> Wxu, Whu; Adam::Grad<VecD> bu;

  AdamGrad(const LSTM& lstm);
};

template<> inline void LSTM::backward2<LSTM::WXI>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Wxi.noalias() += cur->deli*cur->xt.transpose();
}
template<> inline void LSTM::backward2<LSTM::WXF>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Wxf.noalias() += cur->delf*cur->xt.transpose();
}
template<> inline void LSTM::backward2<LSTM::WXO>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Wxo.noalias() += cur->delo*cur->xt.transpose();
}
template<> inline void LSTM::backward2<LSTM::WXU>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Wxu.noalias() += cur->delu*cur->xt.transpose();
}
template<> inline void LSTM::backward2<LSTM::WHI>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Whi.noalias() += cur->deli*cur->prevH->transpose();
}
template<> inline void LSTM::backward2<LSTM::WHF>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Whf.noalias() += cur->delf*cur->prevH->transpose();
}
template<> inline void LSTM::backward2<LSTM::WHO>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Who.noalias() += cur->delo*cur->prevH->transpose();
}
template<> inline void LSTM::backward2<LSTM::WHU>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.Whu.noalias() += cur->delu*cur->prevH->transpose();
}
/*
template<> inline void LSTM::backward2<LSTM::BI>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.bi += cur->deli;
}
template<> inline void LSTM::backward2<LSTM::BF>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.bf += cur->delf;
}
template<> inline void LSTM::backward2<LSTM::BO>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.bo += cur->delo;
}
template<> inline void LSTM::backward2<LSTM::BU>(const LSTM::State* cur, LSTM::Grad& grad){
  grad.bu += cur->delu;
}
*/

class LSTMGradChecker : public GradChecker{
public:
  LSTM lstm;
  LossFunc::MeanSquaredError lossFunc;

  LSTM::Grad grad;

  std::vector<LSTM::State*> lstmState;
  std::vector<LossFunc::State*> lossFuncState;

  std::vector<VecD> x;
  std::vector<VecD> delx;
  std::vector<VecD> g;// gold output

  LSTMGradChecker(const int inputDim, const int hiddenDim, const int len);

  Real calcLoss();
  void calcGrad();

  static void test();
};
