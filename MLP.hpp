#pragma once

#include "ActFunc2.hpp"
#include "Adam.hpp"
#include "GradChecker.hpp"
#include "Optimizer.hpp"

class MLP{
public:
  class State;
  class Grad;
  class AdamGrad;

  class Data;

  enum PARAM{
    L0, L1, L2,
  };

  enum LAYER{
    CHILD, PARENT,
  };

  // CHILD is for sentence and paragprah
  // PARENT is for section, not using c1, l1 and f1

  // a Dropout rate of each layer is set by hand
  Real dropoutRate;
  Real orgDropoutRate;
  bool useDropout;

  std::pair<int, int> c0[2];
  MatD W0;
  VecD b0;
  ActFunc::Tanh f0;

  std::pair<int, int> c1[2];
  MatD W1;
  VecD b1;
  ActFunc::Tanh f1;

  VecD W2;
  Real b2;
  ActFunc::Sigmoid f2;

  MLP(const int inputDim00, const int inputDim01, const int hiddenDim0, const int inputDim1, const int hiddemDim1, const Real dropoutRate_);// for CHILD
  MLP(const int inputDim00, const int inputDim01, const int hiddenDim0, const Real dropoutRate_);// for PARENT
  MLP(){}

  void init(Rand& rnd, const Real scale);

  // for child
  void forward(const VecD& x0, const VecD& x1, const VecD& x2, MLP::State* cur);
  void backward1(VecD& delx0, VecD& delx1, VecD& delx2, MLP::State* cur, MLP::Grad& grad);

  // for parent
  void forward(const VecD& x0, const VecD& x1, MLP::State* cur);
  void backward1(VecD& delx0, VecD& delx1, MLP::State* cur, MLP::Grad& grad);

  template<MLP::PARAM> void backward2(const MLP::State* cur, MLP::Grad& grad);

  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

 // for dropout
  void dropout(const MODE mode);
};

class MLP::State{
public:
  VecD maskX0, x0, y0, delx0, dely0;
  VecD maskX1, x1, y1, delx1, dely1;
  VecD maskX2, x2;
  Real y2, dely2;

  // reference
  Real* y;
  Real* dely;
  VecD* yy;
  VecD* delyy;

  State(){}
  State(const MLP& mlp):
  y2(0), dely2(0)
  {
    this->maskX0 = VecD::Zero(mlp.W0.cols());
    this->x0 = VecD::Zero(mlp.W0.cols());
    this->y0 = VecD::Zero(mlp.W0.rows());
    this->delx0 = VecD::Zero(mlp.W0.cols());
    this->dely0 = VecD::Zero(mlp.W0.rows());

    this->maskX1 = VecD::Zero(mlp.W1.cols());
    this->x1 = VecD::Zero(mlp.W1.cols());
    this->y1 = VecD::Zero(mlp.W1.rows());
    this->delx1 = VecD::Zero(mlp.W1.cols());
    this->dely1 = VecD::Zero(mlp.W1.rows());

    this->maskX2 = VecD::Zero(mlp.W2.rows());
    this->x2 = VecD::Zero(mlp.W2.rows());

    // reference
    this->y = &this->y2;
    this->dely = &this->dely2;

    this->yy = &this->y1;
    this->delyy = &this->dely1;
  }
  void clear(){
    this->maskX0 = VecD();
    this->x0 = VecD();
    this->y0 = VecD();
    this->delx0 = VecD();
    this->dely0 = VecD();

    this->maskX1 = VecD();
    this->x1 = VecD();
    this->y1 = VecD();
    this->delx1 = VecD();
    this->dely1 = VecD();

    this->maskX2 = VecD();
    this->x2 = VecD();

    this->y = NULL;
    this->dely = NULL;

    this->yy = NULL;
    this->delyy = NULL;
  }
  ~State(){
    this->clear();
  }
  void setZero(){
    this->delx0.setZero();
    this->dely0.setZero();
    this->delx1.setZero();
    this->dely1.setZero();
    this->dely2 = 0;
  }
};

class MLP::Grad{
public:
  MLP::Grad* gradHist;
  MLP::AdamGrad* adamGradHist;

  MatD W0; VecD b0;
  MatD W1; VecD b1;
  VecD W2; Real b2;

  Grad():gradHist(0), adamGradHist(0){};
  Grad(MLP& mlp);
  void init();
  Real norm();
  void operator += (const MLP::Grad& grad);
  void sgd(const Real lr, MLP& mlp);
  void adam(const Real lr, const Adam::HyperParam& hp, MLP& mlp);

  void clear();
};

class MLP::AdamGrad{
public:
  Adam::Grad<MatD> W0; Adam::Grad<VecD> b0;
  Adam::Grad<MatD> W1; Adam::Grad<VecD> b1;
  Adam::Grad<VecD> W2; Adam::Grad<Real> b2;

  AdamGrad(const MLP& mlp){
    this->W0 = Adam::Grad<MatD>(mlp.W0);
    this->b0 = Adam::Grad<VecD>(mlp.b0);
    this->W1 = Adam::Grad<MatD>(mlp.W1);
    this->b1 = Adam::Grad<VecD>(mlp.b1);
    this->W2 = Adam::Grad<VecD>(mlp.W2);
    this->b2 = Adam::Grad<Real>(mlp.b2);
  }
};

template<> inline void MLP::backward2<MLP::L0>(const MLP::State* cur, MLP::Grad& grad){
  grad.W0.noalias() += cur->dely0*cur->x0.transpose();
}
template<> inline void MLP::backward2<MLP::L1>(const MLP::State* cur, MLP::Grad& grad){
  grad.W1.noalias() += cur->dely1*cur->x1.transpose();
}

class MLP::Data{
public:
  VecD x0, x1;

  VecD delx0, delx1;

  Real g;

  MLP::State* mlpState;
  LossFunc::State* lossFuncState;

  MLP::State* parent;

  Data(){}
  Data(const int inputDim, const int additionalInputDim)
  {
    this->x0 = VecD(inputDim);
    this->delx0 = VecD::Zero(inputDim);

      this->x1 = VecD(additionalInputDim);
      this->delx1 = VecD::Zero(additionalInputDim);

      this->g = 0;
    }
    void init(Rand& rnd, const Real scale){
      rnd.uniform(x0, scale);
      rnd.uniform(x1, scale);
      this->g = rnd.zero2one();
    }
};

class MLPGradChecker : public GradChecker{
public:
  MLP mlpSent;
  MLP::Grad gradSent;
  std::vector<MLP::Data*> dataSent;

  MLP mlpPar;
  MLP::Grad gradPar;
  std::vector<MLP::Data*> dataPar;

  MLP mlpSec;
  MLP::Grad gradSec;
  std::vector<MLP::Data*> dataSec;

  LossFunc::BinaryCrossEntropy lossFunc;

  MLPGradChecker(const int inputDim00, const int inputDim01, const int hiddenDim0, const int inputDim1, const int hiddenDim1, const Real dropoutRate_);

  Real calcLoss();
  void calcGrad();
  static void test();
};
