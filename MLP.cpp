#include "MLP.hpp"
#include "Utils.hpp"

MLP::MLP(const int inputDim00, const int inputDim01, const int hiddenDim0, const int inputDim1, const int hiddemDim1, const Real dropoutRate_):
dropoutRate(dropoutRate_), orgDropoutRate(dropoutRate_)
{
  this->c0[0] = std::make_pair(0, inputDim00);
  this->c0[1] = std::make_pair(inputDim00, inputDim01);

  this->W0 = MatD(hiddenDim0, inputDim00+inputDim01);
  this->b0 = VecD::Zero(hiddenDim0);

  this->c1[0] = std::make_pair(0, hiddenDim0);
  this->c1[1] = std::make_pair(hiddenDim0, inputDim1);

  this->W1 = MatD(hiddemDim1, hiddenDim0+inputDim1);
  this->b1 = VecD::Zero(hiddemDim1);

  this->W2 = VecD(hiddemDim1);
  this->b2 = 0;
}

MLP::MLP(const int inputDim00, const int inputDim01, const int hiddenDim0, const Real dropoutRate_):
dropoutRate(dropoutRate_), orgDropoutRate(dropoutRate_)
{

  /* dummy */
  this->c0[0] = std::make_pair(0, 1);
  this->c0[1] = std::make_pair(1, 1);

  this->W0 = MatD(1, 1);
  this->b0 = VecD::Zero(1);
  /* dummy */

  this->c1[0] = std::make_pair(0, inputDim00);
  this->c1[1] = std::make_pair(inputDim00, inputDim01);

  this->W1 = MatD(hiddenDim0, inputDim00+inputDim01);
  this->b1 = VecD::Zero(hiddenDim0);

  this->W2 = VecD(hiddenDim0);
  this->b2 = 0;
}

void MLP::init(Rand& rnd, const Real scale){
  rnd.uniform(this->W0, scale);
  rnd.uniform(this->W1, scale);
  rnd.uniform(this->W2, scale);
}

void MLP::forward(const VecD& x0, const VecD& x1, const VecD& x2, MLP::State* cur){
// x2 is provided by other MLP
  if(this->dropoutRate > 0){
    // dropout ON
    cur->x0.segment(this->c0[0].first, this->c0[0].second) = x0;
    cur->x0.segment(this->c0[1].first, this->c0[1].second) = x1;
    cur->x0.array() *= cur->maskX0.array();

    cur->y0 = this->b0;
    cur->y0.noalias() += this->W0*cur->x0;
    this->f0.forward(cur->y0);

    cur->x1.segment(this->c1[0].first, this->c1[0].second) = cur->y0;
    cur->x1.segment(this->c1[1].first, this->c1[1].second) = x2;
    cur->x1.array() *= cur->maskX1.array();

    cur->y1 = this->b1;
    cur->y1.noalias() += this->W1*cur->x1;
    this->f1.forward(cur->y1);

    cur->x2.array() = cur->y1.array()*cur->maskX2.array();
    cur->y2 = this->b2;
    cur->y2 += this->W2.dot(cur->x2);
    this->f2.forward(cur->y2);
  }
  else{
    // dropout OFF
    cur->x0.segment(this->c0[0].first, this->c0[0].second) = x0;
    cur->x0.segment(this->c0[1].first, this->c0[1].second) = x1;

    cur->y0 = this->b0;
    cur->y0.noalias() += this->W0*cur->x0;
    this->f0.forward(cur->y0);

    cur->x1.segment(this->c1[0].first, this->c1[0].second) = cur->y0;
    cur->x1.segment(this->c1[1].first, this->c1[1].second) = x2;

    cur->y1 = this->b1;
    cur->y1.noalias() += this->W1*cur->x1;
    this->f1.forward(cur->y1);

    cur->x2 = cur->y1;
    cur->y2 = this->b2;
    cur->y2 += this->W2.dot(cur->x2);
    this->f2.forward(cur->y2);
  }
}

void MLP::backward1(VecD& delx0, VecD& delx1, VecD& delx2, MLP::State* cur, MLP::Grad& grad){
  // cur->dely2 is given
  if(this->dropoutRate > 0){
    // dropout ON
    cur->dely2 *= this->f2.backward(cur->y2);

    grad.W2.noalias() += cur->dely2*cur->x2;
    grad.b2 += cur->dely2;

    cur->dely1.array() += (cur->dely2*this->W2).array()*cur->maskX2.array();

    cur->dely1.array() *= this->f1.backward(cur->y1).array();

    cur->delx1.array() += (this->W1.transpose()*cur->dely1).array()*cur->maskX1.array();
    cur->dely0 += cur->delx1.segment(this->c1[0].first, this->c1[0].second);
    delx2 += cur->delx1.segment(this->c1[1].first, this->c1[1].second);
    grad.b1 += cur->dely1;

    cur->dely0.array() *= this->f0.backward(cur->y0).array();
    cur->delx0.array() += (this->W0.transpose()*cur->dely0).array()*cur->maskX0.array();
    grad.b0 += cur->dely0;

    delx0 += cur->delx0.segment(this->c0[0].first, this->c0[0].second);
    delx1 += cur->delx0.segment(this->c0[1].first, this->c0[1].second);
  }
  else{
    // dropout OFF
    cur->dely2 *= this->f2.backward(cur->y2);

    grad.W2.noalias() += cur->dely2*cur->x2;
    grad.b2 += cur->dely2;

    cur->dely1.noalias() += cur->dely2*this->W2;

    cur->dely1.array() *= this->f1.backward(cur->y1).array();

    cur->delx1.noalias() += this->W1.transpose()*cur->dely1;
    cur->dely0 += cur->delx1.segment(this->c1[0].first, this->c1[0].second);
    delx2 += cur->delx1.segment(this->c1[1].first, this->c1[1].second);
    grad.b1 += cur->dely1;

    cur->dely0.array() *= this->f0.backward(cur->y0).array();
    cur->delx0.noalias() += this->W0.transpose()*cur->dely0;
    grad.b0 += cur->dely0;

    delx0 += cur->delx0.segment(this->c0[0].first, this->c0[0].second);
    delx1 += cur->delx0.segment(this->c0[1].first, this->c0[1].second);
  }
}

void MLP::forward(const VecD& x0, const VecD& x1, MLP::State* cur){
  if(this->dropoutRate > 0){
    // dropout ON
    cur->x1.segment(this->c1[0].first, this->c1[0].second) = x0;
    cur->x1.segment(this->c1[1].first, this->c1[1].second) = x1;
    cur->x1.array() *= cur->maskX1.array();

    cur->y1 = this->b1;
    cur->y1.noalias() += this->W1*cur->x1;
    this->f1.forward(cur->y1);

    cur->x2.array() = cur->y1.array()*cur->maskX2.array();
    cur->y2 = this->b2;
    cur->y2 += this->W2.dot(cur->x2);
    this->f2.forward(cur->y2);
  }
  else{
    // dropout OFF
    cur->x1.segment(this->c1[0].first, this->c1[0].second) = x0;
    cur->x1.segment(this->c1[1].first, this->c1[1].second) = x1;

    cur->y1 = this->b1;
    cur->y1.noalias() += this->W1*cur->x1;
    this->f1.forward(cur->y1);

    cur->x2 = cur->y1;
    cur->y2 = this->b2;
    cur->y2 += this->W2.dot(cur->x2);
    this->f2.forward(cur->y2);
  }
}

void MLP::backward1(VecD& delx0, VecD& delx1, MLP::State* cur, MLP::Grad& grad){
  // cur->dely2 is given
  if(this->dropoutRate > 0){
    // dropout ON
    cur->dely2 *= this->f2.backward(cur->y2);

    grad.W2.noalias() += cur->dely2*cur->x2;
    grad.b2 += cur->dely2;

    cur->dely1.array() += (cur->dely2*this->W2).array()*cur->maskX2.array();

    cur->dely1.array() *= this->f1.backward(cur->y1).array();

    cur->delx1.array() += (this->W1.transpose()*cur->dely1).array()*cur->maskX1.array();
    delx0 += cur->delx1.segment(this->c1[0].first, this->c1[0].second);
    delx1 += cur->delx1.segment(this->c1[1].first, this->c1[1].second);
    grad.b1 += cur->dely1;
  }
  else{
    // dropout OFF
    cur->dely2 *= this->f2.backward(cur->y2);

    grad.W2.noalias() += cur->dely2*cur->x2;
    grad.b2 += cur->dely2;

    cur->dely1.noalias() += cur->dely2*this->W2;

    cur->dely1.array() *= this->f1.backward(cur->y1).array();

    cur->delx1.noalias() += this->W1.transpose()*cur->dely1;
    delx0 += cur->delx1.segment(this->c1[0].first, this->c1[0].second);
    delx1 += cur->delx1.segment(this->c1[1].first, this->c1[1].second);
    grad.b1 += cur->dely1;
  }
}

void MLP::save(std::ofstream& ofs){
  Utils::save(ofs, this->W0);
  Utils::save(ofs, this->b0);
  Utils::save(ofs, this->W1);
  Utils::save(ofs, this->b1);
  Utils::save(ofs, this->W2);
  Utils::save(ofs, this->b2);
}

void MLP::load(std::ifstream& ifs){
  Utils::load(ifs, this->W0);
  Utils::load(ifs, this->b0);
  Utils::load(ifs, this->W1);
  Utils::load(ifs, this->b1);
  Utils::load(ifs, this->W2);
  Utils::load(ifs, this->b2);
}

void MLP::dropout(const MODE mode){
  if(mode == TEST){
    // training -> testing
    if(this->dropoutRate <= 0.0) {
      this->dropoutRate = 1.0;
    }

    this->W0 *= this->dropoutRate;
    this->W1 *= this->dropoutRate;
    this->W2 *= this->dropoutRate;

    this->dropoutRate = -1.0;
  }
  else if(mode == TRAIN){
    // testing -> training
    if(this->orgDropoutRate > 0.0){
      this->dropoutRate = this->orgDropoutRate;
    }
    else{
      this->dropoutRate = 1.0;
    }

      this->W0 *= 1/this->dropoutRate;
      this->W1 *= 1/this->dropoutRate;
      this->W2 *= 1/this->dropoutRate;

    if(this->orgDropoutRate <= 0.0){
      this->dropoutRate = this->orgDropoutRate;
    }
  }
}

MLP::Grad::Grad(MLP& mlp)
:gradHist(0), adamGradHist(0)
{
  this->W0 = MatD(mlp.W0.rows(), mlp.W0.cols());
  this->b0 = VecD(mlp.b0.rows());

  this->W1 = MatD(mlp.W1.rows(), mlp.W1.cols());
  this->b1 = VecD(mlp.b1.rows());

  this->W2 = VecD(mlp.W2.rows());
  this->b2 = 0;

  this->init();
}
void MLP::Grad::init(){
  this->W0.setZero();
  this->b0.setZero();
  this->W1.setZero();
  this->b1.setZero();
  this->W2.setZero();
  this->b2 = 0;
}
Real MLP::Grad::norm(){
  Real res = 0;
  res += this->W0.squaredNorm();
  res += this->b0.squaredNorm();
  res += this->W1.squaredNorm();
  res += this->b1.squaredNorm();
  res += this->W2.squaredNorm();
  res += this->b2*this->b2;
  return res;
}
void MLP::Grad::operator += (const MLP::Grad& grad){
  this->W0 += grad.W0;
  this->b0 += grad.b0;
  this->W1 += grad.W1;
  this->b1 += grad.b1;
  this->W2 += grad.W2;
  this->b2 += grad.b2;
}
void MLP::Grad::sgd(const Real lr, MLP& mlp){
  Optimizer::sgd(this->W0, lr, mlp.W0);
  Optimizer::sgd(this->b0, lr, mlp.b0);
  Optimizer::sgd(this->W1, lr, mlp.W1);
  Optimizer::sgd(this->b1, lr, mlp.b1);
  Optimizer::sgd(this->W2, lr, mlp.W2);
  Optimizer::sgd(this->b2, lr, mlp.b2);
}
void MLP::Grad::adam(const Real lr, const Adam::HyperParam& hp, MLP& mlp){
  if (this->adamGradHist == 0){
    this->adamGradHist = new MLP::AdamGrad(mlp);
  }
  Adam::adam(this->W0, lr, hp, this->adamGradHist->W0, mlp.W0);
  Adam::adam(this->b0, lr, hp, this->adamGradHist->b0, mlp.b0);
  Adam::adam(this->W1, lr, hp, this->adamGradHist->W1, mlp.W1);
  Adam::adam(this->b1, lr, hp, this->adamGradHist->b1, mlp.b1);
  Adam::adam(this->W2, lr, hp, this->adamGradHist->W2, mlp.W2);
  Adam::adam(this->b2, lr, hp, this->adamGradHist->b2, mlp.b2);
}
void MLP::Grad::clear(){
  if(this->gradHist != 0){
    delete this->gradHist;
    this->gradHist = NULL;
    std::cout << "MLP::Grad->gradHist clear" << std::endl;// for checking
  }
  if(this->adamGradHist != 0){
    delete this->adamGradHist;
    this->adamGradHist = NULL;
    std::cout << "MLP::Grad->adamGradHist clear" << std::endl;// for checking
  }
}

MLPGradChecker::MLPGradChecker(const int inputDim00, const int inputDim01, const int hiddenDim0, const int inputDim1, const int hiddemDim1, const Real dropoutRate_){
  Rand rnd;
  const Real scale = 0.05;

  this->mlpSent = MLP(inputDim00, inputDim01, hiddenDim0, inputDim1, hiddemDim1, dropoutRate_);// for CHILD
  this->mlpSent.init(rnd, scale);
  this->gradSent = MLP::Grad(this->mlpSent);

  this->mlpPar = MLP(inputDim00, inputDim01, hiddenDim0, inputDim1, hiddemDim1, dropoutRate_);// for CHILD
  this->mlpPar.init(rnd, scale);
  this->gradPar = MLP::Grad(this->mlpPar);

  this->mlpSec = MLP(inputDim00, inputDim01, hiddenDim0, dropoutRate_);// for PARENT
  this->mlpSec.init(rnd, scale);
  this->gradSec = MLP::Grad(this->mlpSec);

//  std::cout << "this->mlpSec.l3.W.rows() = " << this->mlpSec.l3.W.rows() << ", this->mlpSec.l3.W.cols() = " << this->mlpSec.l3.W.cols() << std::endl;
//  std::cout << "this->gradSec.l3.W.rows() = " << this->gradSec.l3.W.rows() << ", this->gradSec.l3.W.cols() = " << this->gradSec.l3.W.cols() << std::endl;

  for(int i = 0; i < 2; ++i){
    this->dataSec.push_back(new MLP::Data(inputDim00, inputDim01));
    this->dataSec.back()->init(rnd, scale);

    this->dataSec.back()->mlpState = new MLP::State(this->mlpSec);
    rnd.setMask(this->dataSec.back()->mlpState->maskX0, dropoutRate_);
    rnd.setMask(this->dataSec.back()->mlpState->maskX1, dropoutRate_);
    rnd.setMask(this->dataSec.back()->mlpState->maskX2, dropoutRate_);

    this->dataSec.back()->lossFuncState = new LossFunc::State;

    this->dataSec.back()->parent = NULL;
  }
  for(int i = 0; i < 4; ++i){
    this->dataPar.push_back(new MLP::Data(inputDim00, inputDim01));
    this->dataPar.back()->init(rnd, scale);

    this->dataPar.back()->mlpState = new MLP::State(this->mlpPar);

    rnd.setMask(this->dataPar.back()->mlpState->maskX0, dropoutRate_);
    rnd.setMask(this->dataPar.back()->mlpState->maskX1, dropoutRate_);
    rnd.setMask(this->dataPar.back()->mlpState->maskX2, dropoutRate_);

    this->dataPar.back()->lossFuncState = new LossFunc::State;

    if(i < 2){
      this->dataPar[i]->parent = this->dataSec[0]->mlpState;
    }
    else{
      this->dataPar[i]->parent = this->dataSec[1]->mlpState;
    }
  }

  for(int i = 0; i < 6; ++i){
    this->dataSent.push_back(new MLP::Data(inputDim00, inputDim01));
    this->dataSent.back()->init(rnd, scale);

    this->dataSent.back()->mlpState = new MLP::State(this->mlpSent);

    rnd.setMask(this->dataSent.back()->mlpState->maskX0, dropoutRate_);
    rnd.setMask(this->dataSent.back()->mlpState->maskX1, dropoutRate_);
    rnd.setMask(this->dataSent.back()->mlpState->maskX2, dropoutRate_);

    this->dataSent.back()->lossFuncState = new LossFunc::State;

    if(i == 0){
      this->dataSent[i]->parent = this->dataPar[0]->mlpState;
    }
    else if(i == 1 || i == 2){
      this->dataSent[i]->parent = this->dataPar[1]->mlpState;
    }
    else if(i == 3 || i == 4){
      this->dataSent[i]->parent = this->dataPar[2]->mlpState;
    }
    else{
      this->dataSent[i]->parent = this->dataPar[3]->mlpState;
    }
  }

}

Real MLPGradChecker::calcLoss(){
  Real lossC = 0, lossP = 0, lossS = 0;

  for(int i = 0; i < 2; ++i){
    MLP::Data* data = this->dataSec[i];
    this->mlpSec.forward(data->x0, data->x1, data->mlpState);
    lossC += this->lossFunc.forward(*data->mlpState->y, data->g, data->lossFuncState);
  }

  for(int i = 0; i < 4; ++i){
    MLP::Data* data = this->dataPar[i];
    this->mlpPar.forward(data->x0, data->x1, *data->parent->yy, data->mlpState);
    lossP += this->lossFunc.forward(*data->mlpState->y, data->g, data->lossFuncState);
  }

  for(int i = 0; i < 6; ++i){
    MLP::Data* data = this->dataSent[i];
    this->mlpSent.forward(data->x0, data->x1, *data->parent->yy, data->mlpState);
    lossS += this->lossFunc.forward(*data->mlpState->y, data->g, data->lossFuncState);
  }

  return lossC + lossP + lossS;

}
void MLPGradChecker::calcGrad(){
  std::cout << "In calcGrad" << std::endl;

  const Real loss = this->calcLoss();

  std::cout << "LOSS = " << loss << std::endl;

  std::cout << "forward OK" << std::endl;

  // setZero
  for(int i = 0; i < 2; ++i){
    this->dataSec[i]->mlpState->setZero();
  }
  for(int i = 0; i < 4; ++i){
    this->dataPar[i]->mlpState->setZero();
  }
  for(int i = 0; i < 6; ++i){
    this->dataSent[i]->mlpState->setZero();
  }

  std::cout << "setZero OK" << std::endl;

  // backward1
  for(int i = 0; i < 6; ++i){
    MLP::Data* data = this->dataSent[i];
    this->lossFunc.backward(*data->mlpState->dely, data->g, data->lossFuncState);
    this->mlpSent.backward1(data->delx0, data->delx1, *data->parent->delyy, data->mlpState, this->gradSent);
  }
  std::cout << "backward1 on sent layer" << std::endl;

  for(int i = 0; i < 4; ++i){
    MLP::Data* data = this->dataPar[i];
    this->lossFunc.backward(*data->mlpState->dely, data->g, data->lossFuncState);
    this->mlpPar.backward1(data->delx0, data->delx1, *data->parent->delyy, data->mlpState, this->gradPar);
  }
  std::cout << "backward1 on par layer" << std::endl;

  for(int i = 0; i < 2; ++i){
    MLP::Data* data = this->dataSec[i];
    this->lossFunc.backward(*data->mlpState->dely, data->g, data->lossFuncState);
    this->mlpSec.backward1(data->delx0, data->delx1, data->mlpState, this->gradSec);
  }
  std::cout << "backward1 on sec layer" << std::endl;

  std::cout << "backward1 OK" << std::endl;

  // backward2
  for(int i = 0; i < 2; ++i){
    this->mlpSec.backward2<MLP::L1>(this->dataSec[i]->mlpState, this->gradSec);
  }

  for(int i = 0; i < 4; ++i){
    this->mlpPar.backward2<MLP::L0>(this->dataPar[i]->mlpState, this->gradPar);
  }
  for(int i = 0; i < 4; ++i){
    this->mlpPar.backward2<MLP::L1>(this->dataPar[i]->mlpState, this->gradPar);
  }

  for(int i = 0; i < 6; ++i){
    this->mlpSent.backward2<MLP::L0>(this->dataSent[i]->mlpState, this->gradSent);
  }
  for(int i = 0; i < 6; ++i){
    this->mlpSent.backward2<MLP::L1>(this->dataSent[i]->mlpState, this->gradSent);
  }

  std::cout << "backward2 OK" << std::endl;

}
void MLPGradChecker::test(){
  std::cout << "Start" << std::endl;

  const int inputDim11 = 40;
  const int inputDim12 = 50;

  const int hiddenDim1 = inputDim11 + inputDim12;

  const int inputDim2 = hiddenDim1;
  const int hiddenDim2 = hiddenDim1;

  const Real dropoutRate_ = 0.8;
  // const Real dropoutRate_ = -1.0;

  MLPGradChecker gc(inputDim11, inputDim12, hiddenDim1, inputDim2, hiddenDim2, dropoutRate_);

  std::cout << "Constructor OK" << std::endl;

  gc.calcGrad();

  std::cout << "calcGrad OK" << std::endl;

  std::cout << "Section" << std::endl;
  std::cout << "W2" << std::endl;
  gc.gradCheck(gc.gradSec.W2, gc.mlpSec.W2);
  std::cout << "b2" << std::endl;
  gc.gradCheck(gc.gradSec.b2, gc.mlpSec.b2);

  std::cout << "W1" << std::endl;
  gc.gradCheck(gc.gradSec.W1, gc.mlpSec.W1);
  std::cout << "b1" << std::endl;
  gc.gradCheck(gc.gradSec.b1, gc.mlpSec.b1);

  std::cout << "x" << std::endl;
  for(int i = 0; i < 2; ++i){
    gc.gradCheck(gc.dataSec[i]->delx0, gc.dataSec[i]->x0);
    gc.gradCheck(gc.dataSec[i]->delx1, gc.dataSec[i]->x1);
  }

  std::cout << "Pargraph" << std::endl;

  std::cout << "W2" << std::endl;
  gc.gradCheck(gc.gradPar.W2, gc.mlpPar.W2);
  std::cout << "b2" << std::endl;
  gc.gradCheck(gc.gradPar.b2, gc.mlpPar.b2);

  std::cout << "W1" << std::endl;
  gc.gradCheck(gc.gradPar.W1, gc.mlpPar.W1);
  std::cout << "b1" << std::endl;
  gc.gradCheck(gc.gradPar.b1, gc.mlpPar.b1);

  std::cout << "W0" << std::endl;
  gc.gradCheck(gc.gradPar.W0, gc.mlpPar.W0);
  std::cout << "b0" << std::endl;
  gc.gradCheck(gc.gradPar.b0, gc.mlpPar.b0);

  std::cout << "x" << std::endl;
  for(int i = 0; i < 4; ++i){
    gc.gradCheck(gc.dataPar[i]->delx0, gc.dataPar[i]->x0);
    gc.gradCheck(gc.dataPar[i]->delx1, gc.dataPar[i]->x1);
  }

  std::cout << "Sentence" << std::endl;

  std::cout << "W2" << std::endl;
  gc.gradCheck(gc.gradSent.W2, gc.mlpSent.W2);
  std::cout << "b2" << std::endl;
  gc.gradCheck(gc.gradSent.b2, gc.mlpSent.b2);

  std::cout << "W1" << std::endl;
  gc.gradCheck(gc.gradSent.W1, gc.mlpSent.W1);
  std::cout << "b1" << std::endl;
  gc.gradCheck(gc.gradSent.b1, gc.mlpSent.b1);

  std::cout << "W0" << std::endl;
  gc.gradCheck(gc.gradSent.W0, gc.mlpSent.W0);
  std::cout << "b0" << std::endl;
  gc.gradCheck(gc.gradSent.b0, gc.mlpSent.b0);

  std::cout << "x" << std::endl;
  for(int i = 0; i < 6; ++i){
    gc.gradCheck(gc.dataSent[i]->delx0, gc.dataSent[i]->x0);
    gc.gradCheck(gc.dataSent[i]->delx1, gc.dataSent[i]->x1);
  }
}
