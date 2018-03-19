#include "LSTM.hpp"
#include "Utils.hpp"
#include "Optimizer.hpp"

// edit by kinugawa, 2017/12/03

LSTM::LSTM(const int inputDim, const int hiddenDim, const Real dropoutRateX_):
dropoutRateX(dropoutRateX_), orgDropoutRateX(dropoutRateX_)
{
  this->Wxi = MatD(hiddenDim, inputDim);
  this->Whi = MatD(hiddenDim, hiddenDim);
  this->bi = VecD::Zero(hiddenDim);

  this->Wxf = MatD(hiddenDim, inputDim);
  this->Whf = MatD(hiddenDim, hiddenDim);
  //this->bf = VecD::Zero(hiddenDim);
  this->bf = VecD::Ones(hiddenDim); // by kinu

  this->Wxo = MatD(hiddenDim, inputDim);
  this->Who = MatD(hiddenDim, hiddenDim);
  this->bo = VecD::Zero(hiddenDim);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->Whu = MatD(hiddenDim, hiddenDim);
  this->bu = VecD::Zero(hiddenDim);
}

void LSTM::init(Rand& rnd, const Real scale){
  rnd.uniform(this->Wxi, scale);
  rnd.uniform(this->Whi, scale);

  rnd.uniform(this->Wxf, scale);
  rnd.uniform(this->Whf, scale);

  rnd.uniform(this->Wxo, scale);
  rnd.uniform(this->Who, scale);

  rnd.uniform(this->Wxu, scale);
  rnd.uniform(this->Whu, scale);
}

void LSTM::forward(const VecD& xt_, LSTM::State* cur){
  // top of the sequnece
  if(this->dropoutRateX > 0){
    cur->xt = xt_.array()*cur->maskXt.array();
  }
  else{
    cur->xt = xt_;
  }

  cur->i = this->bi;
  cur->i.noalias() += this->Wxi*cur->xt;
  this->sigmoid.forward(cur->i);

  cur->o = this->bo;
  cur->o.noalias() += this->Wxo*cur->xt;
  this->sigmoid.forward(cur->o);

  cur->u = this->bu;
  cur->u.noalias() += this->Wxu*cur->xt;
  this->tanh.forward(cur->u);

  cur->c = cur->i.array()*cur->u.array();
  cur->cTanh = cur->c;
  this->tanh.forward(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LSTM::forward(const VecD& xt_, VecD& prevC, VecD& prevH, LSTM::State* cur){
  if(this->dropoutRateX > 0){
    cur->xt = xt_.array()*cur->maskXt.array();
  }
  else{
    cur->xt = xt_;
  }

  cur->prevC = &prevC;
  cur->prevH = &prevH;

  cur->i = this->bi;
  cur->i.noalias() += this->Wxi*cur->xt + this->Whi*prevH;
  this->sigmoid.forward(cur->i);

  cur->f = this->bf;
  cur->f.noalias() += this->Wxf*cur->xt + this->Whf*prevH;
  this->sigmoid.forward(cur->f);

  cur->o = this->bo;
  cur->o.noalias() += this->Wxo*cur->xt + this->Who*prevH;
  this->sigmoid.forward(cur->o);

  cur->u = this->bu;
  cur->u.noalias() += this->Wxu*cur->xt + this->Whu*prevH;
  this->tanh.forward(cur->u);

  cur->c = cur->i.array()*cur->u.array() + cur->f.array()*prevC.array();
  cur->cTanh = cur->c;
  this->tanh.forward(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LSTM::backward1(VecD& delx, VecD& delPrevC, VecD& delPrevH, LSTM::State* cur, LSTM::Grad& grad){

  cur->delc.array() += this->tanh.backward(cur->cTanh).array()*cur->delh.array()*cur->o.array();
  delPrevC.array() += cur->delc.array()*cur->f.array();
  cur->delo = this->sigmoid.backward(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  cur->deli = this->sigmoid.backward(cur->i).array()*cur->delc.array()*cur->u.array();
  cur->delf = this->sigmoid.backward(cur->f).array()*cur->delc.array()*cur->prevC->array();
  cur->delu = this->tanh.backward(cur->u).array()*cur->delc.array()*cur->i.array();

  if(this->dropoutRateX > 0){
    delx.array() += cur->maskXt.array()*
    (this->Wxi.transpose()*cur->deli+
    this->Wxf.transpose()*cur->delf+
    this->Wxo.transpose()*cur->delo+
    this->Wxu.transpose()*cur->delu).array();
  }
  else{
    delx.noalias() +=
    this->Wxi.transpose()*cur->deli+
    this->Wxf.transpose()*cur->delf+
    this->Wxo.transpose()*cur->delo+
    this->Wxu.transpose()*cur->delu;
  }

  delPrevH.noalias() +=
  this->Whi.transpose()*cur->deli+
  this->Whf.transpose()*cur->delf+
  this->Who.transpose()*cur->delo+
  this->Whu.transpose()*cur->delu;

  grad.bi += cur->deli;
  grad.bf += cur->delf;
  grad.bo += cur->delo;
  grad.bu += cur->delu;
}
void LSTM::backward1(VecD& delx, LSTM::State* cur, LSTM::Grad& grad){

  cur->delc.array() += this->tanh.backward(cur->cTanh).array()*cur->delh.array()*cur->o.array();
  cur->delo = this->sigmoid.backward(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  cur->deli = this->sigmoid.backward(cur->i).array()*cur->delc.array()*cur->u.array();
  cur->delu = this->tanh.backward(cur->u).array()*cur->delc.array()*cur->i.array();

  if(this->dropoutRateX > 0){
    delx.array() += cur->maskXt.array()*
    (this->Wxi.transpose()*cur->deli+
    this->Wxo.transpose()*cur->delo+
    this->Wxu.transpose()*cur->delu).array();
  }
  else{
    delx.noalias() +=
    this->Wxi.transpose()*cur->deli+
    this->Wxo.transpose()*cur->delo+
    this->Wxu.transpose()*cur->delu;
  }

  grad.bi += cur->deli;
  grad.bo += cur->delo;
  grad.bu += cur->delu;
}

void LSTM::sgd(const LSTM::Grad& grad, const Real lr){
  this->Wxi -= lr*grad.Wxi;
  this->Whi -= lr*grad.Whi;
  this->bi -= lr*grad.bi;

  this->Wxf -= lr*grad.Wxf;
  this->Whf -= lr*grad.Whf;
  this->bf -= lr*grad.bf;

  this->Wxo -= lr*grad.Wxo;
  this->Who -= lr*grad.Who;
  this->bo -= lr*grad.bo;

  this->Wxu -= lr*grad.Wxu;
  this->Whu -= lr*grad.Whu;
  this->bu -= lr*grad.bu;
}

void LSTM::save(std::ofstream& ofs){
  Utils::save(ofs, this->Wxi); Utils::save(ofs, this->Whi); Utils::save(ofs, this->bi);
  Utils::save(ofs, this->Wxf); Utils::save(ofs, this->Whf); Utils::save(ofs, this->bf);
  Utils::save(ofs, this->Wxo); Utils::save(ofs, this->Who); Utils::save(ofs, this->bo);
  Utils::save(ofs, this->Wxu); Utils::save(ofs, this->Whu); Utils::save(ofs, this->bu);
}

void LSTM::load(std::ifstream& ifs){
  Utils::load(ifs, this->Wxi); Utils::load(ifs, this->Whi); Utils::load(ifs, this->bi);
  Utils::load(ifs, this->Wxf); Utils::load(ifs, this->Whf); Utils::load(ifs, this->bf);
  Utils::load(ifs, this->Wxo); Utils::load(ifs, this->Who); Utils::load(ifs, this->bo);
  Utils::load(ifs, this->Wxu); Utils::load(ifs, this->Whu); Utils::load(ifs, this->bu);
}

void LSTM::dropout(const MODE mode){
  if (mode == TEST){
    if(this->dropoutRateX <= 0.0){
      this->dropoutRateX = 1.0;
    }
      this->Wxi *= this->dropoutRateX;
      this->Wxf *= this->dropoutRateX;
      this->Wxo *= this->dropoutRateX;
      this->Wxu *= this->dropoutRateX;

      this->dropoutRateX = -1.0;
  }
  else if(mode == TRAIN){
  if(this->orgDropoutRateX > 0.0){
    this->dropoutRateX = this->orgDropoutRateX;
  }else{
    this->dropoutRateX = 1.0;
  }

    this->Wxi *= 1.0/this->dropoutRateX;
    this->Wxf *= 1.0/this->dropoutRateX;
    this->Wxo *= 1.0/this->dropoutRateX;
    this->Wxu *= 1.0/this->dropoutRateX;

    if(this->orgDropoutRateX <= 0.0){
      this->dropoutRateX = this->orgDropoutRateX;
    }
  }
}

/*
void LSTM::dr4test(){
  if(this->dropoutRateX <= 0.0){
    this->dropoutRateX = 1.0;
  }
  this->dropout(true);
  this->dropoutRateX = -1.0;
}

void LSTM::dr4train(const Real dropoutRateX){
  if(dropoutRateX > 0.0){
    this->dropoutRateX = dropoutRateX;
  }else{
    this->dropoutRateX = 1.0;
  }
  this->dropout(false);
  if(dropoutRateX <= 0.0){
    this->dropoutRateX = dropoutRateX;
  }
}
*/
void LSTM::operator += (const LSTM& lstm){
  this->Wxi += lstm.Wxi; this->Wxf += lstm.Wxf; this->Wxo += lstm.Wxo; this->Wxu += lstm.Wxu;
  this->Whi += lstm.Whi; this->Whf += lstm.Whf; this->Who += lstm.Who; this->Whu += lstm.Whu;
  this->bi += lstm.bi; this->bf += lstm.bf; this->bo += lstm.bo; this->bu += lstm.bu;
}

void LSTM::operator /= (const Real val){
  this->Wxi /= val; this->Wxf /= val; this->Wxo /= val; this->Wxu /= val;
  this->Whi /= val; this->Whf /= val; this->Who /= val; this->Whu /= val;
  this->bi /= val; this->bf /= val; this->bo /= val; this->bu /= val;
}

void LSTM::State::clear(){
  this->maskXt = VecD();

  this->xt = VecD();
  this->prevH = 0;
  this->prevC = 0;
  this->h = VecD();
  this->c = VecD();
  this->u = VecD();
  this->i = VecD();
  this->f = VecD();
  this->o = VecD();
  this->cTanh = VecD();

  this->delh = VecD();
  this->delc = VecD();
  // this->delx = VecD();
  this->deli = VecD();
  this->delf = VecD();
  this->delo = VecD();
  this->delu = VecD();
}

LSTM::Grad::Grad(const LSTM& lstm):
gradHist(0), adamGradHist(0)
{
  this->Wxi = MatD::Zero(lstm.Wxi.rows(), lstm.Wxi.cols());
  this->Whi = MatD::Zero(lstm.Whi.rows(), lstm.Whi.cols());
  this->bi = VecD::Zero(lstm.bi.rows());

  this->Wxf = MatD::Zero(lstm.Wxf.rows(), lstm.Wxf.cols());
  this->Whf = MatD::Zero(lstm.Whf.rows(), lstm.Whf.cols());
  this->bf = VecD::Zero(lstm.bf.rows());

  this->Wxo = MatD::Zero(lstm.Wxo.rows(), lstm.Wxo.cols());
  this->Who = MatD::Zero(lstm.Who.rows(), lstm.Who.cols());
  this->bo = VecD::Zero(lstm.bo.rows());

  this->Wxu = MatD::Zero(lstm.Wxu.rows(), lstm.Wxu.cols());
  this->Whu = MatD::Zero(lstm.Whu.rows(), lstm.Whu.cols());
  this->bu = VecD::Zero(lstm.bu.rows());
};

void LSTM::Grad::init(){
  this->Wxi.setZero(); this->Whi.setZero(); this->bi.setZero();
  this->Wxf.setZero(); this->Whf.setZero(); this->bf.setZero();
  this->Wxo.setZero(); this->Who.setZero(); this->bo.setZero();
  this->Wxu.setZero(); this->Whu.setZero(); this->bu.setZero();
}

Real LSTM::Grad::norm(){
  return
  this->Wxi.squaredNorm()+this->Whi.squaredNorm()+this->bi.squaredNorm()+
  this->Wxf.squaredNorm()+this->Whf.squaredNorm()+this->bf.squaredNorm()+
  this->Wxo.squaredNorm()+this->Who.squaredNorm()+this->bo.squaredNorm()+
  this->Wxu.squaredNorm()+this->Whu.squaredNorm()+this->bu.squaredNorm();
}

void LSTM::Grad::l2reg(const Real lambda, const LSTM& lstm){
  this->Wxi += lambda*lstm.Wxi; this->Whi += lambda*lstm.Whi;
  this->Wxf += lambda*lstm.Wxf; this->Whf += lambda*lstm.Whf;
  this->Wxo += lambda*lstm.Wxo; this->Who += lambda*lstm.Who;
  this->Wxu += lambda*lstm.Wxu; this->Whu += lambda*lstm.Whu;
}

void LSTM::Grad::l2reg(const Real lambda, const LSTM& lstm, const LSTM& target){
  this->Wxi += lambda*(lstm.Wxi-target.Wxi); this->Whi += lambda*(lstm.Whi-target.Whi); this->bi += lambda*(lstm.bi-target.bi);
  this->Wxf += lambda*(lstm.Wxf-target.Wxf); this->Whf += lambda*(lstm.Whf-target.Whf); this->bf += lambda*(lstm.bf-target.bf);
  this->Wxo += lambda*(lstm.Wxo-target.Wxo); this->Who += lambda*(lstm.Who-target.Who); this->bo += lambda*(lstm.bo-target.bo);
  this->Wxu += lambda*(lstm.Wxu-target.Wxu); this->Whu += lambda*(lstm.Whu-target.Whu); this->bu += lambda*(lstm.bu-target.bu);
}

void LSTM::Grad::sgd(const Real lr, LSTM& lstm){
  Optimizer::sgd(this->Wxi, lr, lstm.Wxi);
  Optimizer::sgd(this->Wxf, lr, lstm.Wxf);
  Optimizer::sgd(this->Wxo, lr, lstm.Wxo);
  Optimizer::sgd(this->Wxu, lr, lstm.Wxu);

  Optimizer::sgd(this->Whi, lr, lstm.Whi);
  Optimizer::sgd(this->Whf, lr, lstm.Whf);
  Optimizer::sgd(this->Who, lr, lstm.Who);
  Optimizer::sgd(this->Whu, lr, lstm.Whu);

  Optimizer::sgd(this->bi, lr, lstm.bi);
  Optimizer::sgd(this->bf, lr, lstm.bf);
  Optimizer::sgd(this->bo, lr, lstm.bo);
  Optimizer::sgd(this->bu, lr, lstm.bu);
}

void LSTM::Grad::adagrad(const Real lr, LSTM& lstm, const Real initVal){
  if (this->gradHist == 0){
    this->gradHist = new LSTM::Grad(lstm);
    this->gradHist->fill(initVal);
  }

  Optimizer::adagrad(this->Wxi, lr, this->gradHist->Wxi, lstm.Wxi);
  Optimizer::adagrad(this->Wxf, lr, this->gradHist->Wxf, lstm.Wxf);
  Optimizer::adagrad(this->Wxo, lr, this->gradHist->Wxo, lstm.Wxo);
  Optimizer::adagrad(this->Wxu, lr, this->gradHist->Wxu, lstm.Wxu);

  Optimizer::adagrad(this->Whi, lr, this->gradHist->Whi, lstm.Whi);
  Optimizer::adagrad(this->Whf, lr, this->gradHist->Whf, lstm.Whf);
  Optimizer::adagrad(this->Who, lr, this->gradHist->Who, lstm.Who);
  Optimizer::adagrad(this->Whu, lr, this->gradHist->Whu, lstm.Whu);

  Optimizer::adagrad(this->bi, lr, this->gradHist->bi, lstm.bi);
  Optimizer::adagrad(this->bf, lr, this->gradHist->bf, lstm.bf);
  Optimizer::adagrad(this->bo, lr, this->gradHist->bo, lstm.bo);
  Optimizer::adagrad(this->bu, lr, this->gradHist->bu, lstm.bu);
}

void LSTM::Grad::momentum(const Real lr, const Real m, LSTM& lstm){
  if (this->gradHist == 0){
    const Real initVal = 0.0;

    this->gradHist = new LSTM::Grad(lstm);
    this->gradHist->fill(initVal);
  }

  Optimizer::momentum(this->Wxi, lr, m, this->gradHist->Wxi, lstm.Wxi);
  Optimizer::momentum(this->Wxf, lr, m, this->gradHist->Wxf, lstm.Wxf);
  Optimizer::momentum(this->Wxo, lr, m, this->gradHist->Wxo, lstm.Wxo);
  Optimizer::momentum(this->Wxu, lr, m, this->gradHist->Wxu, lstm.Wxu);

  Optimizer::momentum(this->Whi, lr, m, this->gradHist->Whi, lstm.Whi);
  Optimizer::momentum(this->Whf, lr, m, this->gradHist->Whf, lstm.Whf);
  Optimizer::momentum(this->Who, lr, m, this->gradHist->Who, lstm.Who);
  Optimizer::momentum(this->Whu, lr, m, this->gradHist->Whu, lstm.Whu);

  Optimizer::momentum(this->bi, lr, m, this->gradHist->bi, lstm.bi);
  Optimizer::momentum(this->bf, lr, m, this->gradHist->bf, lstm.bf);
  Optimizer::momentum(this->bo, lr, m, this->gradHist->bo, lstm.bo);
  Optimizer::momentum(this->bu, lr, m, this->gradHist->bu, lstm.bu);
}

void LSTM::Grad::fill(const Real initVal){
  this->Wxi.fill(initVal);
  this->Wxf.fill(initVal);
  this->Wxo.fill(initVal);
  this->Wxu.fill(initVal);

  this->Whi.fill(initVal);
  this->Whf.fill(initVal);
  this->Who.fill(initVal);
  this->Whu.fill(initVal);

  this->bi.fill(initVal);
  this->bf.fill(initVal);
  this->bo.fill(initVal);
  this->bu.fill(initVal);
}

void LSTM::Grad::saveHist(std::ofstream& ofs){
  Utils::save(ofs, this->gradHist->Wxi);
  Utils::save(ofs, this->gradHist->Wxf);
  Utils::save(ofs, this->gradHist->Wxo);
  Utils::save(ofs, this->gradHist->Wxu);

  Utils::save(ofs, this->gradHist->Whi);
  Utils::save(ofs, this->gradHist->Whf);
  Utils::save(ofs, this->gradHist->Who);
  Utils::save(ofs, this->gradHist->Whu);

  Utils::save(ofs, this->gradHist->bi);
  Utils::save(ofs, this->gradHist->bf);
  Utils::save(ofs, this->gradHist->bo);
  Utils::save(ofs, this->gradHist->bu);
}

void LSTM::Grad::loadHist(std::ifstream& ifs){
  Utils::load(ifs, this->gradHist->Wxi);
  Utils::load(ifs, this->gradHist->Wxf);
  Utils::load(ifs, this->gradHist->Wxo);
  Utils::load(ifs, this->gradHist->Wxu);

  Utils::load(ifs, this->gradHist->Whi);
  Utils::load(ifs, this->gradHist->Whf);
  Utils::load(ifs, this->gradHist->Who);
  Utils::load(ifs, this->gradHist->Whu);

  Utils::load(ifs, this->gradHist->bi);
  Utils::load(ifs, this->gradHist->bf);
  Utils::load(ifs, this->gradHist->bo);
  Utils::load(ifs, this->gradHist->bu);
}

void LSTM::Grad::operator += (const LSTM::Grad& grad){
  this->Wxi += grad.Wxi; this->Whi += grad.Whi; this->bi += grad.bi;
  this->Wxf += grad.Wxf; this->Whf += grad.Whf; this->bf += grad.bf;
  this->Wxo += grad.Wxo; this->Who += grad.Who; this->bo += grad.bo;
  this->Wxu += grad.Wxu; this->Whu += grad.Whu; this->bu += grad.bu;
}

//NOT USED!!
void LSTM::Grad::operator /= (const Real val){
  this->Wxi /= val; this->Whi /= val; this->bi /= val;
  this->Wxf /= val; this->Whf /= val; this->bf /= val;
  this->Wxo /= val; this->Who /= val; this->bo /= val;
  this->Wxu /= val; this->Whu /= val; this->bu /= val;
}

void LSTM::Grad::adam(const Real lr, const Adam::HyperParam& hp, LSTM& lstm){
  if (this->adamGradHist == 0){
    this->adamGradHist = new LSTM::AdamGrad(lstm);
  }

  Adam::adam(this->Wxi, lr, hp, this->adamGradHist->Wxi, lstm.Wxi);
  Adam::adam(this->Wxf, lr, hp, this->adamGradHist->Wxf, lstm.Wxf);
  Adam::adam(this->Wxo, lr, hp, this->adamGradHist->Wxo, lstm.Wxo);
  Adam::adam(this->Wxu, lr, hp, this->adamGradHist->Wxu, lstm.Wxu);

  Adam::adam(this->Whi, lr, hp, this->adamGradHist->Whi, lstm.Whi);
  Adam::adam(this->Whf, lr, hp, this->adamGradHist->Whf, lstm.Whf);
  Adam::adam(this->Who, lr, hp, this->adamGradHist->Who, lstm.Who);
  Adam::adam(this->Whu, lr, hp, this->adamGradHist->Whu, lstm.Whu);

  Adam::adam(this->bi, lr, hp, this->adamGradHist->bi, lstm.bi);
  Adam::adam(this->bf, lr, hp, this->adamGradHist->bf, lstm.bf);
  Adam::adam(this->bo, lr, hp, this->adamGradHist->bo, lstm.bo);
  Adam::adam(this->bu, lr, hp, this->adamGradHist->bu, lstm.bu);
}
void LSTM::Grad::clear(){
  if(this->gradHist != 0){
    delete this->gradHist;
    this->gradHist = NULL;
    std::cout << "LSTM::Grad->gradHist clear" << std::endl;// for checking
  }
  if(this->adamGradHist != 0){
    delete this->adamGradHist;
    this->adamGradHist = NULL;
    std::cout << "LSTM::Grad->adamGradHist clear" << std::endl;// for checking
  }
}

LSTM::AdamGrad::AdamGrad(const LSTM& lstm){
  this->Wxi = Adam::Grad<MatD>(lstm.Wxi); this->Whi = Adam::Grad<MatD>(lstm.Whi); this->bi = Adam::Grad<VecD>(lstm.bi);
  this->Wxf = Adam::Grad<MatD>(lstm.Wxf); this->Whf = Adam::Grad<MatD>(lstm.Whf); this->bf = Adam::Grad<VecD>(lstm.bf);
  this->Wxo = Adam::Grad<MatD>(lstm.Wxo); this->Who = Adam::Grad<MatD>(lstm.Who); this->bo = Adam::Grad<VecD>(lstm.bo);
  this->Wxu = Adam::Grad<MatD>(lstm.Wxu); this->Whu = Adam::Grad<MatD>(lstm.Whu); this->bu = Adam::Grad<VecD>(lstm.bu);
}

LSTMGradChecker::LSTMGradChecker(const int inputDim, const int hiddenDim, const int len)
{
  Rand rnd;
  const Real scale = 0.05;
  const Real dropoutRateX_ = 0.8;

  this->lstm = LSTM(inputDim, hiddenDim, dropoutRateX_);
  this->lstm.init(rnd, scale);

  this->grad = LSTM::Grad(this->lstm);
  this->grad.init();

  for(int i = 0; i < len; ++i){
    this->x.push_back(VecD(inputDim));
    rnd.uniform(this->x.back(), scale);

    this->delx.push_back(VecD::Zero(inputDim));

    this->g.push_back(VecD(hiddenDim));
    rnd.uniform(this->g.back(), scale);

    this->lstmState.push_back(new LSTM::State(this->lstm));
    rnd.setMask(this->lstmState.back()->maskXt, dropoutRateX_);
    this->lossFuncState.push_back(new LossFunc::State);
  }
}

Real LSTMGradChecker::calcLoss(){
  Real loss = 0;
  this->lstm.forward(this->x[0], this->lstmState[0]);
  const Real tmp = this->lossFunc.forward(this->lstmState[0]->h, this->g[0], this->lossFuncState[0]);

  /*
  std::cout << "this->x[0] = " << this->x[0] << std::endl;
  std::cout << "this->lstmState[0]->h = " << this->lstmState[0]->h << std::endl;
  std::cout << "this->g[0] = " << this->g[0] << std::endl;
  std::cout << "\tLoss = " << tmp << std::endl;
  */
  loss += tmp;

  for(size_t i = 1, i_end = lstmState.size(); i < i_end; ++i){
    this->lstm.forward(this->x[i], this->lstmState[i-1]->c, this->lstmState[i-1]->h, this->lstmState[i]);
    const Real tmp_ = this->lossFunc.forward(this->lstmState[i]->h, this->g[i], this->lossFuncState[i]);

    /*
    std::cout << "this->x[" << i << "] = " << this->x[i] << std::endl;
    std::cout << "this->lstmState[" << i << "]->h = " << this->lstmState[i]->h << std::endl;
    std::cout << "this->g[" << i << "] = " << this->g[i] << std::endl;
    std::cout << "\tLoss = " << tmp_ << std::endl;
    */
    loss += tmp_;
  }
  return loss;
}

void LSTMGradChecker::calcGrad(){

  const Real loss = this->calcLoss();

  std::cout << "forward OK" << std::endl;

  std::cout << "Loss = " << loss << std::endl;

  const int i_end = this->lstmState.size();

  for(int i = 0; i < i_end; ++i){
    this->lstmState[i]->setZero();
    this->delx[i].setZero();
  }
    std::cout << "setZero OK" << std::endl;

  for (int i = i_end - 1 ; i >= 1; --i) {
    this->lossFunc.backward(this->lstmState[i]->delh, this->g[i], this->lossFuncState[i]);
    this->lstm.backward1(this->delx[i], this->lstmState[i-1]->delc, this->lstmState[i-1]->delh, this->lstmState[i], this->grad);
  }
  this->lossFunc.backward(this->lstmState[0]->delh, this->g[0], this->lossFuncState[0]);
  this->lstm.backward1(this->delx[0], this->lstmState[0], this->grad);

  std::cout << "backward1 ok" << std::endl;

  for (int i = i_end - 1; i >= 0; --i){
    this->lstm.backward2<LSTM::WXI>(this->lstmState[i], this->grad);
  }
  for (int i = i_end - 1; i >= 1; --i){
    this->lstm.backward2<LSTM::WXF>(this->lstmState[i], this->grad);
  }
  for (int i = i_end - 1; i >= 0; --i){
    this->lstm.backward2<LSTM::WXO>(this->lstmState[i], this->grad);
  }
  for (int i = i_end - 1; i >= 0; --i){
    this->lstm.backward2<LSTM::WXU>(this->lstmState[i], this->grad);
  }
  for (int i = i_end - 1; i >= 1; --i){
    this->lstm.backward2<LSTM::WHI>(this->lstmState[i], this->grad);
  }
  for (int i = i_end - 1; i >= 1; --i){
    this->lstm.backward2<LSTM::WHF>(this->lstmState[i], this->grad);
  }
  for (int i = i_end - 1; i >= 1; --i){
    this->lstm.backward2<LSTM::WHO>(this->lstmState[i], this->grad);
  }
  for (int i = i_end - 1; i >= 1; --i){
    this->lstm.backward2<LSTM::WHU>(this->lstmState[i], this->grad);
  }
  std::cout << "backward2 ok" << std::endl;
}

void LSTMGradChecker::test(){

  const int inputDim = 30;
  const int hiddenDim = 60;
  const int len = 30;

  LSTMGradChecker gc(inputDim, hiddenDim, len);

  /* gradient checking */
  std::cout << "calcGrad" << std::endl;
  gc.calcGrad();

  std::cout << "X" << std::endl;
  for(int i = 0; i < len; ++i){
    gc.gradCheck(gc.delx[i], gc.x[i]);
  }

  std::cout << "Wxi" << std::endl;
  gc.gradCheck(gc.grad.Wxi, gc.lstm.Wxi);
  std::cout << "Wxf" << std::endl;
  gc.gradCheck(gc.grad.Wxf, gc.lstm.Wxf);
  std::cout << "Wxo" << std::endl;
  gc.gradCheck(gc.grad.Wxo, gc.lstm.Wxo);
  std::cout << "Wxu" << std::endl;
  gc.gradCheck(gc.grad.Wxu, gc.lstm.Wxu);

  std::cout << "Whi" << std::endl;
  gc.gradCheck(gc.grad.Whi, gc.lstm.Whi);
  std::cout << "Whf" << std::endl;
  gc.gradCheck(gc.grad.Whf, gc.lstm.Whf);
  std::cout << "Who" << std::endl;
  gc.gradCheck(gc.grad.Who, gc.lstm.Who);
  std::cout << "Whu" << std::endl;
  gc.gradCheck(gc.grad.Whu, gc.lstm.Whu);

  std::cout << "bi" << std::endl;
  gc.gradCheck(gc.grad.bi, gc.lstm.bi);
  std::cout << "bf" << std::endl;
  gc.gradCheck(gc.grad.bf, gc.lstm.bf);
  std::cout << "bo" << std::endl;
  gc.gradCheck(gc.grad.bo, gc.lstm.bo);
  std::cout << "bu" << std::endl;
  gc.gradCheck(gc.grad.bu, gc.lstm.bu);
}
