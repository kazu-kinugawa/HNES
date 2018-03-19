#include "Convolution.hpp"
#include "Utils.hpp"
#include "Optimizer.hpp"

Convolution::Convolution(const int patchSize, const int kernelNum){
  this->W = MatD(patchSize, kernelNum);
//  this->b = VecD::Zero(kernelNum);
  this->b = MatD::Zero(1, kernelNum);
}

void Convolution::init(Rand& rnd, const Real scale){
  rnd.uniform(this->W, scale);
}

void Convolution::forward(MatD& x, Convolution::State* cur){
  cur->x = &x;
  cur->y.noalias() = x*this->W;
  //cur->y.rowwise() += this->b.transpose();
  cur->y.rowwise() += this->b;
  // Note that column-wise operations return a row vector, while row-wise operations return a column vector.

  // cur->dely = MatD::Zero(cur->y.rows(), cur->y.cols());// init here for backprop
  cur->dely.resize(cur->y.rows(), cur->y.cols());// init here for backprop
}

void Convolution::backward1(MatD& delx, const Convolution::State* cur){
  delx.noalias() += cur->dely*this->W.transpose();
}
void Convolution::backward1(MatD& delx, const Convolution::State* cur, Convolution::Grad& grad){
  delx.noalias() += cur->dely*this->W.transpose();
  grad.b += cur->dely.colwise().sum();
}
void Convolution::save(std::ofstream& ofs){
  Utils::save(ofs, this->W);
  Utils::save(ofs, this->b);
}
void Convolution::load(std::ifstream& ifs){
  Utils::load(ifs, this->W);
  Utils::load(ifs, this->b);
}

Convolution::Grad::Grad(const Convolution& conv):
  gradHist(0), adamGradHist(0)
{
  this->W = MatD(conv.W.rows(), conv.W.cols());
//  this->b = VecD(conv.b.rows());
  this->b = MatD(conv.b.rows(), conv.b.cols());  // mat
  this->init();
}

void Convolution::Grad::init(){
    this->W.setZero();
    this->b.setZero();
}

Real Convolution::Grad::norm(){
  return this->W.squaredNorm() + this->b.squaredNorm();
}

void Convolution::Grad::l2reg(const Real lambda, const Convolution& conv){
    this->W += lambda*conv.W;
    // this->b += lambda*conv.b;
}

void Convolution::Grad::l2reg(const Real lambda, const Convolution& conv, const Convolution& target){
    this->W += lambda*(conv.W-target.W);
    // this->b += lambda*(conv.b-target.b);
}

void Convolution::Grad::sgd(const Real lr, Convolution& conv){
    Optimizer::sgd(this->W, lr, conv.W);
    Optimizer::sgd(this->b, lr, conv.b);
}

void Convolution::Grad::adagrad(const Real lr, Convolution& conv, const Real initVal){
  if (this->gradHist == 0){
    this->gradHist = new Convolution::Grad(conv);
    this->gradHist->fill(initVal);
  }
    Optimizer::adagrad(this->W, lr, this->gradHist->W, conv.W);
    Optimizer::adagrad(this->b, lr, this->gradHist->b, conv.b);
}

void Convolution::Grad::momentum(const Real lr, const Real m, Convolution& conv){
  if (this->gradHist == 0){
    const Real initVal = 0.0;

    this->gradHist = new Convolution::Grad(conv);
    this->gradHist->fill(initVal);
  }
    Optimizer::momentum(this->W, lr, m, this->gradHist->W, conv.W);
    Optimizer::momentum(this->b, lr, m, this->gradHist->b, conv.b);
}

void Convolution::Grad::fill(const Real initVal){
    this->W.fill(initVal);
    this->b.fill(initVal);
}

void Convolution::Grad::saveHist(std::ofstream& ofs){
    Utils::save(ofs, this->gradHist->W);
    Utils::save(ofs, this->gradHist->b);
}

void Convolution::Grad::loadHist(std::ifstream& ifs){
    Utils::load(ifs, this->gradHist->W);
    Utils::load(ifs, this->gradHist->b);
}

void Convolution::Grad::operator += (const Convolution::Grad& grad){
    this->W += grad.W;
    this->b += grad.b;
}

//NOT USED!!
void Convolution::Grad::operator /= (const Real val){
    this->W /= val;
    this->b /= val;
}

void Convolution::Grad::adam(const Real lr, const Adam::HyperParam& adam, Convolution& conv){
  if (!this->adamGradHist){
      this->adamGradHist = new Convolution::AdamGrad(conv);
  }

    Adam::adam(this->W, lr, adam, this->adamGradHist->W, conv.W);
    Adam::adam(this->b, lr, adam, this->adamGradHist->b, conv.b);
}
void Convolution::Grad::clear(){
  if(this->adamGradHist != 0){
    delete this->adamGradHist;
    std::cout << "Convolution::Grad->adamGradHist clear" << std::endl;  // for checking
  }
  if(this->gradHist != 0){
    delete this->gradHist;
    std::cout << "Convolution::Grad->gradHist clear" << std::endl;  // for checking
  }
}

Convolution::AdamGrad::AdamGrad(const Convolution& conv){
  this->W = Adam::Grad<MatD>(conv.W);
  // this->b = Adam::Grad<VecD>(conv.b);
  this->b = Adam::Grad<MatD>(conv.b);// mat
}

ConvolutionGradChecker::ConvolutionGradChecker(const int patchNum, const int patchSize, const int kernelNum){

  Rand rnd;
  const Real scale = 0.05;

  // set paramters
  this->conv = Convolution(patchSize, kernelNum);
  this->conv.init(rnd, scale);

  // set grad
  this->grad = Convolution::Grad(this->conv);

  // set state
  this->cur = new Convolution::State;

  // set inputs
  this->x = MatD(patchNum, patchSize);
  rnd.uniform(this->x, scale);

  this->delx = MatD::Zero(patchNum, patchSize);

  this->g = MatD(patchNum, kernelNum);
  rnd.uniform(this->g, scale);
}
Real ConvolutionGradChecker::calcLoss(){
  this->conv.forward(this->x, this->cur);
  return (this->cur->y - this->g).squaredNorm()/2;
}
void ConvolutionGradChecker::calcGrad(){

  const Real loss = this->calcLoss();
  std::cout << "LOSS = " << loss << std::endl;

  this->cur->dely += this->cur->y - this->g;
  this->conv.backward1(this->delx, this->cur, this->grad);
  this->conv.backward2<Convolution::KERNEL>(this->cur, this->grad);
}
/*
void ConvolutionGradChecker::test(){

  const int patchNum = 100;
  const int patchSize = 100;
  const int kernelNum = 200;

  ConvolutionGradChecker gc(patchNum, patchSize, kernelNum);

  // gradient checking
  std::cout << "calcGrad" << std::endl;
  gc.calcGrad();

  std::cout << "x" << std::endl;
  gc.gradCheck(gc.delx, gc.x);
  std::cout << "W" << std::endl;
  gc.gradCheck(gc.grad.W, gc.conv.W);
  std::cout << "b" << std::endl;
  gc.gradCheck(gc.grad.b, gc.conv.b);
}
*/
