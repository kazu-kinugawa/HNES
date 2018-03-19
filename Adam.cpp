#include "Adam.hpp"
#include "Utils.hpp"
#include <cmath>

void Adam::adam(const MatD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::MatGrad& gradHist, MatD& param){
  gradHist.getGrad(grad, hp);
  param -= learningRate * gradHist.grad;
}

void Adam::adam(const VecD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::VecGrad& gradHist, VecD& param){
  gradHist.getGrad(grad, hp);
  param -= learningRate * gradHist.grad;
}

void Adam::adam(const Real& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::SclGrad& gradHist, Real& param){
  gradHist.getGrad(grad, hp);
  param -= learningRate * gradHist.grad;
}

void Adam::adam(const MatD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::Grad<MatD>& gradHist, MatD& param){
  gradHist.getGrad(grad, hp);
  param -= learningRate * gradHist.grad;
}

void Adam::adam(const VecD& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::Grad<VecD>& gradHist, VecD& param){
  gradHist.getGrad(grad, hp);
  param -= learningRate * gradHist.grad;
}

void Adam::adam(const Real& grad, const Real learningRate, const Adam::HyperParam& hp, Adam::Grad<Real>& gradHist, Real& param){
  gradHist.getGrad(grad, hp);
  param -= learningRate * gradHist.grad;
}

void Adam::HyperParam::operator = (const Adam::HyperParam& hp){
  this->alpha = hp.alpha;
  this->beta1 = hp.beta1;
  this->beta2 = hp.beta2;
  this->eps = hp.eps;
}


Real Adam::HyperParam::lr(){
  //can not use inline?
  ++this->t;
  return this->alpha * std::sqrt(1-std::pow(this->beta2,this->t)) / (1-std::pow(this->beta1,this->t));
}

Adam::MatGrad::MatGrad(const MatD temp){
  this->m = MatD::Zero(temp.rows(), temp.cols());
  this->v = MatD::Zero(temp.rows(), temp.cols());
  this->grad = MatD::Zero(temp.rows(), temp.cols());
}

Adam::MatGrad::MatGrad(const int row, const int col){
  this->m = MatD::Zero(row, col);
  this->v = MatD::Zero(row, col);
  this->grad = MatD::Zero(row, col);
}

void Adam::MatGrad::getGrad(const MatD& grad, const Adam::HyperParam& hp){
  this->m += (1-hp.beta1)*(grad-this->m);
  this->v.array() += (1-hp.beta2)*(grad.array()*grad.array()-this->v.array());
  this->grad = this->m.array() / (this->v.array().sqrt() + hp.eps);
}

void Adam::MatGrad::clear(){
  this->m = MatD();
  this->v = MatD();
  this->grad = MatD();
}

Adam::VecGrad::VecGrad(const VecD temp){
  this->m = VecD::Zero(temp.rows());
  this->v = VecD::Zero(temp.rows());
  this->grad = VecD::Zero(temp.rows());
}

Adam::VecGrad::VecGrad(const int row){
  this->m = VecD::Zero(row);
  this->v = VecD::Zero(row);
  this->grad = VecD::Zero(row);
}

void Adam::VecGrad::getGrad(const VecD& grad, const Adam::HyperParam& hp){
  this->m += (1-hp.beta1)*(grad-this->m);
  this->v.array() += (1-hp.beta2)*(grad.array()*grad.array()-this->v.array());
  this->grad = this->m.array() / (this->v.array().sqrt() + hp.eps);
}

void Adam::VecGrad::clear(){
  this->m = VecD();
  this->v = VecD();
  this->grad = VecD();
}

void Adam::SclGrad::getGrad(const Real grad, const Adam::HyperParam& hp){
  this->m += (1-hp.beta1)*(grad-this->m);
  this->v += (1-hp.beta2)*(grad*grad-this->v);
  this->grad = this->m / (std::sqrt(this->v) + hp.eps);
}
