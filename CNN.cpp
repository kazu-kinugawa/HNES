#include "CNN.hpp"

CNN::CNN(const int inputDim, const int left, const int right, const int colBlock, const int kernelNum){
    this->pad = Padding(left, right);
    this->im2col = Im2Col(inputDim, colBlock);
    this->conv = Convolution(inputDim*colBlock, kernelNum);
}

void CNN::init(Rand& rnd, const Real scale){
  this->conv.init(rnd, scale);
}

void CNN::forward(const MatD& x, CNN::State* cur){
  /*
  this->pad.forward(x, cur->pad);
  this->im2col.forward(cur->pad->y, cur->im2col);
  this->conv.forward(cur->im2col->y, cur->conv);
  this->af.forward(cur->conv->y, cur->af);
  this->mp.forward(cur->af->y, cur->mp);
  */

  // Originally, af.forward should be followed by mp.forward
  // But tanh/Relu is a monotonuically increasing function
  // So if the order of the two operations is reversed, results are the same (?)
  this->pad.forward(x, cur->pad);
  this->im2col.forward(cur->pad->y, cur->im2col);
  this->conv.forward(cur->im2col->y, cur->conv);
  this->mp.forward(cur->conv->y, cur->mp);
  this->af.forward(cur->mp->y, cur->af);
}

void CNN::backward1(MatD& delx, CNN::State* cur, CNN::Grad& grad){
  /*
  this->mp.backward(cur->af->dely, cur->mp);
  this->af.backward(cur->conv->dely, cur->af);
  this->conv.backward1(cur->im2col->dely, cur->conv, grad.conv);
  this->im2col.backward(cur->pad->dely, cur->im2col);
  this->pad.backward(delx, cur->pad);
*/

  // Originally, af.forward should be followed by mp.forward
  // But tanh/Relu is a monotonuically increasing function
  // So if the order of the two operations is reversed, results are the same (?)
  this->af.backward(cur->mp->dely, cur->af);
  this->mp.backward(cur->conv->dely, cur->mp);
  this->conv.backward1(cur->im2col->dely, cur->conv, grad.conv);
  this->im2col.backward(cur->pad->dely, cur->im2col);
  this->pad.backward(delx, cur->pad);
}
void CNN::backward2(CNN::State* cur, CNN::Grad& grad){
  this->conv.backward2<Convolution::KERNEL>(cur->conv, grad.conv);
}

void CNN::forward(const MatD& x, CNN::State* cur, const MaxPooling::GRAD_CHECK flag){
  /*
  this->pad.forward(x, cur->pad);
  this->im2col.forward(cur->pad->y, cur->im2col);
  this->conv.forward(cur->im2col->y, cur->conv);
  this->af.forward(cur->conv->y, cur->af);
  this->mp.forward(cur->af->y, cur->mp, flag);
  */

  // Originally, af.forward should be followed by mp.forward
  // But tanh/Relu is a monotonuically increasing function
  // So if the order of the two operations is reversed, results are the same (?)
  this->pad.forward(x, cur->pad);
  this->im2col.forward(cur->pad->y, cur->im2col);
  this->conv.forward(cur->im2col->y, cur->conv);
  this->mp.forward(cur->conv->y, cur->mp, flag);
  this->af.forward(cur->mp->y, cur->af);
}

void CNN::save(std::ofstream& ofs){
  this->conv.save(ofs);
}
void CNN::load(std::ifstream& ifs){
  this->conv.load(ifs);
}

CNN::Grad::Grad(CNN& cnn){
  this->conv = Convolution::Grad(cnn.conv);
  this->conv.init();
}
void CNN::Grad::init(){
  this->conv.init();
}
Real CNN::Grad::norm(){
  return this->conv.norm();
}
void CNN::Grad::operator += (const CNN::Grad& grad){
  this->conv += grad.conv;
}
void CNN::Grad::l2reg(const Real lambda, const CNN& cnn){
  this->conv.l2reg(lambda, cnn.conv);
}
void CNN::Grad::l2reg(const Real lambda, const CNN& cnn, const CNN& target){
  this->conv.l2reg(lambda, cnn.conv, target.conv);
}
void CNN::Grad::sgd(const Real lr, CNN& cnn){
  this->conv.sgd(lr, cnn.conv);
}
void CNN::Grad::adam(const Real lr, const Adam::HyperParam& hp, CNN& cnn){
  this->conv.adam(lr, hp, cnn.conv);
}

CNNGradChecker::CNNGradChecker(const int inputDim, const int len, const int left, const int right, const int colBlock, const int kernelNum){
  Rand rnd;
  const Real scale = 0.05;

  // set parameter
  this->cnn = CNN(inputDim, left, right, colBlock, kernelNum);
  this->cnn.init(rnd, scale);

  // set grad
  this->grad = CNN::Grad(this->cnn);

  // set state
  this->cur = new CNN::State(this->cnn);

  // set sample
  this->x = MatD(inputDim, len);
  rnd.uniform(this->x, scale);

  this->delx = MatD::Zero(inputDim, len);

  this->g = VecD(kernelNum);
  rnd.uniform(this->g, scale);

  // set flag for MaxPooling
  this->flag = MaxPooling::CALC_GRAD;
}

Real CNNGradChecker::calcLoss(){
  this->cnn.forward(this->x, this->cur, this->flag);
  return (*this->cur->y - this->g).squaredNorm()/2;
}
void CNNGradChecker::calcGrad(){
  const Real loss = this->calcLoss();
  std::cout << "Loss = " << loss << std::endl;

  *this->cur->dely += (*this->cur->y - this->g);
  std::cout << "loss backprop" << std::endl;

  this->cnn.backward1(this->delx, this->cur, this->grad);
  std::cout << "backward1" << std::endl;

  this->cnn.backward2(this->cur, this->grad);
  std::cout << "backward2" << std::endl;
}

void CNNGradChecker::test1(){
  const int inputDim = 150;
  const int len = 15;
  const int left = 2;
  const int right = 3;
  const int colBlock = 4;
  const int kernelNum = 100;

  CNNGradChecker gc(inputDim, len, left, right, colBlock, kernelNum);

  /* gradient checking */
  std::cout << "calcGrad" << std::endl;
  gc.flag = MaxPooling::CALC_GRAD;
  gc.calcGrad();

  /* gradient checking */
  gc.flag = MaxPooling::CALC_LOSS;

  std::cout << "conv.W" << std::endl;
  gc.gradCheck(gc.grad.conv.W, gc.cnn.conv.W);
  std::cout << "conv.b" << std::endl;
  gc.gradCheck(gc.grad.conv.b, gc.cnn.conv.b);
}

void CNNGradChecker::test2(){
  const int inputDim = 3;
  const int len = 3;
  const int left = 1;
  const int right = 1;
  const int colBlock = 2;
  const int kernelNum = 5;

  CNNGradChecker gc(inputDim, len, left, right, colBlock, kernelNum);

  std::cout << "gc.x" << std::endl;
  std::cout << gc.x << std::endl;

  std::cout << "res1" << std::endl;

  /*
  Padding::State ptmp;

  gc.cnn.pad.forward(gc.x, &ptmp);
  std::cout << "padding ok" << std::endl;

  gc.cnn.im2col.forward(ptmp.y, gc.cur->im2col);
  std::cout << "im2col ok" << std::endl;
  */

  gc.cnn.pad.forward(gc.x, gc.cur->pad);
  std::cout << "padding ok" << std::endl;

  gc.cnn.im2col.forward(gc.cur->pad->y, gc.cur->im2col);
  std::cout << "im2col ok" << std::endl;

  gc.cnn.conv.forward(gc.cur->im2col->y, gc.cur->conv);
  std::cout << "convolution ok" << std::endl;

  MatD tmp = gc.cur->conv->y.unaryExpr(std::ptr_fun(::tanh));
  std::cout << "activation ok" << std::endl;

  VecD res1 = tmp.colwise().maxCoeff();
  std::cout << "max pooling" << std::endl;

  std::cout << res1 << std::endl;

  std::cout << "res2" << std::endl;

  gc.cnn.pad.forward(gc.x, gc.cur->pad);
  std::cout << "padding ok" << std::endl;

  gc.cnn.im2col.forward(gc.cur->pad->y, gc.cur->im2col);
  std::cout << "im2col ok" << std::endl;

  gc.cnn.conv.forward(gc.cur->im2col->y, gc.cur->conv);
  std::cout << "convolution ok" << std::endl;

  gc.cnn.mp.forward(gc.cur->conv->y, gc.cur->mp);
  std::cout << "max pooling" << std::endl;

  gc.cnn.af.forward(gc.cur->mp->y, gc.cur->af);
  std::cout << "activation ok" << std::endl;

  VecD res2 = gc.cur->af->y;

  std::cout << res2 << std::endl;
}
