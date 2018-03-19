#pragma once

#include "Matrix.hpp"

class Optimizer{
public:
  static void sgd(const MatD& grad, const Real learningRate, MatD& param);
  static void sgd(const VecD& grad, const Real learningRate, VecD& param);
  static void sgd(const Real& grad, const Real learningRate, Real& param);
  static void adagrad(MatD& grad, const Real learningRate, MatD& gradHist, MatD& param);
  static void adagrad(VecD& grad, const Real learningRate, VecD& gradHist, VecD& param);
  static void adagrad(Real& grad, const Real learningRate, Real& gradHist, Real& param);
  static void momentum(MatD& grad, const Real learningRate, const Real m, MatD& gradHist, MatD& param);
  static void momentum(VecD& grad, const Real learningRate, const Real m, VecD& gradHist, VecD& param);
  static void momentum(Real& grad, const Real learningRate, const Real m, Real& gradHist, Real& param);
};
