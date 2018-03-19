#pragma once

#include "Matrix.hpp"
#include "Vocabulary.hpp"
#include "Rand.hpp"
#include "Adam.hpp"
#include <iostream>

class Embed{
public:
  enum FORMAT{
    IM, SEQ,
  };

  template<Embed::FORMAT> class State;
  class Grad;
  class AdamGrad;

  MatD embed;

  Embed(){}
  Embed(const int row, const int col)
  {
    // this->embed = MatD(dim, voc_->token2index.size());
    this->embed = MatD(row, col);
  }
  void init(Rand& rnd, const Real scale);
  void init(const std::string& path, Vocabulary& voc);

  void forward(const std::vector<int>& x, const int x_len, Embed::State<Embed::IM>* cur);
  void backward(const std::vector<int>& x, const int x_len, const Embed::State<Embed::IM>* cur, Embed::Grad& grad);

  void forward(const std::vector<int>& x, const int x_len, Embed::State<Embed::SEQ>* cur);
  void backward(const std::vector<int>& x, const int x_len, const Embed::State<Embed::SEQ>* cur, Embed::Grad& grad);

  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

template<>
class Embed::State<Embed::IM>{
public:
  const int dim;
  MatD y;
  MatD dely;

  State(const Embed& embed_):dim(embed_.embed.rows()){}

  void setZero(){
    this->dely.setZero();
  }
  void clear(){
    this->y = MatD();
    this->dely = MatD();
  }
  ~State(){
    this->clear();
  }
};

template<>
class Embed::State<Embed::SEQ>{
public:
  const int dim;
  std::vector<VecD> y;
  std::vector<VecD> dely;

  State(const Embed& embed_):dim(embed_.embed.rows()){}

  void setZero(){
    for(int i = 0, i_end = this->dely.size(); i < i_end; ++i){
      this->dely[i].setZero();
    }
  }
  void clear(){
    for(int i = 0, i_end = this->dely.size(); i < i_end; ++i){
      this->y[i] = VecD();
      this->dely[i] = VecD();
    }
  }
  ~State(){
    this->clear();
  }
};

class Embed::Grad{
public:
  MatD* gradHist;
  Embed::AdamGrad* adamGradHist;

  std::unordered_map<int, VecD> embed;

  Grad():gradHist(0), adamGradHist(0){}
  Grad(const Embed& embed):gradHist(0), adamGradHist(0){}

  void init();
  Real norm();
  void l2reg(const Real lambda, const Embed& embed_);
  void l2reg(const Real lambda, const Embed& embed_, const Embed& target);
  void sgd(const Real lr, Embed& embed_);
  void adagrad(const Real lr, Embed& embed_, const Real initVal = 1.0);
  void momentum(const Real lr, const Real m, Embed& embed_);
  void adam(const Real lr, const Adam::HyperParam& adam, Embed& embed_);//add by kinu
  void saveHist(std::ofstream& ofs);
  void loadHist(std::ifstream& ifs);

  void operator += (const Embed::Grad& grad);
  void operator /= (const Real val);

  void clear();
};

class Embed::AdamGrad{
public:
  std::vector< Adam::Grad<VecD> > embed;

  AdamGrad(const Embed& embed_){
    for(int i = 0, i_end = embed_.embed.cols(); i < i_end; ++i){
      this->embed.push_back( Adam::Grad<VecD>(embed_.embed.rows()) );
    }
  }
};
