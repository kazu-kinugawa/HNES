#include "Embed.hpp"
#include "Utils.hpp"
#include <cassert>

void Embed::init(Rand& rnd, const Real scale){
  rnd.uniform(this->embed, scale);
}
void Embed::init(const std::string& path, Vocabulary& voc){
  std::ifstream fin(path);
  if(!fin){
    std::cout << path << " cann't open" << std::endl;
    assert(fin);
  }
  fin.unsetf(std::ios::skipws);

  std::stringstream ss;
  std::string buf;
  VecD v = VecD::Zero(this->embed.rows());
  getline(fin, buf, '\n');
  Real val;
  while(getline(fin, buf, '\n')){
    std::istringstream stream(buf);
    std::string tmp;
    getline(stream, tmp, ' ');
    if(!voc.token2index.count(tmp)){
      continue;
    }
    const int id = voc.token2index.at(tmp);
    int i = 0;
    while(getline(stream, tmp, ' ')){
      ss.str(tmp);
      ss >> val;
      v.coeffRef(i++,0) = val;
      ss.str("");
      ss.clear(std::stringstream::goodbit);
    }
    this->embed.col(id) = v;
  }
  fin.close();
}

// image
void Embed::forward(const std::vector<int>& x, const int x_len, Embed::State<Embed::IM>* cur){
  cur->y.resize(cur->dim, x_len);
  cur->dely.resize(cur->dim, x_len);
  // cur->dely.setZero();
  for(int i = 0; i < x_len; ++i){
    cur->y.col(i) = this->embed.col(x[i]);
  }
}
void Embed::backward(const std::vector<int>& x, const int x_len, const Embed::State<Embed::IM>* cur, Embed::Grad& grad){
  for(int i = 0; i < x_len; ++i){
    const int index = x[i];
    if (grad.embed.count(index)){
      grad.embed.at(index) += cur->dely.col(i);
    }
    else {
      grad.embed.insert(std::pair<int,VecD>(index, cur->dely.col(i)));
    }
  }
}

// seq
void Embed::forward(const std::vector<int>& x, const int x_len, Embed::State<Embed::SEQ>* cur){
  for(int i = 0; i < x_len; ++i){
    cur->y[i] = this->embed.col(x[i]);
    // cur->dely[i] = VecD::Zero(cur->y[i].rows());
  }
}
void Embed::backward(const std::vector<int>& x, const int x_len, const Embed::State<Embed::SEQ>* cur, Embed::Grad& grad){
  for(int i = 0; i < x_len; ++i){
    const int index = x[i];
    if (grad.embed.count(index)){
      grad.embed.at(index) += cur->dely[i];
    }
    else {
      grad.embed.insert(std::pair<int,VecD>(index, cur->dely[i]));
    }
  }
}

void Embed::save(std::ofstream& ofs){
  Utils::save(ofs, this->embed);
}
void Embed::load(std::ifstream& ifs){
  Utils::load(ifs, this->embed);
}

void Embed::Grad::init(){
    this->embed.clear();
}

Real Embed::Grad::norm(){
  Real res = 0;
  for(auto itr = this->embed.begin(), itrEnd = this->embed.end(); itr != itrEnd; ++itr){
    res += itr->second.squaredNorm();
  }
  return res;
}

void Embed::Grad::l2reg(const Real lambda, const Embed& embed_){
    for(auto itr = this->embed.begin(), itrEnd = this->embed.end(); itr != itrEnd; ++itr){
      itr->second += lambda*embed_.embed.col(itr->first);
    }
}

void Embed::Grad::l2reg(const Real lambda, const Embed& embed_, const Embed& target){
    for(auto itr = this->embed.begin(), itrEnd = this->embed.end(); itr != itrEnd; ++itr){
      itr->second += lambda*(embed_.embed.col(itr->first)-target.embed.col(itr->first));
    }
}

void Embed::Grad::sgd(const Real lr, Embed& embed_){
    for (auto it = this->embed.begin(), itEnd = this->embed.end(); it != itEnd; ++it){
      embed_.embed.col(it->first) -= lr * it->second;
    }
}

void Embed::Grad::adagrad(const Real lr, Embed& embed_, const Real initVal){
  if (this->gradHist == 0){
    this->gradHist = new MatD(embed_.embed.rows(), embed_.embed.cols());
    this->gradHist->fill(initVal);
  }
    for (auto it = this->embed.begin(), itEnd = this->embed.end(); it != itEnd; ++it){
      this->gradHist->col(it->first).array() += it->second.array().square();
      it->second.array() /= this->gradHist->col(it->first).array().sqrt();
      embed_.embed.col(it->first) -= lr * it->second;
    }
}

void Embed::Grad::momentum(const Real lr, const Real m, Embed& embed_){
  if (this->gradHist == 0){
    const Real initVal = 0.0;

    this->gradHist = new MatD(embed_.embed.rows(), embed_.embed.cols());
    this->gradHist->fill(initVal);
  }

    for (auto it = this->embed.begin(), itEnd = this->embed.end(); it != itEnd; ++it){
      this->gradHist->col(it->first).array() *= m;
      this->gradHist->col(it->first) += lr*it->second;
      embed_.embed.col(it->first) -= this->gradHist->col(it->first);
    }
}

void Embed::Grad::saveHist(std::ofstream& ofs){
    Utils::save(ofs, *(this->gradHist));
}

void Embed::Grad::loadHist(std::ifstream& ifs){
    Utils::load(ifs, *(this->gradHist));
}

void Embed::Grad::operator += (const Embed::Grad& grad){
  for (auto it = grad.embed.begin(), itEnd = grad.embed.end(); it != itEnd; ++it){
    if (this->embed.count(it->first)){
      this->embed.at(it->first) += it->second;
    }
    else {
      this->embed.insert(std::make_pair(it->first, it->second));
    }
  }
}

//NOT USED!!
void Embed::Grad::operator /= (const Real val){
  const Real coeff = 1.0/val;
  for (auto it = this->embed.begin(), itEnd = this->embed.end(); it != itEnd; ++it){
    it->second *= coeff;
  }
}

void Embed::Grad::adam(const Real lr, const Adam::HyperParam& adam, Embed& embed_){
  if (!this->adamGradHist){
      this->adamGradHist = new Embed::AdamGrad(embed_);
  }

  for (auto it = this->embed.begin(), it_end = this->embed.end(); it != it_end; ++it){
    const int id = it->first;
    Adam::Grad<VecD>& tmp = this->adamGradHist->embed[id];
    tmp.getGrad(it->second, adam);
    embed_.embed.col(id) -= lr * tmp.grad;
//    this->adamGradHist->embed[id].getGrad(it->second, adam);
//    embed_.embed.col(id) -= lr * this->adamGradHist->embed[id].grad;
  }
}

void Embed::Grad::clear(){
  if(this->gradHist != 0){
    *this->gradHist = MatD();
    delete this->gradHist;
    this->gradHist = NULL;
    std::cout << "Embed::Grad->gradHist clear" << std::endl;// for checking
  }
  if(this->adamGradHist != 0){
    delete this->adamGradHist;
    this->adamGradHist = NULL;
    std::cout << "Embed::Grad->adamGradHist clear" << std::endl;// for checking
  }
}
