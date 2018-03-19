#pragma once

#include "Matrix.hpp"
#include <cmath>

class LossFunc{
public:
  class MeanSquaredError;
  class BinaryCrossEntropy;
  class State;
};

class LossFunc::State{
public:
  VecD x;
  Real x_real;
  State(){}
  void clear(){
    this->x = VecD();
  }
  ~State(){this->clear();}
};

class LossFunc::MeanSquaredError{
public:
  Real forward(const VecD& x, const VecD& g, LossFunc::State* cur){
    cur->x = x;
    return (x - g).squaredNorm()/2;
  }
  void backward(VecD& delx, const VecD& g, const LossFunc::State* cur){
    delx += cur->x - g;
  }
};

class LossFunc::BinaryCrossEntropy{
public:
  Real forward(const VecD& x, const VecD& g, LossFunc::State* cur){
    cur->x = x;
    return (-g.array()*x.array().log()-(1-g.array())*(1-x.array()).log()).sum();
  }
  void backward(VecD& delx, const VecD& g, const LossFunc::State* cur){
    delx.array() += (cur->x.array() - g.array())/(cur->x.array()*(1-cur->x.array()));
  }
  Real forward(const Real x, const Real g, LossFunc::State* cur){
    cur->x_real = x;
    return - g * std::log(x) - ( 1 - g ) * std::log(1-x);
  }
  void backward(Real& delx, const Real g, const LossFunc::State* cur){
    delx += (cur->x_real - g)/(cur->x_real * (1 - cur->x_real) );
  }
};

/*
template<class Type>
class LossFunc{
private:
Type type;
public:
Real forward(const Real output, const Real goldOutput){
return this->type.forward(output, goldOutput);
}
Real forward(const VecD& output, const VecD& goldOutput){
return this->type.forward(output, goldOutput);
}
Real forward(const MatD& output, const MatD& goldOutput){
return this->type.forward(output, goldOutput);
}
Real backward(const Real output, const Real goldOutput){
return this->type.backward(output, goldOutput);
}
VecD backward(const VecD& output, const VecD& goldOutput){
return this->type.backward(output, goldOutput);
}
MatD backward(const MatD& output, const MatD& goldOutput){
return this->type.backward(output, goldOutput);
}
};

class Square{
public:
Real forward(const Real output, const Real goldOutput){
return std::pow((output - goldOutput), 2)/2;
}
Real forward(const VecD& output, const VecD& goldOutput){
return (output - goldOutput).squaredNorm()/2;
}
Real forward(const MatD& output, const MatD& goldOutput){
return (output - goldOutput).squaredNorm()/2;
}
Real backward(const Real output, const Real goldOutput){
return output - goldOutput;
}
VecD backward(const VecD& output, const VecD& goldOutput){
return output - goldOutput;
}
MatD backward(const MatD& output, const MatD& goldOutput){
return output - goldOutput;
}
};

class BinaryCrossEntropy{
public:
Real forward(const Real output, const Real goldOutput){
return -goldOutput*std::log(output)-(1-goldOutput)*std::log(1-output);
}
Real forward(const VecD& output, const VecD& goldOutput){
return (-goldOutput.array()*output.array().log()-(1-goldOutput.array())*(1-output.array()).log()).sum();
}
Real forward(const MatD& output, const MatD& goldOutput){
return (-goldOutput.array()*output.array().log()-(1-goldOutput.array())*(1-output.array()).log()).sum();
}
Real backward(const Real output, const Real goldOutput){
return (output - goldOutput)/(output*(1-output));
}
VecD backward(const VecD& output, const VecD& goldOutput){
return (output.array() - goldOutput.array())/(output.array()*(1-output.array()));
}
MatD backward(const MatD& output, const MatD& goldOutput){
return (output.array() - goldOutput.array())/(output.array()*(1-output.array()));
}
};

class MultiCrossEntropy{
public:
Real forward(const Real output, const Real goldOutput){
return -goldOutput*std::log(output);
}
Real forward(const VecD& output, const VecD& goldOutput){
return -(goldOutput.array()*output.array().log()).sum();
}
Real forward(const MatD& output, const MatD& goldOutput){
return -(goldOutput.array()*output.array().log()).sum();
}
Real backward(const Real output, const Real goldOutput){
return -goldOutput/output;
}
VecD backward(const VecD& output, const VecD& goldOutput){
return -goldOutput.array()/output.array();
}
MatD backward(const MatD& output, const MatD& goldOutput){
return -goldOutput.array()/output.array();
}
};
*/
