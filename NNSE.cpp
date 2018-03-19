#include "NNSE.hpp"
#include "Utils.hpp"
#include "Timer.hpp"
#include <omp.h>
#include <sys/time.h>
#include "Optimizer.hpp"
#include "Writer.hpp"

NNSE::Config::Config(const CHECK check){

  if(check == ONE_EPOCH_CHECK){
    this->date = Timer::getCurrentTime() + "-ONE-EPOCH-CHECK";// auto
  }
  else{
    this->date = Timer::getCurrentTime();
  }

  const int dataSize = 30000;
  const std::string modelName = "NNSE10";
  const std::string testModelName = "2018-01-15-19-46-full.54ITR";

    // path
  if(check == GRAD_CHECK){
    this->base = "C:/Users/K_Kinugawa/Desktop/workspace/";
    this->trainDataPath = this->base + "path-list/170721/170721.30000.train.path.org";
    this->validDataPath = this->base + "path-list/170721/170721.30000.train.path.org";
    this->testDataPath = this->base + "path-list/170721/170721.30000.train.path.org";
  }
  else{
    this->base = "/data/local/kinugawa/";// for tuna, pigeon0, owl1
    this->trainDataPath = this->base + "path-list/170721/170721." + std::to_string(dataSize) + ".train.path.org";
    this->validDataPath = this->base + "path-list/170721/170721." + std::to_string(dataSize) + ".valid.path.org";
    this->testDataPath = this->base + "path-list/170721/170721." + std::to_string(dataSize) + ".test.path.org";
  }

 // Log path for training
  this->validLogPath = this->base + modelName + "/log/valid-" + this->date + ".csv";
  this->trainLogPath = this->base + modelName + "/log/train-" + this->date + ".csv";
  this->modelSavePath = this->base + modelName + "/model/model-" + this->date + "-full.";

  // Log path for testing
  this->modelLoadPath = this->base + modelName + "/model/model-" + testModelName + ".bin";
  this->testResultSavePath = this->base + modelName + "/log/test-" + testModelName + ".csv";
  this->writerRootPath = this->base + "Eval/170721." + std::to_string(dataSize) + "/system/" + modelName + "/" + testModelName;

  // const
  this->bodyMaxWordNum = 120;
  this->bodyMaxSentNum = 320;
  this->bodyMaxParNum = 320;
  this->bodyMaxSecNum = 170;

  // parameters
  if(check == OFF){
    this->useWord2vec = true;
  }
  else{
    this->useWord2vec = false;
  }

  this->scale = 0.05;

  if(check == GRAD_CHECK){
    this->wordEmbDim = 18;
    this->sentEmbDim = 18;
    this->docEmbDim = 18;
  }
  else if(check == ONE_EPOCH_CHECK){
    this->wordEmbDim = CNNS_KERNEL_NUM;
    this->sentEmbDim = CNNS_KERNEL_NUM;
    this->docEmbDim = CNNS_KERNEL_NUM;
  }
  else if(check == OFF){
    this->wordEmbDim = 300;
    this->sentEmbDim = 600;
    this->docEmbDim = 600;
  }

  // word2vec path, 150 or 300 currently
  this->initEmbMatLoadPath = this->base + "word2vec-corpus/170721." + std::to_string(dataSize) + ".train.vec" + std::to_string(this->wordEmbDim) + ".csv";

  this->cnnsInputDim = this->wordEmbDim;
  this->cnnsOutputDim = this->sentEmbDim;

  this->encSentInputDim = this->sentEmbDim;  this->encSentOutputDim = this->docEmbDim;
  this->encParInputDim = this->docEmbDim;    this->encParOutputDim = this->docEmbDim;
  this->encSecInputDim = this->docEmbDim;    this->encSecOutputDim = this->docEmbDim;

  this->decSentInputDim = this->sentEmbDim;  this->decSentOutputDim = this->docEmbDim;
  this->decParInputDim = this->docEmbDim;    this->decParOutputDim = this->docEmbDim;
  this->decSecInputDim = this->docEmbDim;    this->decSecOutputDim = this->docEmbDim;

  this->classifierHiddenDim = this->docEmbDim * 2;

  this->learningRate = 1.0;
  this->clip = 5.0;
  this->decay = 1.0e-02;

  // l2reg
  this->lambda = 1.0e-04;
  this->useCnnsL2reg = false;
  // adam
  this->adam.alpha = 0.001;
  this->adam.beta1 = 0.9;// 0.99 by Cheng and Lapata
  this->adam.beta2 = 0.999;
  this->adam.eps = 1.0e-08;

  const Real dr = 0.5;

  this->dropoutRateEncSentX = dr;
  this->dropoutRateDecSentX = dr;
  this->dropoutRateEncParX = dr;
  this->dropoutRateDecParX = dr;
  this->dropoutRateEncSecX = dr;
  this->dropoutRateDecSecX = dr;

  this->dropoutRateMLPSent = dr;
  this->dropoutRateMLPPar = dr;
  this->dropoutRateMLPSec = dr;

  this->tokenFreqThreshold = 5;
  this->nameFreqThreshold = 5;

  if(check == OFF){
  this->threadNum = 10;
  this->testThreadNum = 2;
  this->miniBatchSize = 20;
  this->itrThreshold = (dataSize*0.9/this->miniBatchSize)/3;
  this->epochThreshold = 20;
  }
  else{
  this->threadNum = 1;
  this->testThreadNum = 1;
  this->miniBatchSize = 1;
  this->epochThreshold = 1;
  this->itrThreshold = dataSize;
  }

  this->seed = (unsigned long)time(NULL);
  // this->seed = 1506938329;

  std::cout << "date = " << this->date << std::endl;
  std::cout << "miniBatchSize = " << this->miniBatchSize << std::endl;
  std::cout << "itrThreshold = " << this->itrThreshold << std::endl;
  std::cout << "seed = " << this->seed << std::endl;
}

NNSE::NNSE(Vocabulary& v, NNSE::Config& _config, const MODE mode_):
 voc(v), config(_config), mode(mode_),
 zeroSent(VecD::Zero(_config.decSentInputDim)),
 zeroPar(VecD::Zero(_config.decParInputDim)),
 zeroSec(VecD::Zero(_config.decSecInputDim)),
 grad(0)
{
  // load data
  this->loadData(mode_);

  this->rnd.init(_config.seed);

  this->embed = Embed(_config.wordEmbDim, v.token2index.size());

  this->cnns = CNNS<CNNS_KERNEL_NUM>(_config.cnnsInputDim, _config.cnnsOutputDim);

  // note! NOT _config.decSentInputDim BUT _config.decSecInputDim
  this->EOD = VecD(_config.decSecInputDim);

  this->encSent = LSTM(_config.encSentInputDim, _config.encSentOutputDim, _config.dropoutRateEncSentX);
  this->encPar = LSTM(_config.encParInputDim, _config.encParOutputDim, _config.dropoutRateEncParX);
  this->encSec = LSTM(_config.encSecInputDim, _config.encSecOutputDim, _config.dropoutRateEncSecX);

  this->decSec = LSTM(_config.decSecInputDim, _config.decSecOutputDim, _config.dropoutRateDecSecX);
  this->classifierSec = MLP(_config.encSecOutputDim, _config.decSecOutputDim, _config.classifierHiddenDim, _config.dropoutRateMLPSec);

  this->decPar = LSTM(_config.decParInputDim, _config.decParOutputDim, _config.dropoutRateDecParX);
  this->classifierPar = MLP(_config.encParOutputDim, _config.decParOutputDim, _config.classifierHiddenDim,
    _config.classifierHiddenDim, _config.classifierHiddenDim, _config.dropoutRateMLPPar);

  this->decSent = LSTM(_config.decSentInputDim, _config.decSentOutputDim, _config.dropoutRateDecSentX);
  this->classifierSent = MLP(_config.encSentOutputDim, _config.decSentOutputDim, _config.classifierHiddenDim,
    _config.classifierHiddenDim, _config.classifierHiddenDim, _config.dropoutRateMLPSent);

  if(mode_ == TRAIN){
    // if train mode, grad is on. if test mode, grad is off
    this->grad = new NNSE::Grad(*this);
  }
}

void NNSE::init(Rand& rnd, const Real scale){

  if(!this->config.useWord2vec){
    this->embed.init(rnd, scale);
  }
  else{
    this->embed.init(this->config.initEmbMatLoadPath, this->voc);
    std::cout << "Embed is initiallized by word2vec" << std::endl;
  }

  this->cnns.init(rnd, scale);
  rnd.uniform(this->EOD, scale);

  this->encSent.init(rnd, scale);
  this->encPar.init(rnd, scale);
  this->encSec.init(rnd, scale);

  this->decSent.init(rnd, scale);
  this->decPar.init(rnd, scale);
  this->decSec.init(rnd, scale);

  this->classifierSent.init(rnd, scale);
  this->classifierPar.init(rnd, scale);
  this->classifierSec.init(rnd, scale);
}

void NNSE::cnnsForward(NNSE::ThreadArg& arg){
  for (int i = 0, i_end = arg.sentSeqEndIndex; i <= i_end; ++i){
    NNSE::Sentence& state = arg.orgSentState[i];
    this->embed.forward(state.sent->word, state.sent->wordNum, state.embedState);
    this->cnns.forward(state.embedState->y, state.cnnsState);
  }
}
void NNSE::cnnsForward(NNSE::ThreadArg& arg, const MaxPooling::GRAD_CHECK flag){
  // for grad checking
  for (int i = 0, i_end = arg.sentSeqEndIndex; i <= i_end; ++i){
    NNSE::Sentence& state = arg.orgSentState[i];
    this->embed.forward(state.sent->word, state.sent->wordNum, state.embedState);
    this->cnns.forward(state.embedState->y, state.cnnsState, flag);
  }
}
void NNSE::cnnsBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  for (int i = 0, i_end = arg.sentSeqEndIndex; i <= i_end; ++i){
    NNSE::Sentence& state = arg.orgSentState[i];
    this->cnns.backward1(state.embedState->dely, state.cnnsState, grad.cnnsGrad);
    this->embed.backward(state.sent->word, state.sent->wordNum, state.embedState, grad.embedGrad);
  }
}
void NNSE::cnnsBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // for speeding up
  for(int i = 0; i < CNNS_KERNEL_NUM; ++i){
    for(int j = 0, j_end = arg.sentSeqEndIndex; j <= j_end; ++j){
      this->cnns.backward2(arg.orgSentState[j].cnnsState, i, grad.cnnsGrad);
    }
  }
}

void NNSE::encoderSentForward(NNSE::Paragraph& par){
  // within paragraph
  this->encSent.forward(par.orgSentState[0]->cnnsState->y, par.encSentState[0]);
  // std::cout << "ENC : " << "\t\tSENT#" << par.orgSentState[0]->index << std::endl;
  for(size_t i = 1, i_end = par.encSentState.size() - 1; i <= i_end; ++i){
    this->encSent.forward(par.orgSentState[i]->cnnsState->y, par.encSentState[i-1]->c, par.encSentState[i-1]->h, par.encSentState[i]);
    // std::cout << "ENC : " << "\t\tSENT#" << par.orgSentState[i]->index << std::endl;
  }
}
void NNSE::encoderSentBackward1(NNSE::Paragraph& par, NNSE::Grad& grad){
  // within paragraph encoder
  for(size_t i = par.encSentState.size() - 1, i_end = 1; i >= i_end; --i){
    this->encSent.backward1(par.orgSentState[i]->cnnsState->dely, par.encSentState[i-1]->delc, par.encSentState[i-1]->delh, par.encSentState[i], grad.encSentGrad);
    // std::cout << "ENC_BACK1 : " << "\t\tSENT#" << par.orgSentState[i]->index << std::endl;
  }
  this->encSent.backward1(par.orgSentState[0]->cnnsState->dely, par.encSentState[0], grad.encSentGrad);
  // std::cout << "ENC_BACK1 : " << "\t\tSENT#" << par.orgSentState[0]->index << std::endl;
}
void NNSE::encoderSentBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // for speeding up
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 0; --j){
      this->encSent.backward2<LSTM::WXI>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 1; --j){
      this->encSent.backward2<LSTM::WXF>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 0; --j){
      this->encSent.backward2<LSTM::WXO>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 0; --j){
      this->encSent.backward2<LSTM::WXU>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 1; --j){
      this->encSent.backward2<LSTM::WHI>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 1; --j){
      this->encSent.backward2<LSTM::WHF>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 1; --j){
      this->encSent.backward2<LSTM::WHO>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].encSentState.size() - 1; j >= 1; --j){
      this->encSent.backward2<LSTM::WHU>(arg.orgParState[i].encSentState[j], grad.encSentGrad);
    }
  }
}

void NNSE::encoderParForward(NNSE::Section& sec){
  // within section
  this->encoderSentForward(*sec.orgParState[0]);
  this->encPar.forward(sec.orgParState[0]->encSentEndState->h, sec.encParState[0]);
  // std::cout << "ENC : " << "\tPar#" << sec.orgParState[0]->index << std::endl;
  for(size_t i = 1, i_end = sec.encParState.size() - 1; i <= i_end; ++i){
    this->encoderSentForward(*sec.orgParState[i]);
    this->encPar.forward(sec.orgParState[i]->encSentEndState->h, sec.encParState[i-1]->c, sec.encParState[i-1]->h, sec.encParState[i]);
    // std::cout << "ENC : " << "\tPar#" << sec.orgParState[i]->index << std::endl;
  }
}
void NNSE::encoderParBackward1(NNSE::Section& sec, NNSE::Grad& grad){
  // within section encoder
  for(size_t i = sec.encParState.size() - 1, i_end = 1; i >= i_end; --i){
    this->encPar.backward1(sec.orgParState[i]->encSentEndState->delh, sec.encParState[i-1]->delc, sec.encParState[i-1]->delh, sec.encParState[i], grad.encParGrad);
    // std::cout << "ENC_BACK1 : " << "\tPar#" << sec.orgParState[i]->index << std::endl;
    this->encoderSentBackward1(*sec.orgParState[i], grad);
  }
  this->encPar.backward1(sec.orgParState[0]->encSentEndState->delh, sec.encParState[0], grad.encParGrad);
  // std::cout << "ENC_BACK1 : " << "\tPar#" << sec.orgParState[0]->index << std::endl;
  this->encoderSentBackward1(*sec.orgParState[0], grad);
}
void NNSE::encoderParBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // for speeding up
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 0; --j){
      this->encPar.backward2<LSTM::WXI>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 1; --j){
      this->encPar.backward2<LSTM::WXF>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 0; --j){
      this->encPar.backward2<LSTM::WXO>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 0; --j){
      this->encPar.backward2<LSTM::WXU>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 1; --j){
      this->encPar.backward2<LSTM::WHI>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 1; --j){
      this->encPar.backward2<LSTM::WHF>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 1; --j){
      this->encPar.backward2<LSTM::WHO>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].encParState.size() - 1; j >= 1; --j){
      this->encPar.backward2<LSTM::WHU>(arg.orgSecState[i].encParState[j], grad.encParGrad);
    }
  }
}

void NNSE::encoderSecForward(NNSE::ThreadArg& arg){
  // within document
  this->encoderParForward(arg.orgSecState[0]);
  this->encSec.forward(arg.orgSecState[0].encParEndState->h, arg.encSecState[0]);
  // std::cout << "ENC : " << "SEC#" << 0 << std::endl;
  for(size_t i = 1, i_end = arg.secSeqEndIndex; i <= i_end; ++i){
    this->encoderParForward(arg.orgSecState[i]);
    this->encSec.forward(arg.orgSecState[i].encParEndState->h, arg.encSecState[i-1]->c, arg.encSecState[i-1]->h, arg.encSecState[i]);
    // std::cout << "ENC : " << "SEC#" << i << std::endl;
  }
}
void NNSE::encoderSecBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // within document encoder
  for(size_t i = arg.secSeqEndIndex, i_end = 1; i >= i_end; --i){
    this->encSec.backward1(arg.orgSecState[i].encParEndState->delh, arg.encSecState[i-1]->delc, arg.encSecState[i-1]->delh, arg.encSecState[i], grad.encSecGrad);
    // std::cout << "ENC_BACK1 : " << "SEC#" << doc.orgSecState[i]->index << std::endl;
    this->encoderParBackward1(arg.orgSecState[i], grad);
  }
  this->encSec.backward1(arg.orgSecState[0].encParEndState->delh, arg.encSecState[0], grad.encSecGrad);
  // std::cout << "ENC_BACK1 : " << "SEC#" << doc.orgSecState[0]->index << std::endl;
  this->encoderParBackward1(arg.orgSecState[0], grad);
}
void NNSE::encoderSecBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // for speeding up
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->encSec.backward2<LSTM::WXI>(arg.encSecState[j], grad.encSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 1; --j){
    this->encSec.backward2<LSTM::WXF>(arg.encSecState[j], grad.encSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->encSec.backward2<LSTM::WXO>(arg.encSecState[j], grad.encSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->encSec.backward2<LSTM::WXU>(arg.encSecState[j], grad.encSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 1; --j){
    this->encSec.backward2<LSTM::WHI>(arg.encSecState[j], grad.encSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 1; --j){
    this->encSec.backward2<LSTM::WHF>(arg.encSecState[j], grad.encSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 1; --j){
    this->encSec.backward2<LSTM::WHO>(arg.encSecState[j], grad.encSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 1; --j){
    this->encSec.backward2<LSTM::WHU>(arg.encSecState[j], grad.encSecGrad);
  }
}

void NNSE::encoderForward(NNSE::ThreadArg& arg){
  this->encoderSecForward(arg);
}
void NNSE::encoderBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  this->encoderSecBackward1(arg, grad);
}
void NNSE::decoderForwardTrain(NNSE::ThreadArg& arg){
  this->decoderSecForwardTrain(arg);
}
void NNSE::decoderForwardTest(NNSE::ThreadArg& arg){
  this->decoderSecForwardTest(arg);
}
void NNSE::decoderBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  this->decoderSecBackward1(arg, grad);
}

Real NNSE::classifierSecForward(const int i, NNSE::ThreadArg& arg){
  NNSE::Section& secState = arg.orgSecState[i];
  this->classifierSec.forward(secState.encSecState->h, secState.decSecState->h, secState.classifierSecState);
  secState.sec->score = *secState.classifierSecState->y;
  return this->lossFunc.forward(secState.sec->score, secState.sec->label, secState.lossFuncSecState);
}
void NNSE::classifierSecBackward1(const int i, NNSE::ThreadArg& arg, NNSE::Grad& grad){
  NNSE::Section& secState = arg.orgSecState[i];
  this->lossFunc.backward(*secState.classifierSecState->dely, secState.sec->label, secState.lossFuncSecState);
  this->classifierSec.backward1(secState.encSecState->delh, secState.decSecState->delh, secState.classifierSecState, grad.classifierSecGrad);
}
void NNSE::classifierSecBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    this->classifierSec.backward2<MLP::L1>(arg.orgSecState[i].classifierSecState, grad.classifierSecGrad);
  }
}

void NNSE::decoderSecForwardTrain(NNSE::ThreadArg& arg){
  // within document
  this->decSec.forward(this->EOD, arg.encEndState->c, arg.encEndState->h, arg.decSecState[0]);
  arg.metrics.loss.coeffRef(2,0) += this->classifierSecForward(0, arg);
  // std::cout << "DEC : SEC#" << 0 << std::endl;
  this->decoderParForwardTrain(arg.orgSecState[0], arg);// note! train
  for(size_t i = 1, i_end = arg.secSeqEndIndex; i <= i_end; ++i){
     // note! train
     if(arg.orgSecState[i-1].sec->label > 0.5){
       // positive
       // which one is better for decoder input?
       // this->decSec.forward(doc.orgSecState[i-1]->decParState.back()->h, doc.decSecState[i-1]->c, doc.decSecState[i-1]->h, doc.decSecState[i]);
       this->decSec.forward(arg.orgSecState[i-1].encParEndState->h, arg.decSecState[i-1]->c, arg.decSecState[i-1]->h, arg.decSecState[i]);
     }
     else{
       // negative
       this->decSec.forward(this->zeroSec, arg.decSecState[i-1]->c, arg.decSecState[i-1]->h, arg.decSecState[i]);
     }
    arg.metrics.loss.coeffRef(2,0) += this->classifierSecForward(i, arg);
    // std::cout << "DEC : SEC#" << i << std::endl;
    this->decoderParForwardTrain(arg.orgSecState[i], arg);
  }
}
void NNSE::decoderSecBackward1(ThreadArg& arg, NNSE::Grad& grad){
  // within document decoder
  for(size_t i = arg.secSeqEndIndex, i_end = 1; i >= i_end; --i){
    this->decoderParBackward1(arg.orgSecState[i], grad);
    this->classifierSecBackward1(i, arg, grad);
    // train
    if(arg.orgSecState[i-1].sec->label > 0.5){
      // positive
      // which one is better for decoder input?
      // this->decSec.backward1(doc.orgSecState[i-1]->decParState.back()->delh, doc.decSecState[i-1]->delc, doc.decSecState[i-1]->delh, doc.decSecState[i], grad.decSecGrad);
      this->decSec.backward1(arg.orgSecState[i-1].encParEndState->delh, arg.decSecState[i-1]->delc, arg.decSecState[i-1]->delh, arg.decSecState[i], grad.decSecGrad);
    }
    else{
      // negative
      this->decSec.backward1(grad.zeroSecGrad, arg.decSecState[i-1]->delc, arg.decSecState[i-1]->delh, arg.decSecState[i], grad.decSecGrad);
    }
    //std::cout << "DEC_BACK1 : " << "SEC #" << doc.orgSecState[i]->index << " -> SEC #" << doc.orgSecState[i-1]->index << " : par # " << doc.orgSecState[i-1]->orgParState.back()->index << std::endl;
    // std::cout << "DEC_BACK1 : " << "SEC #" << i << std::endl;
  }
  this->decoderParBackward1(arg.orgSecState[0], grad);
  this->classifierSecBackward1(0, arg, grad);
  this->decSec.backward1(grad.EODGrad, arg.encEndState->delc, arg.encEndState->delh, arg.decSecState[0], grad.decSecGrad);
  // std::cout << "DEC_BACK1 : " << "SEC #" << 0 << std::endl;
}
void NNSE::decoderSecBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // section decoder
  // for speeding up
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WXI>(arg.decSecState[j], grad.decSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WXF>(arg.decSecState[j], grad.decSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WXO>(arg.decSecState[j], grad.decSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WXU>(arg.decSecState[j], grad.decSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WHI>(arg.decSecState[j], grad.decSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WHF>(arg.decSecState[j], grad.decSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WHO>(arg.decSecState[j], grad.decSecGrad);
  }
  for (int j = arg.secSeqEndIndex; j >= 0; --j){
    this->decSec.backward2<LSTM::WHU>(arg.decSecState[j], grad.decSecGrad);
  }
}
void NNSE::decoderSecForwardTest(NNSE::ThreadArg& arg){
  // within document
  this->decSec.forward(this->EOD, arg.encEndState->c, arg.encEndState->h, arg.decSecState[0]);
  arg.metrics.loss.coeffRef(2,0) += this->classifierSecForward(0, arg);
  // std::cout << "DEC : SEC#" << 0 << std::endl;
  this->decoderParForwardTest(arg.orgSecState[0], arg);// note! TEST
  for(size_t i = 1, i_end = arg.secSeqEndIndex; i <= i_end; ++i){
    // note! test
    // which one is better for decoder input?
    // this->decSec.forward(doc.orgSecState[i-1]->sec->score*doc.orgSecState[i-1]->decParState.back()->h, doc.decSecState[i-1]->c, doc.decSecState[i-1]->h, doc.decSecState[i]);
    this->decSec.forward(arg.orgSecState[i-1].sec->score*arg.orgSecState[i-1].encParEndState->h, arg.decSecState[i-1]->c, arg.decSecState[i-1]->h, arg.decSecState[i]);
    arg.metrics.loss.coeffRef(2,0) += this->classifierSecForward(i, arg);
    // std::cout << "DEC : SEC#" << i << std::endl;
    this->decoderParForwardTest(arg.orgSecState[i], arg);// note! TEST
  }
}
Real NNSE::classifierParForward(const int i, NNSE::Section& sec){
  NNSE::Paragraph& parState = *sec.orgParState[i];
  this->classifierPar.forward(parState.encParState->h, parState.decParState->h, *parState.classifierSecState->yy, parState.classifierParState);
  parState.par->score = *parState.classifierParState->y;
  return this->lossFunc.forward(parState.par->score, parState.par->label, parState.lossFuncParState);
}
void NNSE::classifierParBackward1(const int i, NNSE::Section& sec, NNSE::Grad& grad){
  NNSE::Paragraph& parState = *sec.orgParState[i];
  this->lossFunc.backward(*parState.classifierParState->dely, parState.par->label, parState.lossFuncParState);
  this->classifierPar.backward1(parState.encParState->delh, parState.decParState->delh, *parState.classifierSecState->delyy, parState.classifierParState, grad.classifierParGrad);
}
void NNSE::classifierParBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    this->classifierPar.backward2<MLP::L1>(arg.orgParState[i].classifierParState, grad.classifierParGrad);
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    this->classifierPar.backward2<MLP::L0>(arg.orgParState[i].classifierParState, grad.classifierParGrad);
  }
}

void NNSE::decoderParForwardTrain(NNSE::Section& sec, NNSE::ThreadArg& arg){
  // within section
  this->decPar.forward(sec.decSecState->h, sec.decParState[0]);
  arg.metrics.loss.coeffRef(1,0) += this->classifierParForward(0, sec);
  this->decoderSentForwardTrain(*sec.orgParState[0], arg);// note! Train
  for(size_t i = 1, i_end = sec.decParState.size() - 1; i <= i_end; ++i){
    // note! train
    if(sec.orgParState[i-1]->par->label > 0.5){
      // positive
      // which is better for decoder input?
      // this->decPar.forward(sec.orgParState[i-1]->decSentState.back()->h, sec.decParState[i-1]->c, sec.decParState[i-1]->h, sec.decParState[i]);
      this->decPar.forward(sec.orgParState[i-1]->encSentEndState->h, sec.decParState[i-1]->c, sec.decParState[i-1]->h, sec.decParState[i]);
    }
    else{
      // negative
      this->decPar.forward(this->zeroPar, sec.decParState[i-1]->c, sec.decParState[i-1]->h, sec.decParState[i]);
    }
    arg.metrics.loss.coeffRef(1,0) += this->classifierParForward(i, sec);
    this->decoderSentForwardTrain(*sec.orgParState[i], arg);// note! train
  }
}
void NNSE::decoderParBackward1(NNSE::Section& sec, NNSE::Grad& grad){
  // within section decoder
  for(size_t i = sec.decParState.size() - 1, i_end = 1; i >= i_end; --i){
    this->decoderSentBackward1(*sec.orgParState[i], grad);
    this->classifierParBackward1(i, sec, grad);
    // note train!
    if(sec.orgParState[i-1]->par->label > 0.5){
      // positive
      // which is better for decoder input?
      // this->decPar.backward1(sec.orgParState[i-1]->decSentState.back()->delh, sec.decParState[i-1]->delc, sec.decParState[i-1]->delh, sec.decParState[i], grad.decParGrad);
      this->decPar.backward1(sec.orgParState[i-1]->encSentEndState->delh, sec.decParState[i-1]->delc, sec.decParState[i-1]->delh, sec.decParState[i], grad.decParGrad);
    }
    else{
      // negative
      this->decPar.backward1(grad.zeroParGrad, sec.decParState[i-1]->delc, sec.decParState[i-1]->delh, sec.decParState[i], grad.decParGrad);
    }
    // std::cout << "DEC_BACK1 : " << "\tpar #" << sec.orgParState[i]->index << std::endl;
    // std::cout << "DEC_BACK1 : " << "\tpar #" << sec.orgParState[i]->index << " -> par #" << sec.orgParState[i-1]->index << " : SENT # " << sec.orgParState[i-1]->orgSentState.back()->index << std::endl;
  }
  this->decoderSentBackward1(*sec.orgParState[0], grad);
  this->classifierParBackward1(0, sec, grad);
  this->decPar.backward1(sec.decSecState->delh, sec.decParState[0], grad.decParGrad);
  // std::cout << "DEC_BACK1 : " << "\tpar #" << sec.orgParState[0]->index << std::endl;
  // std::cout << "DEC_BACK1 : " << "\tpar #" << sec.orgParState[0]->index << " -> SEC #" << sec.index << std::endl;
}
void NNSE::decoderParBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // paragraph decoder
  // for speeding up
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 0; --j){
      this->decPar.backward2<LSTM::WXI>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 1; --j){
      this->decPar.backward2<LSTM::WXF>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 0; --j){
      this->decPar.backward2<LSTM::WXO>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 0; --j){
      this->decPar.backward2<LSTM::WXU>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 1; --j){
      this->decPar.backward2<LSTM::WHI>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 1; --j){
      this->decPar.backward2<LSTM::WHF>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 1; --j){
      this->decPar.backward2<LSTM::WHO>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
  for(int i = arg.secSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgSecState[i].decParState.size() - 1; j >= 1; --j){
      this->decPar.backward2<LSTM::WHU>(arg.orgSecState[i].decParState[j], grad.decParGrad);
    }
  }
}
void NNSE::decoderParForwardTest(NNSE::Section& sec, NNSE::ThreadArg& arg){
  // within section
  this->decPar.forward(sec.decSecState->h, sec.decParState[0]);
  arg.metrics.loss.coeffRef(1,0) += this->classifierParForward(0, sec);
  this->decoderSentForwardTest(*sec.orgParState[0], arg);// note! Test
  for(size_t i = 1, i_end = sec.decParState.size() - 1; i <= i_end; ++i){
    // note! test
    // which is better for decoder input?
    // this->decPar.forward(sec.orgParState[i-1]->par->score*sec.orgParState[i-1]->decSentState.back()->h, sec.decParState[i-1]->c, sec.decParState[i-1]->h, sec.decParState[i]);
    this->decPar.forward(sec.orgParState[i-1]->par->score*sec.orgParState[i-1]->encSentEndState->h, sec.decParState[i-1]->c, sec.decParState[i-1]->h, sec.decParState[i]);
    arg.metrics.loss.coeffRef(1,0) += this->classifierParForward(i, sec);
    this->decoderSentForwardTest(*sec.orgParState[i], arg);// note! Test
  }
}

Real NNSE::classifierSentForward(const int i, NNSE::Paragraph& par){
  NNSE::Sentence& state = *par.orgSentState[i];
  this->classifierSent.forward(state.encSentState->h, state.decSentState->h, *state.classifierParState->yy, state.classifierSentState);
  state.sent->score = *state.classifierSentState->y;
  return this->lossFunc.forward(state.sent->score, state.sent->label, state.lossFuncSentState);
}
void NNSE::classifierSentBackward1(const int i, NNSE::Paragraph& par, NNSE::Grad& grad){
  NNSE::Sentence& state = *par.orgSentState[i];
  this->lossFunc.backward(*state.classifierSentState->dely, state.sent->label, state.lossFuncSentState);
  this->classifierSent.backward1(state.encSentState->delh, state.decSentState->delh, *state.classifierParState->delyy, state.classifierSentState, grad.classifierSentGrad);
}
void NNSE::classifierSentBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  for(int i = arg.sentSeqEndIndex; i >= 0; --i){
    this->classifierSent.backward2<MLP::L1>(arg.orgSentState[i].classifierSentState, grad.classifierSentGrad);
  }
  for(int i = arg.sentSeqEndIndex; i >= 0; --i){
    this->classifierSent.backward2<MLP::L0>(arg.orgSentState[i].classifierSentState, grad.classifierSentGrad);
  }
}

void NNSE::decoderSentForwardTrain(NNSE::Paragraph& par, NNSE::ThreadArg& arg){
  // within paragraph
  this->decSent.forward(par.decParState->h, par.decSentState[0]);
  arg.metrics.loss.coeffRef(0,0) += this->classifierSentForward(0, par);
  // std::cout << "SENT #" << par.orgSentState[0]->index << " -> classifierSentForward" << std::endl;
  for(size_t i = 1, i_end = par.decSentState.size() - 1; i <= i_end; ++i){
    // note! train
    if(par.orgSentState[i-1]->sent->label > 0.5){
      // positive
      this->decSent.forward(par.orgSentState[i-1]->cnnsState->y, par.decSentState[i-1]->c, par.decSentState[i-1]->h, par.decSentState[i]);
    }
    else{
      // negative
      this->decSent.forward(this->zeroSent, par.decSentState[i-1]->c, par.decSentState[i-1]->h, par.decSentState[i]);
    }
    arg.metrics.loss.coeffRef(0,0) += this->classifierSentForward(i, par);
    // std::cout << "SENT #" << par.orgSentState[i]->index << " -> classifierSentForward" << std::endl;
  }
}
void NNSE::decoderSentBackward1(NNSE::Paragraph& par, NNSE::Grad& grad){
  // within paragraph decoder
  // std::cout << "DEC_BACK1 : " << "\t\tSENT ";
  // std::cout << "DEC_BACK1 : SENT # " << par.orgSentState[par.decSentState.size() - 1]->index << " -> # " << par.orgSentState[0]->index << std::endl;
  for(size_t i = par.decSentState.size() - 1, i_end = 1; i >= i_end; --i){
    this->classifierSentBackward1(i, par, grad);
    // std::cout << "SENT #" << par.orgSentState[i]->index << " -> classifierSentBackward1" << std::endl;
    // std::cout << "SENT #"<< par.orgSentState[i]->index;
    // std::cout << " -> score = " << par.orgSentState[i]->sent->score << ", label = " << par.orgSentState[i]->sent->label << std::endl;
    // std::cout << "LREG -> SENT #" << par.orgSentState[i]->index << std::endl;
    if(par.orgSentState[i-1]->sent->label > 0.5){
      // positive
      this->decSent.backward1(par.orgSentState[i-1]->cnnsState->dely, par.decSentState[i-1]->delc, par.decSentState[i-1]->delh, par.decSentState[i], grad.decSentGrad);
    }
    else{
      // negative
      this->decSent.backward1(grad.zeroSentGrad, par.decSentState[i-1]->delc, par.decSentState[i-1]->delh, par.decSentState[i], grad.decSentGrad);
    }
    // std::cout << "DEC_BACK1 : " << "\t\tSENT #" << par.orgSentState[i]->index << std::endl;
    // std::cout << "# " << par.orgSentState[i]->index << ", ";
  }
  this->classifierSentBackward1(0, par, grad);
  // std::cout << "SENT #" << par.orgSentState[0]->index << " -> classifierSentBackward1" << std::endl;
  // std::cout << "SENT #"<< par.orgSentState[0]->index;
  // std::cout << " -> score = " << par.orgSentState[0]->sent->score << ", label = " << par.orgSentState[0]->sent->label << std::endl;
  // std::cout << "LREG -> SENT #" << par.orgSentState[0]->index << std::endl;
  this->decSent.backward1(par.decParState->delh, par.decSentState[0], grad.decSentGrad);
  // std::cout << "DEC_BACK1 : " << "\t\tSENT #" << par.orgSentState[0]->index << std::endl;
  // std::cout << "# " << par.orgSentState[0]->index << " ";
  // std::cout << " -> par #" << par.index << std::endl;
}
void NNSE::decoderSentBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad){
  // for speeding up
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 0; --j){
      this->decSent.backward2<LSTM::WXI>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 1; --j){
      this->decSent.backward2<LSTM::WXF>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 0; --j){
      this->decSent.backward2<LSTM::WXO>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 0; --j){
      this->decSent.backward2<LSTM::WXU>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 1; --j){
      this->decSent.backward2<LSTM::WHI>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 1; --j){
      this->decSent.backward2<LSTM::WHF>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 1; --j){
      this->decSent.backward2<LSTM::WHO>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
  for(int i = arg.parSeqEndIndex; i >= 0; --i){
    for(int j = arg.orgParState[i].decSentState.size() - 1; j >= 1; --j){
      this->decSent.backward2<LSTM::WHU>(arg.orgParState[i].decSentState[j], grad.decSentGrad);
    }
  }
}
void NNSE::decoderSentForwardTest(NNSE::Paragraph& par, NNSE::ThreadArg& arg){
  // within paragraph
  this->decSent.forward(par.decParState->h, par.decSentState[0]);
  arg.metrics.loss.coeffRef(0,0) += this->classifierSentForward(0, par);
  for(size_t i = 1, i_end = par.decSentState.size() - 1; i <= i_end; ++i){
    // note! test
    this->decSent.forward(par.orgSentState[i-1]->sent->score*par.orgSentState[i-1]->cnnsState->y, par.decSentState[i-1]->c, par.decSentState[i-1]->h, par.decSentState[i]);
    arg.metrics.loss.coeffRef(0,0) += this->classifierSentForward(i, par);
    // std::cout << "SENT #" << par.orgSentState[i]->index << " -> LREG" << std::endl;
  }
}

void NNSE::train(NNSE::ThreadArg& arg){
  // train

  // forward
  this->cnnsForward(arg);
  this->encoderForward(arg);
  this->decoderForwardTrain(arg);

  // backward start
  arg.setZero();
  // backward1
  this->decoderBackward1(arg, arg.grad);
  this->encoderBackward1(arg, arg.grad);
  this->cnnsBackward1(arg, arg.grad);
  // backward2
  this->classifierSecBackward2(arg, arg.grad);
  this->classifierParBackward2(arg, arg.grad);
  this->classifierSentBackward2(arg, arg.grad);
  this->decoderSecBackward2(arg, arg.grad);
  this->decoderParBackward2(arg, arg.grad);
  this->decoderSentBackward2(arg, arg.grad);
  this->encoderSentBackward2(arg, arg.grad);
  this->encoderParBackward2(arg, arg.grad);
  this->encoderSecBackward2(arg, arg.grad);
  this->cnnsBackward2(arg, arg.grad);

  arg.doc->rerank();
  arg.metrics.recall.coeffRef(0,0) += arg.doc->getSentRecall(arg.doc->resSent, arg.doc->resSentNum);
  arg.metrics.recall.coeffRef(1,0) += arg.doc->getPrgRecall(arg.doc->resPrg, arg.doc->resPrgNum);
  arg.metrics.recall.coeffRef(2,0) += arg.doc->getSecRecall(arg.doc->resSec, arg.doc->resSecNum);

  arg.doc->classifyError(arg.doc->resSent, arg.doc->resSentNum);
  arg.metrics.errorRate += arg.doc->errorRate;
  arg.metrics.errorRegRate += arg.doc->errorRegRate;
}
void NNSE::valid(NNSE::ThreadArg& arg){
  // test
  // forward
  this->cnnsForward(arg);
  this->encoderForward(arg);
  this->decoderForwardTest(arg);

  arg.doc->rerank();
  arg.metrics.recall.coeffRef(0,0) += arg.doc->getSentRecall(arg.doc->resSent, arg.doc->resSentNum);
  arg.metrics.recall.coeffRef(1,0) += arg.doc->getPrgRecall(arg.doc->resPrg, arg.doc->resPrgNum);
  arg.metrics.recall.coeffRef(2,0) += arg.doc->getSecRecall(arg.doc->resSec, arg.doc->resSecNum);

  arg.doc->classifyError(arg.doc->resSent, arg.doc->resSentNum);
  arg.metrics.errorRate += arg.doc->errorRate;
  arg.metrics.errorRegRate += arg.doc->errorRegRate;

  arg.doc->repOrgToken(arg.doc->resSent, arg.doc->resSentNum);
  arg.doc->evaluate(arg.doc->resSent, arg.doc->resSentNum, arg.metrics.rouge);
  arg.doc->repUnkToken(arg.doc->resSent, arg.doc->resSentNum);
}
void NNSE::test(NNSE::ThreadArg& arg, NNSE::Metrics& targetMetrics){
  // test
  const VecD tmp = arg.metrics.loss;

  // forward
  this->cnnsForward(arg);
  this->encoderForward(arg);
  this->decoderForwardTest(arg);

  targetMetrics.loss = arg.metrics.loss - tmp;

  arg.doc->rerank();
  targetMetrics.recall.coeffRef(0,0) = arg.doc->getSentRecall(arg.doc->resSent, arg.doc->resSentNum);
  targetMetrics.recall.coeffRef(1,0) = arg.doc->getPrgRecall(arg.doc->resPrg, arg.doc->resPrgNum);
  targetMetrics.recall.coeffRef(2,0) = arg.doc->getSecRecall(arg.doc->resSec, arg.doc->resSecNum);
  arg.metrics.recall += targetMetrics.recall;

  arg.doc->classifyError(arg.doc->resSent, arg.doc->resSentNum);
  targetMetrics.errorRate = arg.doc->errorRate;
  arg.metrics.errorRate += arg.doc->errorRate;
  // reg
  targetMetrics.errorRegRate = arg.doc->errorRegRate;
  arg.metrics.errorRegRate += arg.doc->errorRegRate;

  const ROUGE tmpRouge = arg.metrics.rouge;

  arg.doc->repOrgToken(arg.doc->resSent, arg.doc->resSentNum);
  arg.doc->evaluate(arg.doc->resSent, arg.doc->resSentNum, arg.metrics.rouge);
  arg.doc->repUnkToken(arg.doc->resSent, arg.doc->resSentNum);

  targetMetrics.rouge.rouge1 = arg.metrics.rouge.rouge1 - tmpRouge.rouge1;
  targetMetrics.rouge.rouge2 = arg.metrics.rouge.rouge2 - tmpRouge.rouge2;
  targetMetrics.rouge.rougeL = arg.metrics.rouge.rougeL - tmpRouge.rougeL;
}

void NNSE::trainOpenMP(const unsigned int epoch, const unsigned int epochEnd){
  /* global */
  static std::ofstream ofsValid(this->config.validLogPath, std::ios::app);//append
  static std::ofstream ofsTrain(this->config.trainLogPath, std::ios::app);//append
  assert(ofsValid);
  assert(ofsTrain);

  // static NNSE::Grad grad;
  static std::vector<NNSE::ThreadArg> args;
  static std::vector<std::pair<unsigned int, unsigned int> > miniBatch;

  // static ROUGE eval;

  // metrics
  NNSE::Metrics trainMetrics, miniBatchMetrics, checkPointMetrics, validationMetrics;

  unsigned int count = 0;
  unsigned int batchCount = 0;

  static const Real lrOrig = this->config.learningRate;
  Timer trainTimer, testTimer;

  trainTimer.start();

  if (args.empty()) {
    for (unsigned int i = 0; i < this->config.threadNum; ++i) {
      args.push_back(NNSE::ThreadArg(*this));
      args.back().metrics.setZero();
    }
    for (int i = 0, step = this->trainData.size()/this->config.miniBatchSize; i < step; ++i) {
      miniBatch.push_back(std::pair<int, int>(i*this->config.miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*this->config.miniBatchSize-1)));
    }

    this->itr = 0;
    this->miniBatchStepNum = 0;

    ofsTrain << "ITR" << "\t" << "MINIBATCH_STEP_NUM" << "\t";
    ofsTrain << "SENT_LOSS" << "\t" << "PAR_LOSS" << "\t" << "SEC_LOSS" << "\t";
    ofsTrain << "SENT_RECALL" << "\t" << "PAR_RECALL" << "\t" << "SEC_RECALL" << "\t";
    ofsTrain << "REG_POS_SEC_POS_PAR_RATE" << "\t" << "REG_POS_SEC_NEGA_PAR_RATE" << "\t" << "REG_NEGA_SEC_NEGA_PAR_RATE" << "\t";
    ofsTrain << std::endl;

    ofsValid << "ITR" << "\t" << "MINIBATCH_STEP_NUM" << "\t";
    ofsValid << "ROUGE_1_F" << "\t" << "ROUGE_2_F" << "\t" << "ROUGE_L_F" << "\t";
    ofsValid << "SENT_LOSS" << "\t" << "PAR_LOSS" << "\t" << "SEC_LOSS" << "\t";
    ofsValid << "SENT_RECALL" << "\t" << "PAR_RECALL" << "\t" << "SEC_RECALL" << "\t";
    ofsValid << "REG_POS_SEC_POS_PAR_RATE" << "\t" << "REG_POS_SEC_NEGA_PAR_RATE" << "\t" << "REG_NEGA_SEC_NEGA_PAR_RATE" << "\t";
    ofsValid << std::endl;
  }

  this->rnd.shuffle(this->trainData);

  for (auto it = miniBatch.begin(), itEnd = miniBatch.end(); it != itEnd; ++it) {
    std::cout << "\r"
    << "At Epoch: " << epoch << " -> Progress: " << ++count << "/" << miniBatch.size()
    << " mini batches" << std::flush;

    ++this->miniBatchStepNum;

    const unsigned int thisStepSize = it->second - it->first + 1;

    for (unsigned id = 0; id < this->config.threadNum; ++id) {
      args[id].metrics.setZero();// metrics
    }

    // setMiniBacth
    this->setMiniBatch(it->first, it->second);

    #pragma omp parallel for num_threads(this->config.threadNum) schedule(dynamic) shared(args)
    //#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic)
    //      for (unsigned int i = it->first; i <= it->second; ++i) {
    for (unsigned int i = 0; i < thisStepSize; ++i) {
      const unsigned int id = omp_get_thread_num();
      args[id].init(this->miniBatchData[i]);
      args[id].mask(*this);// for dropout
      this->train(args[id]);
    }

    // clearMiniBacth
    this->clearMiniBatch(it->first, it->second);

    miniBatchMetrics.setZero();// metrics

    for (unsigned int id = 0; id < this->config.threadNum; ++id) {
      *this->grad += args[id].grad;
      args[id].grad.init();
      miniBatchMetrics += args[id].metrics;// metrics
    }

    // metrics
    trainMetrics += miniBatchMetrics;
    miniBatchMetrics *= 1.0/thisStepSize;

    // ofsTrain << lossMiniBatch << "," << recallMiniBatch << std::endl;

    ++batchCount;
    // metrics
    checkPointMetrics += miniBatchMetrics;

    // SGD & gradinent clipping
    /*
    Real gradNorm = sqrt(grad.norm())/thisStepSize;
    Utils::infNan(gradNorm);
    Real lr = (gradNorm > this->clip ? this->clip*this->lr/gradNorm : this->lr);//clip
    grad.sgd(*this, lr/thisStepSize);
    std::cout << " -> Training Loss (per doc) : " << lossMiniBatch << ", gradNorm : " << gradNorm <<
    ", lr (per doc) : " << lr << std::endl;
    */

    // l2reg for cnns
    if(this->config.useCnnsL2reg){
      this->grad->cnnsGrad.l2reg(this->config.lambda*thisStepSize, this->cnns);
    }

    // adam
    Real lr = this->config.adam.lr();
    // lr = (gradNorm > this->clip ? this->clip*lr/gradNorm : lr);// clip
    this->grad->adam(*this, lr/thisStepSize, this->config.adam);

    if(this->miniBatchStepNum <= 50){
      // metrics
      std::cout << " -> Training Loss (per doc) : " << miniBatchMetrics.getLoss();
      std::cout << ", Recall Rate (per doc) : " << miniBatchMetrics.getSentRecall();
    }
    std::cout << std::endl;
    // std::cout << ", lr (per doc) : " << lr;

    this->grad->init();

    // check point
    if (count%this->config.itrThreshold == 0) {
      this->itr += 1;

      // update the learning rate
      //sgd
      /*
      this->lr = lrOrig/(1.0+this->decay*this->itr);// if use adam, this is unnesacrry?
      std::cout << std::endl;
      std::cout << "lr = " << this->lr << std::endl;
      */

      // save train loss & recall
      // metrics
      checkPointMetrics *= 1.0/batchCount;

      ofsTrain << this->itr << "\t" << this->miniBatchStepNum << "\t";
      // metrics
      ofsTrain << checkPointMetrics.getSentLoss() << "\t" << checkPointMetrics.getParLoss() << "\t" << checkPointMetrics.getSecLoss() << "\t";
      ofsTrain << checkPointMetrics.getSentRecall() << "\t" << checkPointMetrics.getParRecall() << "\t" << checkPointMetrics.getSecRecall() << "\t";
      // "REG_POS_SEC_POS_PAR_RATE" "REG_POS_SEC_NEGA_PAR_RATE" "REG_NEGA_SEC_NEGA_PAR_RATE"
      ofsTrain << checkPointMetrics.errorRegRate.coeffRef(0,0) << "\t" << checkPointMetrics.errorRegRate.coeffRef(1,0) << "\t" << checkPointMetrics.errorRegRate.coeffRef(2,0) << "\t";
      ofsTrain << std::endl;

      // metrics
      checkPointMetrics.setZero();

      batchCount = 0;

      // save
      std::ostringstream oss;
      oss << this->config.modelSavePath << this->itr << "ITR.bin";
      this->save(oss.str());

      // validation
      testTimer.start();

      this->dropout(TEST);//for dropout

      for (unsigned id = 0; id < this->config.threadNum; ++id) {
        // metrics
        args[id].metrics.setZero();
      }

      #pragma omp parallel for num_threads(this->config.threadNum) schedule(dynamic) shared(args)
      for (unsigned int i = 0; i < this->validData.size(); ++i) {
        const unsigned int id = omp_get_thread_num();
        args[id].init(this->validData[i]);
        this->valid(args[id]);
      }

      // metrics
      validationMetrics.setZero();

      for (unsigned int id = 0; id < this->config.threadNum; ++id) {
        // metrics
        validationMetrics += args[id].metrics;
      }
      // metrics
      validationMetrics *= 1.0/this->validData.size();

      ofsValid << this->itr << "\t" << this->miniBatchStepNum << "\t";
      // metrics
      ofsValid << validationMetrics.rouge.rouge1.coeffRef(2,0) << "\t";
      ofsValid << validationMetrics.rouge.rouge2.coeffRef(2,0) << "\t";
      ofsValid << validationMetrics.rouge.rougeL.coeffRef(2,0) << "\t";
      ofsValid << validationMetrics.getSentLoss() << "\t" << validationMetrics.getParLoss() << "\t" << validationMetrics.getSecLoss() << "\t";
      ofsValid << validationMetrics.getSentRecall() << "\t" << validationMetrics.getParRecall() << "\t" << validationMetrics.getSecRecall() << "\t";
      ofsValid << validationMetrics.errorRegRate.coeffRef(0,0) << "\t" << validationMetrics.errorRegRate.coeffRef(1,0) << "\t" << validationMetrics.errorRegRate.coeffRef(2,0) << "\t";
      ofsValid << std::endl;

      this->dropout(TRAIN);//for dropout
      testTimer.stop();
      std::cout << "itr = " << this->itr << std::endl;
      std::cout << "Validation time for this check point: " << testTimer.getMin() << " min." << std::endl;
      validationMetrics.print(TEST);// show result
    }//validation
  }//train
  std::cout << std::endl;
  trainTimer.stop();
  std::cout << "Training time for this epoch: " << trainTimer.getMin() << " min." << std::endl;
  // metrics
  trainMetrics *= 1.0/this->trainData.size();
  trainMetrics.print(TRAIN);// show result

  if(epoch == epochEnd){
    for (unsigned int i = 0; i < args.size(); ++i) {
      args[i].clear(TRAIN);
    }
    this->clear();
  }
}

void NNSE::train(const CHECK check){
  // for training

  NNSE::Config config(check);

  Vocabulary v(config.trainDataPath, config.tokenFreqThreshold, config.nameFreqThreshold);

  NNSE nnse(v, config, TRAIN);
  nnse.init(nnse.rnd, nnse.config.scale);

  for (unsigned int epoch = 1; epoch <= config.epochThreshold; ++epoch){
    std::cout << "\n### Epoch " << epoch << std::endl;
    nnse.trainOpenMP(epoch, config.epochThreshold);
  }
}

void NNSE::test(const CHECK check){
  // for testing

  NNSE::Config config(check);
  std::ofstream fout(config.testResultSavePath);
  if(!fout){
    std::cout << config.testResultSavePath << " cannot open" << std::endl;
    assert(fout);
  }

  Vocabulary v(config.trainDataPath, config.tokenFreqThreshold, config.nameFreqThreshold);

  NNSE nnse(v, config, TEST);
  std::cout << "Model Parameters Loading ... ";
  nnse.load(nnse.config.modelLoadPath);
  std::cout << "End" << std::endl;

  nnse.dropout(TEST);//for dropout
  std::cout << "dropout switch ok" << std::endl;

  // note! should follow the construting nnse
  Writer writer(config.writerRootPath, v);// for record system summaries
  std::cout << "writer set ok" << std::endl;

  Metrics testMetrics;  // metrics

  const unsigned int threadNum = config.testThreadNum;
  std::vector<NNSE::ThreadArg> args;

  for (unsigned int i = 0; i < threadNum; ++i) {
    args.push_back(NNSE::ThreadArg(nnse));
    args.back().metrics.setZero();  // metrics
  }
  std::cout << "args set ok" << std::endl;

  std::vector<Metrics> tmpMetrics;  // metrics

  for(size_t i = 0, i_end = nnse.testData.size(); i < i_end; ++i){
    tmpMetrics.push_back(Metrics());
  }

  Timer testTimer;
  testTimer.start();

#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(args)
  for (unsigned int i = 0; i < nnse.testData.size(); ++i) {
    const unsigned int id = omp_get_thread_num();
    args[id].init(nnse.testData[i]);//need
    nnse.test(args[id], tmpMetrics[i]);// metrics
  }
  std::cout << "test end" << std::endl;

#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(writer)
  for (size_t i = 0;  i < nnse.testData.size(); ++i) {

    Article* doc = nnse.testData[i];
    std::vector<int> res;

    for(int j = 0, j_end = doc->resSentNum; j < j_end; ++j) {
      res.push_back(doc->resSent[j]);
    }

    doc->repOrgToken(res, res.size());

    writer.save(doc, res, res.size());
  }
  std::cout << "writer end" << std::endl;

  for (unsigned int id = 0; id < threadNum; ++id) {
    testMetrics += args[id].metrics;    // metrics
  }

  // test log items
  fout
  << "FILE_NAME"     << "\t"
  << "SENTENCE_NUM"  << "\t" << "PARAGRAPH_NUM"                   << "\t" << "SECTION_NUM" << "\t"
  << "ROUGE_1_R"     << "\t" << "ROUGE_1_P"                       << "\t" << "ROUGE_1_F" << "\t"
  << "ROUGE_2_R"     << "\t" << "ROUGE_2_P"                       << "\t" << "ROUGE_2_F" << "\t"
  << "ROUGE_L_R"     << "\t" << "ROUGE_L_P"                       << "\t" << "ROUGE_L_F" << "\t"
  << "CORRECT_NUM" << "\t" << "POS_SEC_POS_PAR_NUM" << "\t" << "POS_SEC_NEGA_PAR_NUM" << "\t" << "NEGA_SEC_NEGA_PAR_NUM" << "\t"
  << "CORRECT_RATE" << "\t" << "POS_SEC_POS_PAR_RATE" << "\t" << "POS_SEC_NEGA_PAR_RATE" << "\t" << "NEGA_SEC_NEGA_PAR_RATE" << "\t"
  << "REG_POS_SEC_POS_PAR_RATE" << "\t" << "REG_POS_SEC_NEGA_PAR_RATE" << "\t" << "REG_NEGA_SEC_NEGA_PAR_RATE" << "\t"
  << "SENT_GOLD_IDS" << "\t" << "SENT_SYSTEM_IDS" << "\t" << "SENT_RECALL" << "\t"
  << "SENT_IDS/SCORE/REG_SCORE" << "\t"
  << "PAR_GOLD_IDS"  << "\t" << "PAR_SYSTEM_IDS"  << "\t" << "PAR_RECALL" << "\t"
  << "PAR_IDS/SCORE/REG_SCORE" << "\t"
  << "SEC_GOLD_IDS"  << "\t" << "SEC_SYSTEM_IDS"  << "\t" << "SEC_RECALL" << "\t"
  << "SEC_IDS/SCORE/REG_SCORE" << "\t"
  << std::endl;

  for(size_t i = 0, i_end = nnse.testData.size(); i < i_end; ++i){
    Article* doc = nnse.testData[i];
    NNSE::Metrics& metric = tmpMetrics[i];
    doc->getSecRegScore();
    doc->getParRegScore();
    doc->getSentRegScore();
    doc->classifyError(doc->resSent, doc->resSentNum);

    // "FILE_NAME"
    fout << doc->fileName << "\t";
    // "SENTENCE_NUM" "PARAGRAPH_NUM" SECTION_NUM"
    fout << doc->bodySentNum << "\t" << doc->bodyPrgNum << "\t" << doc->bodyUsedSecNum << "\t";

    // "ROUGE_1_R" "ROUGE_1_P" "ROUGE_1_F"
    fout << metric.rouge.rouge1.coeffRef(0,0) << "\t" << metric.rouge.rouge1.coeffRef(1,0) << "\t" << metric.rouge.rouge1.coeffRef(2,0) << "\t";
    // "ROUGE_2_R" "ROUGE_2_P"  "ROUGE_2_F"
    fout << metric.rouge.rouge2.coeffRef(0,0) << "\t" << metric.rouge.rouge2.coeffRef(1,0) << "\t" << metric.rouge.rouge2.coeffRef(2,0) << "\t";
    // "ROUGE_L_R" "ROUGE_L_P"  "ROUGE_L_F"
    fout << metric.rouge.rougeL.coeffRef(0,0) << "\t" << metric.rouge.rougeL.coeffRef(1,0) << "\t" << metric.rouge.rougeL.coeffRef(2,0) << "\t";
    // "CORRECT_NUM" "POS_SEC_POS_PAR_NUM" "POS_SEC_NEGA_PAR_NUM" "NEGA_SEC_NEGA_PAR_NUM"
    fout << doc->errorNum.coeffRef(0,0) << "\t" << doc->errorNum.coeffRef(1,0) << "\t" << doc->errorNum.coeffRef(2,0) << "\t" << doc->errorNum.coeffRef(3,0) << "\t";
    // "CORRECT_RATE" "POS_SEC_POS_PAR_RATE" "POS_SEC_NEGA_PAR_RATE" "NEGA_SEC_NEGA_PAR_RATE"
    fout << doc->errorRate.coeffRef(0,0) << "\t" << doc->errorRate.coeffRef(1,0) << "\t" << doc->errorRate.coeffRef(2,0) << "\t" << doc->errorRate.coeffRef(3,0) << "\t";
    // "REG_POS_SEC_POS_PAR_RATE" "REG_POS_SEC_NEGA_PAR_RATE" "REG_NEGA_SEC_NEGA_PAR_RATE"
    fout << doc->errorRegRate.coeffRef(0,0) << "\t" << doc->errorRegRate.coeffRef(1,0) << "\t" << doc->errorRegRate.coeffRef(2,0) << "\t";

    // "SENT_GOLD_IDS"
    for(int j = 0, j_end = doc->sentGoldLabel.size(); j < j_end; ++j){
      fout << doc->sentGoldLabel[j] << " ";
    }
    fout << "\t";
    // "SENT_SYSTEM_IDS"
    for(int j = 0, j_end = doc->resSentNum; j < j_end; ++j){
      fout << doc->resSent[j] << " ";
    }
    fout << "\t";
    // "SENT_RECALL"
    fout << metric.recall.coeffRef(0,0) << "\t";
    // "SENT_SYSTEM_IDS/SCORE/REG_SCORE"
    for(int j = 0, j_end = doc->bodySentNum; j < j_end; ++j){
      fout << doc->bodySent[j]->index << "/" << doc->bodySent[j]->score << "/" << doc->bodySent[j]->regScore << " ";
    }
    fout << "\t";

    // "PAR_GOLD_IDS"
    for(int j = 0, j_end = doc->prgGoldLabel.size(); j < j_end; ++j){
      fout << doc->prgGoldLabel[j] << " ";
    }
    fout << "\t";
    // "PAR_IDS"
    for(int j = 0, j_end = doc->resPrgNum; j < j_end; ++j){
      fout << doc->resPrg[j] << " ";
    }
    fout << "\t";
    // "PAR_RECALL"
    fout << metric.recall.coeffRef(1,0) << "\t";
    // "PAR_SYSTEM_IDS/SCORE/REG_SCORE"
    for(int j = 0, j_end = doc->bodyPrgNum; j < j_end; ++j){
      fout << doc->bodyPrg[j]->index << "/" << doc->bodyPrg[j]->score << "/" << doc->bodyPrg[j]->regScore << " ";
    }
    fout << "\t";

    // "SEC_GOLD_IDS"
    for(int j = 0, j_end = doc->secGoldLabel.size(); j < j_end; ++j){
      fout << doc->secGoldLabel[j] << " ";
    }
    fout << "\t";
    // "SEC_SYSTEM_IDS"
    for(int j = 0, j_end = doc->resSecNum; j < j_end; ++j){
      fout << doc->resSec[j] << " ";
    }
    fout << "\t";
    // "SEC_RECALL"
    fout << metric.recall.coeffRef(2,0) << "\t";
    // "SEC_IDS/SCORE/REG_SCORE"
    for(int j = 0, j_end = doc->bodyUsedSecNum; j < j_end; ++j){
      fout << doc->bodyUsedSec[j]->index << "/" << doc->bodyUsedSec[j]->score << "/" << doc->bodyUsedSec[j]->regScore << " ";
    }
    fout << "\t";
    fout << std::endl;
  }

  std::cout << "Log end" << std::endl;

  fout.close();

  testTimer.stop();
  std::cout << "Testing time for this check point: " << testTimer.getMin() << " min." << std::endl;
  // metrics
  testMetrics *= 1.0/nnse.testData.size();
  testMetrics.print(TEST);// show result

  for (unsigned int i = 0; i < args.size(); ++i) {
    args[i].clear(TEST);
  }
  nnse.clear();
}

void NNSE::dropout(const MODE mode){
    this->encSent.dropout(mode);
    this->encPar.dropout(mode);
    this->encSec.dropout(mode);
    this->decSent.dropout(mode);
    this->decPar.dropout(mode);
    this->decSec.dropout(mode);
    this->classifierSent.dropout(mode);
    this->classifierPar.dropout(mode);
    this->classifierSec.dropout(mode);
}

void NNSE::save(const std::string& file){
  std::ofstream ofs(file.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);

  this->embed.save(ofs);
  this->cnns.save(ofs);
  Utils::save(ofs, this->EOD);// for EOD token
  this->encSent.save(ofs);
  this->encPar.save(ofs);
  this->encSec.save(ofs);
  this->decSent.save(ofs);
  this->decPar.save(ofs);
  this->decSec.save(ofs);
  this->classifierSent.save(ofs);
  this->classifierPar.save(ofs);
  this->classifierSec.save(ofs);
}
void NNSE::load(const std::string& file){
  std::ifstream ifs(file.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  this->embed.load(ifs);
  this->cnns.load(ifs);
  Utils::load(ifs, this->EOD);// for EOD token
  this->encSent.load(ifs);
  this->encPar.load(ifs);
  this->encSec.load(ifs);
  this->decSent.load(ifs);
  this->decPar.load(ifs);
  this->decSec.load(ifs);
  this->classifierSent.load(ifs);
  this->classifierPar.load(ifs);
  this->classifierSec.load(ifs);
}

void NNSE::setMiniBatch(const int beg, const int end){
  for(int i = beg, j = 0; i <= end; ++i, ++j){
    this->miniBatchData[j] = new Article(this->trainData[i], voc);
    this->miniBatchData[j]->repUnkToken();
  }
}
void NNSE::clearMiniBatch(const int beg, const int end){
  for(int i = beg, j = 0; i <= end; ++i, ++j){
    delete this->miniBatchData[j];
    this->miniBatchData[j] = NULL;
  }
}
void NNSE::loadData(const MODE mode){
  std::cout << "Data Loading ... " << std::flush;

  if(mode == TRAIN){
    // training and validataion data
    std::ifstream fin(this->config.trainDataPath);
    if(!fin){
      std::cout << this->config.trainDataPath << " cannot open" << std::endl;
      assert(fin);
    }

    std::string line;
    while(getline(fin, line, '\n')){
      this->trainData.push_back(line);
    }

    this->miniBatchData.resize(this->config.miniBatchSize);

    Article::set(this->config.validDataPath, this->validData, this->voc);
    Article::repUnkToken(this->validData);
  }
  else if(mode == TEST){
        // testing data
    Article::set(this->config.testDataPath, this->testData, this->voc);
    Article::repUnkToken(this->testData);
  }
  std::cout << "End" << std::endl;

  if(mode == TRAIN){
    std::cout << "# of Train Data:\t" << this->trainData.size() << std::endl;
    std::cout << "# of Development Data:\t" << this->validData.size() << std::endl;
  }
  else if(mode == TEST){
    std::cout << "# of Test Data:\t" << this->testData.size() << std::endl;
  }
}
void NNSE::clear(){
  if(this->mode == TRAIN){
    Article::clear(this->validData);
    this->grad->clear();
  }
  else if (this->mode == TEST){
    Article::clear(this->testData);
  }
  std::cout << "\tnnse cleared" << std::endl;
}

NNSE::Grad::Grad(NNSE& nnse):
adamGradHist(0)
{
  this->embedGrad = Embed::Grad(nnse.embed);
  this->cnnsGrad = CNNS<CNNS_KERNEL_NUM>::Grad(nnse.cnns);
  this->EODGrad = VecD::Zero(nnse.EOD.rows());//for EOD
  this->encSentGrad = LSTM::Grad(nnse.encSent);
  this->encParGrad = LSTM::Grad(nnse.encPar);
  this->encSecGrad = LSTM::Grad(nnse.encSec);
  this->zeroSentGrad = VecD::Zero(nnse.zeroSent.rows());// for Zero vecto, dummy
  this->zeroParGrad = VecD::Zero(nnse.zeroPar.rows());// for Zero vecto, dummy
  this->zeroSecGrad = VecD::Zero(nnse.zeroSec.rows());// for Zero vecto, dummy
  this->decSentGrad = LSTM::Grad(nnse.decSent);
  this->decParGrad = LSTM::Grad(nnse.decPar);
  this->decSecGrad = LSTM::Grad(nnse.decSec);
  this->classifierSentGrad = MLP::Grad(nnse.classifierSent);
  this->classifierParGrad = MLP::Grad(nnse.classifierPar);
  this->classifierSecGrad = MLP::Grad(nnse.classifierSec);
  this->init();
}
void NNSE::Grad::init(){
  this->embedGrad.init();
  this->cnnsGrad.init();
  this->EODGrad.setZero();// for EOD
  this->encSentGrad.init();
  this->encParGrad.init();
  this->encSecGrad.init();
  this->zeroSentGrad.setZero();// for Zero vector dummy
  this->zeroParGrad.setZero();// for Zero vector dummy
  this->zeroSecGrad.setZero();// for Zero vector dummy
  this->decSentGrad.init();
  this->decParGrad.init();
  this->decSecGrad.init();
  this->classifierSentGrad.init();
  this->classifierParGrad.init();
  this->classifierSecGrad.init();
}
Real NNSE::Grad::norm(){
  Real res = 0;
  res += this->embedGrad.norm();
  res += this->cnnsGrad.norm();
  res += this->EODGrad.squaredNorm();// for EOD
  res += this->encSentGrad.norm();
  res += this->encParGrad.norm();
  res += this->encSecGrad.norm();
  res += this->decSentGrad.norm();
  res += this->decParGrad.norm();
  res += this->decSecGrad.norm();
  res += this->classifierSentGrad.norm();
  res += this->classifierParGrad.norm();
  res += this->classifierSecGrad.norm();
  return res;
}

void NNSE::Grad::operator += (const NNSE::Grad& grad){
  this->embedGrad += grad.embedGrad;
  this->cnnsGrad += grad.cnnsGrad;
  this->EODGrad += grad.EODGrad; // for EOD
  this->encSentGrad += grad.encSentGrad;
  this->encParGrad += grad.encParGrad;
  this->encSecGrad += grad.encSecGrad;
  this->decSentGrad += grad.decSentGrad;
  this->decParGrad += grad.decParGrad;
  this->decSecGrad += grad.decSecGrad;
  this->classifierSentGrad += grad.classifierSentGrad;
  this->classifierParGrad += grad.classifierParGrad;
  this->classifierSecGrad += grad.classifierSecGrad;
}

void NNSE::Grad::sgd(NNSE& nnse, const Real lr){
  this->embedGrad.sgd(lr, nnse.embed);
  this->cnnsGrad.sgd(lr, nnse.cnns);
  nnse.EOD -= lr * this->EODGrad; // for EOD
  this->encSentGrad.sgd(lr, nnse.encSent);
  this->encParGrad.sgd(lr, nnse.encPar);
  this->encSecGrad.sgd(lr, nnse.encSec);
  this->decSentGrad.sgd(lr, nnse.decSent);
  this->decParGrad.sgd(lr, nnse.decPar);
  this->decSecGrad.sgd(lr, nnse.decSec);
  this->classifierSentGrad.sgd(lr, nnse.classifierSent);
  this->classifierParGrad.sgd(lr, nnse.classifierPar);
  this->classifierSecGrad.sgd(lr, nnse.classifierSec);
}

void NNSE::Grad::adam(NNSE& nnse, const Real lr, const Adam::HyperParam& adam){
  if (this->adamGradHist == 0){
    this->adamGradHist = new NNSE::AdamGrad(nnse);
  }

  this->embedGrad.adam(lr, adam, nnse.embed);
  Adam::adam(this->EODGrad, lr, adam, this->adamGradHist->EOD, nnse.EOD);
  this->cnnsGrad.adam(lr, adam, nnse.cnns);
  this->encSentGrad.adam(lr, adam, nnse.encSent);
  this->encParGrad.adam(lr, adam, nnse.encPar);
  this->encSecGrad.adam(lr, adam, nnse.encSec);
  this->decSentGrad.adam(lr, adam, nnse.decSent);
  this->decParGrad.adam(lr, adam, nnse.decPar);
  this->decSecGrad.adam(lr, adam, nnse.decSec);
  this->classifierSentGrad.adam(lr, adam, nnse.classifierSent);
  this->classifierParGrad.adam(lr, adam, nnse.classifierPar);
  this->classifierSecGrad.adam(lr, adam, nnse.classifierSec);
}
void NNSE::Grad::clear(){
  if (this->adamGradHist != 0){
    delete this->adamGradHist;
    this->adamGradHist = 0;
    std::cout << "NNSE::Grad->adamGradHist clear" << std::endl;
  }
  this->embedGrad.clear();
  this->cnnsGrad.clear();
  this->encSentGrad.clear();
  this->encParGrad.clear();
  this->encSecGrad.clear();
  this->decSentGrad.clear();
  this->decParGrad.clear();
  this->decSecGrad.clear();
  this->classifierSentGrad.clear();
  this->classifierParGrad.clear();
  this->classifierSecGrad.clear();
}

NNSE::ThreadArg::ThreadArg(NNSE& nnse){
  //this->rnd = Rand(nnse.rnd.next());// for dropout
  this->rnd.init(nnse.rnd.next());//新しいRand使うならコッチでいいか
  // metrics
  this->metrics.setZero();

  for (int i = 0, i_end = nnse.config.bodyMaxSentNum; i < i_end; ++i){
    this->encSentState.push_back(new LSTM::State(nnse.encSent));
    this->decSentState.push_back(new LSTM::State(nnse.decSent));

    this->orgSentState.push_back(NNSE::Sentence(nnse));
  }
  for (int i = 0, i_end = nnse.config.bodyMaxParNum; i < i_end; ++i){
    this->encParState.push_back(new LSTM::State(nnse.encPar));
    this->decParState.push_back(new LSTM::State(nnse.decPar));

    this->orgParState.push_back(NNSE::Paragraph(nnse));
  }
  for (int i = 0, i_end = nnse.config.bodyMaxSecNum; i < i_end; ++i){
    this->encSecState.push_back(new LSTM::State(nnse.encSec));
    this->decSecState.push_back(new LSTM::State(nnse.decSec));

    this->orgSecState.push_back(NNSE::Section(nnse));
  }

  if(nnse.mode == TRAIN){
    // train mode
    this->grad = NNSE::Grad(nnse);
  }
}
void NNSE::ThreadArg::init(Article* doc_){
  // Modelに流す前にこいつを実行する必要がある
  //  std::cout << "INSIDE ARG INIT" << std::endl;
  this->doc = doc_;
  this->sentSeqEndIndex = doc_->bodySentNum - 1;
  this->parSeqEndIndex = doc_->bodyPrgNum - 1;
  this->secSeqEndIndex = doc_->bodyUsedSecNum - 1;

  const int i_end = doc_->bodyUsedSecNum;

  this->encEndState = this->encSecState[i_end-1];

  for(int i = 0, jp = 0, ks = 0; i < i_end; ++i){
    //    std::cout << "SEC #" << i << " SET" << std::endl;
    Article::Section* sec = doc_->bodyUsedSec[i];
    const int j_end = sec->prgNum;

    this->orgSecState[i].index = i;
    this->orgSecState[i].sec = sec;
    this->orgSecState[i].encSecState = this->encSecState[i];
    this->orgSecState[i].decSecState = this->decSecState[i];
    this->orgSecState[i].orgParState.resize(j_end);
    this->orgSecState[i].encParState.resize(j_end);
    this->orgSecState[i].decParState.resize(j_end);
    for(int j = 0; j < j_end; ++j, ++jp){
      // std::cout << "\t\tPRG #" << jp << " SET" << std::endl;
      this->orgSecState[i].orgParState[j] = &(this->orgParState[jp]);
      this->orgSecState[i].encParState[j] = this->encParState[jp];
      this->orgSecState[i].decParState[j] = this->decParState[jp];

      Article::Paragraph* par = sec->pPrg[j];
      const int k_end = par->sentNum;

      this->orgParState[jp].index = jp;
      this->orgParState[jp].par = par;
      this->orgParState[jp].encParState = this->encParState[jp];
      this->orgParState[jp].decParState = this->decParState[jp];
      this->orgParState[jp].orgSentState.resize(k_end);
      this->orgParState[jp].encSentState.resize(k_end);
      this->orgParState[jp].decSentState.resize(k_end);
      this->orgParState[jp].classifierSecState = this->orgSecState[i].classifierSecState;
      for(int k = 0; k < k_end; ++k, ++ks){
        // std::cout << "\t\t\t\tSENT #" << ks << " SET" << std::endl;
        this->orgParState[jp].orgSentState[k] = &(this->orgSentState[ks]);
        this->orgParState[jp].encSentState[k] = this->encSentState[ks];
        this->orgParState[jp].decSentState[k] = this->decSentState[ks];

        Article::Sentence* sent =  par->pSent[k];

        this->orgSentState[ks].index = ks;
        this->orgSentState[ks].sent = sent;
        this->orgSentState[ks].encSentState = this->encSentState[ks];
        this->orgSentState[ks].decSentState = this->decSentState[ks];
        this->orgSentState[ks].classifierParState = this->orgParState[jp].classifierParState;
      }
      this->orgParState[jp].encSentEndState = this->orgParState[jp].encSentState.back();
    }
    this->orgSecState[i].encParEndState = this->orgSecState[i].encParState.back();
  }
}

void NNSE::ThreadArg::mask(const NNSE& nnse){
  // for dropout
  for(int i = 0, i_end = this->doc->bodySentNum; i < i_end; ++i){
    this->rnd.setMask(this->encSentState[i]->maskXt, nnse.config.dropoutRateEncSentX);
    this->rnd.setMask(this->decSentState[i]->maskXt, nnse.config.dropoutRateDecSentX);

    this->rnd.setMask(this->orgSentState[i].classifierSentState->maskX0, nnse.config.dropoutRateMLPSent);
    this->rnd.setMask(this->orgSentState[i].classifierSentState->maskX1, nnse.config.dropoutRateMLPSent);
    this->rnd.setMask(this->orgSentState[i].classifierSentState->maskX2, nnse.config.dropoutRateMLPSent);
  }
  for(int i = 0, i_end = this->doc->bodyPrgNum; i < i_end; ++i){
    this->rnd.setMask(this->encParState[i]->maskXt, nnse.config.dropoutRateEncParX);
    this->rnd.setMask(this->decParState[i]->maskXt, nnse.config.dropoutRateDecParX);

    this->rnd.setMask(this->orgParState[i].classifierParState->maskX0, nnse.config.dropoutRateMLPPar);
    this->rnd.setMask(this->orgParState[i].classifierParState->maskX1, nnse.config.dropoutRateMLPPar);
    this->rnd.setMask(this->orgParState[i].classifierParState->maskX2, nnse.config.dropoutRateMLPPar);
  }
  for(int i = 0, i_end = this->doc->bodyUsedSecNum; i < i_end; ++i){
    this->rnd.setMask(this->encSecState[i]->maskXt, nnse.config.dropoutRateEncSecX);
    this->rnd.setMask(this->decSecState[i]->maskXt, nnse.config.dropoutRateDecSecX);

    // this->rnd.setMask(this->orgSecState[i].classifierSecState->maskX0, nnse.config.dropoutRateMLPSec);
    this->rnd.setMask(this->orgSecState[i].classifierSecState->maskX1, nnse.config.dropoutRateMLPSec);
    this->rnd.setMask(this->orgSecState[i].classifierSecState->maskX2, nnse.config.dropoutRateMLPSec);
  }
}
void NNSE::ThreadArg::clear(const MODE mode_){
  for (int i = 0, i_end = this->orgSecState.size(); i < i_end; ++i){
    this->orgSecState[i].clear();
  }
  for (int i = 0, i_end = this->orgParState.size(); i < i_end; ++i){
    this->orgParState[i].clear();
  }
  for (int i = 0, i_end = this->orgSentState.size(); i < i_end; ++i){
    this->orgSentState[i].clear();
  }

  for (int i = 0, i_end = this->encSentState.size(); i < i_end; ++i){
    delete this->encSentState[i];
    delete this->decSentState[i];
  }
  for (int i = 0, i_end = this->encParState.size(); i < i_end; ++i){
    delete this->encParState[i];
    delete this->decParState[i];
  }
  for (int i = 0, i_end = this->encSecState.size(); i < i_end; ++i){
    delete this->encSecState[i];
    delete this->decSecState[i];
  }
  this->encEndState = NULL;
  if(mode_ == TRAIN){
    this->grad.clear();
    std::cout << "\there called" << std::endl;
  }
  std::cout << "\tthread arg clear" << std::endl;
}
void NNSE::ThreadArg::setZero(){
  for(int i = 0, i_end = this->sentSeqEndIndex; i <= i_end; ++i){
    this->encSentState[i]->setZero();
    this->decSentState[i]->setZero();

    this->orgSentState[i].setZero();
  }
  for(int i = 0, i_end = this->parSeqEndIndex; i <= i_end; ++i){
    this->encParState[i]->setZero();
    this->decParState[i]->setZero();

    this->orgParState[i].setZero();
  }
  for(int i = 0, i_end = this->secSeqEndIndex; i <= i_end; ++i){
    this->encSecState[i]->setZero();
    this->decSecState[i]->setZero();

    this->orgSecState[i].setZero();
  }
}

Real NNSEGradChecker::calcLoss(){
  this->arg.metrics.loss.setZero();

  this->nnse.cnnsForward(this->arg, this->flag);
  this->nnse.encoderForward(this->arg);
  this->nnse.decoderForwardTrain(this->arg);

  return this->arg.metrics.loss.sum(); // metrics
}
void NNSEGradChecker::calcGrad(){
  this->arg.metrics.loss.setZero();

  // forward
  this->nnse.cnnsForward(this->arg);
  std::cout << "cnnsForward o.k." << std::endl;

  this->nnse.encoderForward(this->arg);
  std::cout << "encoderForward o.k." << std::endl;

  this->nnse.decoderForwardTrain(this->arg);
  std::cout << "decoderForwardTrain o.k." << std::endl;

  // backward
  this->arg.setZero();
  std::cout << "setZero o.k." << std::endl;

  this->nnse.decoderBackward1(this->arg, this->arg.grad);
  std::cout << "decoderBackward1 o.k." << std::endl;

  this->nnse.encoderBackward1(this->arg, this->arg.grad);
  std::cout << "encoderBackward1 o.k." << std::endl;

  this->nnse.cnnsBackward1(this->arg, this->arg.grad);
  std::cout << "cnnsBackward1 o.k." << std::endl;

  this->nnse.classifierSecBackward2(this->arg, this->arg.grad);
  std::cout << "classifierSecBackward2 o.k." << std::endl;

  this->nnse.classifierParBackward2(this->arg, this->arg.grad);
  std::cout << "classifierParBackward2 o.k." << std::endl;

  this->nnse.classifierSentBackward2(this->arg, this->arg.grad);
  std::cout << "classifierSentBackward2 o.k." << std::endl;

  this->nnse.decoderSecBackward2(this->arg, this->arg.grad);
  std::cout << "decoderSecBackward2 o.k." << std::endl;

  this->nnse.decoderParBackward2(this->arg, this->arg.grad);
  std::cout << "decoderParBackward2 o.k." << std::endl;

  this->nnse.decoderSentBackward2(this->arg, this->arg.grad);
  std::cout << "decoderSentBackward2 o.k." << std::endl;

  this->nnse.encoderSecBackward2(this->arg, this->arg.grad);
  std::cout << "encoderSecBackward2 o.k." << std::endl;

  this->nnse.encoderParBackward2(this->arg, this->arg.grad);
  std::cout << "encoderParBackward2 o.k." << std::endl;

  this->nnse.encoderSentBackward2(this->arg, this->arg.grad);
  std::cout << "encoderSentBackward2 o.k." << std::endl;

  this->nnse.cnnsBackward2(this->arg, this->arg.grad);
  std::cout << "cnnsBackward2 o.k." << std::endl;

  std::cout << "Loss = " << this->arg.metrics.getLoss() << std::endl;
}

void NNSEGradChecker::test(){

  if(typeid(Real) != typeid(double)){
    std::cout << "Real is not double" << std::endl;
    return;
  }

  const CHECK check = GRAD_CHECK;

  NNSE::Config config(check);

  Vocabulary v(config.trainDataPath, config.tokenFreqThreshold, config.nameFreqThreshold);

  NNSE nnse(v, config, TRAIN);
  nnse.init(nnse.rnd, nnse.config.scale);

  NNSE::ThreadArg arg(nnse);

  std::cout << nnse.validData[0]->filePath << std::endl;

  arg.init(nnse.validData[0]);
  arg.mask(nnse);// for dropout

  NNSEGradChecker gc(nnse, arg);

  std::cout << "arg set o.k." << std::endl;

  gc.flag = MaxPooling::CALC_GRAD;

    gc.calcGrad();
    std::cout << "calcGrad o.k." << std::endl;

    gc.flag = MaxPooling::CALC_LOSS;

    std::cout << "classifierSent" << std::endl;
    std::cout << "W0" << std::endl;
    gc.gradCheck(arg.grad.classifierSentGrad.W0, nnse.classifierSent.W0);
    std::cout << "W1" << std::endl;
    gc.gradCheck(arg.grad.classifierSentGrad.W1, nnse.classifierSent.W1);
    std::cout << "W2" << std::endl;
    gc.gradCheck(arg.grad.classifierSentGrad.W2, nnse.classifierSent.W2);
    std::cout << "b0" << std::endl;
    gc.gradCheck(arg.grad.classifierSentGrad.b0, nnse.classifierSent.b0);
    std::cout << "b1" << std::endl;
    gc.gradCheck(arg.grad.classifierSentGrad.b1, nnse.classifierSent.b1);
    std::cout << "b2" << std::endl;
    gc.gradCheck(arg.grad.classifierSentGrad.b2, nnse.classifierSent.b2);

    std::cout << "classifierPar" << std::endl;
    std::cout << "W0" << std::endl;
    gc.gradCheck(arg.grad.classifierParGrad.W0, nnse.classifierPar.W0);
    std::cout << "W1" << std::endl;
    gc.gradCheck(arg.grad.classifierParGrad.W1, nnse.classifierPar.W1);
    std::cout << "W2" << std::endl;
    gc.gradCheck(arg.grad.classifierParGrad.W2, nnse.classifierPar.W2);
    std::cout << "b0" << std::endl;
    gc.gradCheck(arg.grad.classifierParGrad.b0, nnse.classifierPar.b0);
    std::cout << "b1" << std::endl;
    gc.gradCheck(arg.grad.classifierParGrad.b1, nnse.classifierPar.b1);
    std::cout << "b2" << std::endl;
    gc.gradCheck(arg.grad.classifierParGrad.b2, nnse.classifierPar.b2);

    std::cout << "classifierSec" << std::endl;
    std::cout << "W1" << std::endl;
    gc.gradCheck(arg.grad.classifierSecGrad.W1, nnse.classifierSec.W1);
    std::cout << "W2" << std::endl;
    gc.gradCheck(arg.grad.classifierSecGrad.W2, nnse.classifierSec.W2);
    std::cout << "b1" << std::endl;
    gc.gradCheck(arg.grad.classifierSecGrad.b1, nnse.classifierSec.b1);
    std::cout << "b2" << std::endl;
    gc.gradCheck(arg.grad.classifierSecGrad.b2, nnse.classifierSec.b2);

    std::cout << "decSec" << std::endl;
    std::cout << "Wh" << std::endl;
    gc.gradCheck(arg.grad.decSecGrad.Whi, nnse.decSec.Whi);
    gc.gradCheck(arg.grad.decSecGrad.Whf, nnse.decSec.Whf);
    gc.gradCheck(arg.grad.decSecGrad.Who, nnse.decSec.Who);
    gc.gradCheck(arg.grad.decSecGrad.Whu, nnse.decSec.Whu);
    std::cout << "Wx" << std::endl;
    gc.gradCheck(arg.grad.decSecGrad.Wxi, nnse.decSec.Wxi);
    gc.gradCheck(arg.grad.decSecGrad.Wxf, nnse.decSec.Wxf);
    gc.gradCheck(arg.grad.decSecGrad.Wxo, nnse.decSec.Wxo);
    gc.gradCheck(arg.grad.decSecGrad.Wxu, nnse.decSec.Wxu);
    std::cout << "b" << std::endl;
    gc.gradCheck(arg.grad.decSecGrad.bi, nnse.decSec.bi);
    gc.gradCheck(arg.grad.decSecGrad.bf, nnse.decSec.bf);
    gc.gradCheck(arg.grad.decSecGrad.bo, nnse.decSec.bo);
    gc.gradCheck(arg.grad.decSecGrad.bu, nnse.decSec.bu);

    std::cout << "decPar" << std::endl;
    std::cout << "Wh" << std::endl;
    gc.gradCheck(arg.grad.decParGrad.Whi, nnse.decPar.Whi);
    gc.gradCheck(arg.grad.decParGrad.Whf, nnse.decPar.Whf);
    gc.gradCheck(arg.grad.decParGrad.Who, nnse.decPar.Who);
    gc.gradCheck(arg.grad.decParGrad.Whu, nnse.decPar.Whu);
    std::cout << "Wx" << std::endl;
    gc.gradCheck(arg.grad.decParGrad.Wxi, nnse.decPar.Wxi);
    gc.gradCheck(arg.grad.decParGrad.Wxf, nnse.decPar.Wxf);
    gc.gradCheck(arg.grad.decParGrad.Wxo, nnse.decPar.Wxo);
    gc.gradCheck(arg.grad.decParGrad.Wxu, nnse.decPar.Wxu);
    std::cout << "b" << std::endl;
    gc.gradCheck(arg.grad.decParGrad.bi, nnse.decPar.bi);
    gc.gradCheck(arg.grad.decParGrad.bf, nnse.decPar.bf);
    gc.gradCheck(arg.grad.decParGrad.bo, nnse.decPar.bo);
    gc.gradCheck(arg.grad.decParGrad.bu, nnse.decPar.bu);

    std::cout << "decSent" << std::endl;
    std::cout << "Wh" << std::endl;
    gc.gradCheck(arg.grad.decSentGrad.Whi, nnse.decSent.Whi);
    gc.gradCheck(arg.grad.decSentGrad.Whf, nnse.decSent.Whf);
    gc.gradCheck(arg.grad.decSentGrad.Who, nnse.decSent.Who);
    gc.gradCheck(arg.grad.decSentGrad.Whu, nnse.decSent.Whu);
    std::cout << "Wx" << std::endl;
    gc.gradCheck(arg.grad.decSentGrad.Wxi, nnse.decSent.Wxi);
    gc.gradCheck(arg.grad.decSentGrad.Wxf, nnse.decSent.Wxf);
    gc.gradCheck(arg.grad.decSentGrad.Wxo, nnse.decSent.Wxo);
    gc.gradCheck(arg.grad.decSentGrad.Wxu, nnse.decSent.Wxu);
    std::cout << "b" << std::endl;
    gc.gradCheck(arg.grad.decSentGrad.bi, nnse.decSent.bi);
    gc.gradCheck(arg.grad.decSentGrad.bf, nnse.decSent.bf);
    gc.gradCheck(arg.grad.decSentGrad.bo, nnse.decSent.bo);
    gc.gradCheck(arg.grad.decSentGrad.bu, nnse.decSent.bu);

    std::cout << "EOD" << std::endl;
    gc.gradCheck(arg.grad.EODGrad, nnse.EOD);

    std::cout << "encSec" << std::endl;
    std::cout << "Wh" << std::endl;
    gc.gradCheck(arg.grad.encSecGrad.Whi, nnse.encSec.Whi);
    gc.gradCheck(arg.grad.encSecGrad.Whf, nnse.encSec.Whf);
    gc.gradCheck(arg.grad.encSecGrad.Who, nnse.encSec.Who);
    gc.gradCheck(arg.grad.encSecGrad.Whu, nnse.encSec.Whu);
    std::cout << "Wx" << std::endl;
    gc.gradCheck(arg.grad.encSecGrad.Wxi, nnse.encSec.Wxi);
    gc.gradCheck(arg.grad.encSecGrad.Wxf, nnse.encSec.Wxf);
    gc.gradCheck(arg.grad.encSecGrad.Wxo, nnse.encSec.Wxo);
    gc.gradCheck(arg.grad.encSecGrad.Wxu, nnse.encSec.Wxu);
    std::cout << "b" << std::endl;
    gc.gradCheck(arg.grad.encSecGrad.bi, nnse.encSec.bi);
    gc.gradCheck(arg.grad.encSecGrad.bf, nnse.encSec.bf);
    gc.gradCheck(arg.grad.encSecGrad.bo, nnse.encSec.bo);
    gc.gradCheck(arg.grad.encSecGrad.bu, nnse.encSec.bu);

    std::cout << "encPar" << std::endl;
    std::cout << "Wh" << std::endl;
    gc.gradCheck(arg.grad.encParGrad.Whi, nnse.encPar.Whi);
    gc.gradCheck(arg.grad.encParGrad.Whf, nnse.encPar.Whf);
    gc.gradCheck(arg.grad.encParGrad.Who, nnse.encPar.Who);
    gc.gradCheck(arg.grad.encParGrad.Whu, nnse.encPar.Whu);
    std::cout << "Wx" << std::endl;
    gc.gradCheck(arg.grad.encParGrad.Wxi, nnse.encPar.Wxi);
    gc.gradCheck(arg.grad.encParGrad.Wxf, nnse.encPar.Wxf);
    gc.gradCheck(arg.grad.encParGrad.Wxo, nnse.encPar.Wxo);
    gc.gradCheck(arg.grad.encParGrad.Wxu, nnse.encPar.Wxu);
    std::cout << "b" << std::endl;
    gc.gradCheck(arg.grad.encParGrad.bi, nnse.encPar.bi);
    gc.gradCheck(arg.grad.encParGrad.bf, nnse.encPar.bf);
    gc.gradCheck(arg.grad.encParGrad.bo, nnse.encPar.bo);
    gc.gradCheck(arg.grad.encParGrad.bu, nnse.encPar.bu);

    std::cout << "encSent" << std::endl;
    std::cout << "Wh" << std::endl;
    gc.gradCheck(arg.grad.encSentGrad.Whi, nnse.encSent.Whi);
    gc.gradCheck(arg.grad.encSentGrad.Whf, nnse.encSent.Whf);
    gc.gradCheck(arg.grad.encSentGrad.Who, nnse.encSent.Who);
    gc.gradCheck(arg.grad.encSentGrad.Whu, nnse.encSent.Whu);
    std::cout << "Wx" << std::endl;
    gc.gradCheck(arg.grad.encSentGrad.Wxi, nnse.encSent.Wxi);
    gc.gradCheck(arg.grad.encSentGrad.Wxf, nnse.encSent.Wxf);
    gc.gradCheck(arg.grad.encSentGrad.Wxo, nnse.encSent.Wxo);
    gc.gradCheck(arg.grad.encSentGrad.Wxu, nnse.encSent.Wxu);
    std::cout << "b" << std::endl;
    gc.gradCheck(arg.grad.encSentGrad.bi, nnse.encSent.bi);
    gc.gradCheck(arg.grad.encSentGrad.bf, nnse.encSent.bf);
    gc.gradCheck(arg.grad.encSentGrad.bo, nnse.encSent.bo);
    gc.gradCheck(arg.grad.encSentGrad.bu, nnse.encSent.bu);

    std::cout << "cnns" << std::endl;
    for(int i = 0, i_end = CNNS_KERNEL_NUM; i < i_end; ++i){
      std::cout << "i -> " << i << std::endl;
      gc.gradCheck(arg.grad.cnnsGrad.cnn[i].conv.W, nnse.cnns.cnn[i].conv.W);
      gc.gradCheck(arg.grad.cnnsGrad.cnn[i].conv.b, nnse.cnns.cnn[i].conv.b);
    }

    gc.flag = MaxPooling::CALC_GRAD;

    std::cout << "save-load check" << std::endl;
    std::cout << "before saving&loading, loss = " << gc.calcLoss() << std::endl;

    const std::string path = "./tmp.bin";
    nnse.save(path);
    nnse.load(path);

    std::cout << "after saving&loading , loss = " << gc.calcLoss() << std::endl;
}
