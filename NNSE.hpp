#pragma once

#include "CNNS.hpp"
#include "LSTM.hpp"
#include "Article.hpp"
#include "Vocabulary.hpp"
#include "Adam.hpp"
#include "LossFunc.hpp"
#include "Embed.hpp"
#include "MLP.hpp"

constexpr int CNNS_KERNEL_NUM = 6;

class NNSE{
public:
  class Grad;
  class ThreadArg;
  class Config;
  class AdamGrad;
  class Sentence;
  class Paragraph;
  class Section;

  class Metrics{
  public:
    ROUGE rouge;
    VecD loss;
    VecD recall;
    // "CORRECT_RATE" "POS_SEC_POS_PAR_RATE" "POS_SEC_NEGA_PAR_RATE" "NEGA_SEC_NEGA_PAR_RATE"
    VecD errorRate;
    VecD errorRegRate;
    // "POS_SEC_POS_PAR_RATE" "POS_SEC_NEGA_PAR_RATE" "NEGA_SEC_NEGA_PAR_RATE"

    Metrics(){
      this->loss = VecD::Zero(3);
      this->recall = VecD::Zero(3);
      this->errorRate = VecD::Zero(4);
      this->errorRegRate = VecD::Zero(3);      
      this->setZero();
    }
    void setZero(){
        this->rouge.init();
        this->loss.setZero();
        this->recall.setZero();
        this->errorRate.setZero();
        this->errorRegRate.setZero();
    }
    Real getLoss(){
      return this->loss.sum();
    }
    Real getSentLoss(){
      return this->loss.coeffRef(0,0);
    }
    Real getParLoss(){
      return this->loss.coeffRef(1,0);
    }
    Real getSecLoss(){
      return this->loss.coeffRef(2,0);
    }
    Real getSentRecall(){
      return this->recall.coeffRef(0,0);
    }
    Real getParRecall(){
      return this->recall.coeffRef(1,0);
    }
    Real getSecRecall(){
      return this->recall.coeffRef(2,0);
    }
    void operator += (const NNSE::Metrics& target){
      this->rouge += target.rouge;
      this->loss += target.loss;
      this->recall += target.recall;
      this->errorRate += target.errorRate;
      this->errorRegRate += target.errorRegRate;
    }
    void operator *= (const Real val){
      this->rouge *= val;
      this->loss *= val;
      this->recall *= val;
      this->errorRate *= val;
      this->errorRegRate *= val;
    }
    void print(const MODE mode_){
      if(mode_ != TRAIN){
        std::cout << "ROUGE_1_F\t" << this->rouge.rouge1.coeffRef(2,0) << std::endl;
        std::cout << "ROUGE_2_F\t" << this->rouge.rouge2.coeffRef(2,0) << std::endl;
        std::cout << "ROUGE_L_F\t" << this->rouge.rougeL.coeffRef(2,0) << std::endl;
      }
      std::cout << "LOSS SENT\t"   << this->loss.coeffRef(0,0) << std::endl;
      std::cout << "LOSS PAR\t"    << this->loss.coeffRef(1,0) << std::endl;
      std::cout << "LOSS SEC\t"    << this->loss.coeffRef(2,0) << std::endl;
      std::cout << "RECALL SENT\t" << this->recall.coeffRef(0,0) << std::endl;
      std::cout << "RECALL PAR\t"  << this->recall.coeffRef(1,0) << std::endl;
      std::cout << "RECALL SEC\t"  << this->recall.coeffRef(2,0) << std::endl;
      std::cout << "REG_POS_SEC_POS_PAR_RATE\t" << this->errorRegRate.coeffRef(0,0) << std::endl;
      std::cout << "REG_POS_SEC_NEGA_PAR_RATE\t" << this->errorRegRate.coeffRef(1,0) << std::endl;
      std::cout << "REG_NEGA_SEC_NEGA_PAR_RATE\t" << this->errorRegRate.coeffRef(2,0) << std::endl;
    }
  };

  NNSE(Vocabulary& voc_, NNSE::Config& _config, const MODE mode_);

  /* parameters */
  NNSE::Config& config;

  unsigned int itr;
  unsigned int miniBatchStepNum;
  const MODE mode;

  Vocabulary& voc;
  std::vector<std::string> trainData;
  std::vector<Article*> miniBatchData;
  std::vector<Article*> validData, testData;

  /* variable */
  Rand rnd;

  Embed embed;

  CNNS<CNNS_KERNEL_NUM> cnns;

  LSTM encSent;
  LSTM encPar;
  LSTM encSec;

  VecD EOD;// end of document

  const VecD zeroSec;// zero vector
  LSTM decSec;
  MLP classifierSec;

  const VecD zeroPar;// zero vector
  LSTM decPar;
  MLP classifierPar;

  const VecD zeroSent;// zero vector
  LSTM decSent;
  MLP classifierSent;

  LossFunc::BinaryCrossEntropy lossFunc;

  NNSE::Grad* grad;
  NNSE::Metrics metrics;

  void init(Rand& rnd, const Real scale);

  void cnnsForward(NNSE::ThreadArg& arg);
  void cnnsForward(NNSE::ThreadArg& arg, const MaxPooling::GRAD_CHECK flag);// for gradchecking
  void cnnsBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad);
  void cnnsBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void encoderSentForward(NNSE::Paragraph& par);
  void encoderSentBackward1(NNSE::Paragraph& par, NNSE::Grad& grad);
  void encoderSentBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void encoderParForward(NNSE::Section& sec);
  void encoderParBackward1(NNSE::Section& sec, NNSE::Grad& grad);
  void encoderParBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void encoderSecForward(NNSE::ThreadArg& arg);
  void encoderSecBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad);
  void encoderSecBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void encoderForward(NNSE::ThreadArg& arg);// package
  void encoderBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad);// package

  Real classifierSentForward(const int i, NNSE::Paragraph& par);
  void classifierSentBackward1(const int i, NNSE::Paragraph& par, NNSE::Grad& grad);
  void classifierSentBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  Real classifierParForward(const int i, NNSE::Section& sec);
  void classifierParBackward1(const int i, NNSE::Section& sec, NNSE::Grad& grad);
  void classifierParBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  Real classifierSecForward(const int i, NNSE::ThreadArg& arg);
  void classifierSecBackward1(const int i, NNSE::ThreadArg& arg, NNSE::Grad& grad);
  void classifierSecBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void decoderSecForwardTrain(NNSE::ThreadArg& arg);
  void decoderSecForwardTest(NNSE::ThreadArg& arg);
  void decoderSecBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad);
  void decoderSecBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void decoderParForwardTrain(NNSE::Section& sec, NNSE::ThreadArg& arg);
  void decoderParForwardTest(NNSE::Section& sec, NNSE::ThreadArg& arg);
  void decoderParBackward1(NNSE::Section& sec, NNSE::Grad& grad);
  void decoderParBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void decoderSentForwardTrain(NNSE::Paragraph& par, NNSE::ThreadArg& arg);
  void decoderSentForwardTest(NNSE::Paragraph& par, NNSE::ThreadArg& arg);
  void decoderSentBackward1(NNSE::Paragraph& par, NNSE::Grad& grad);
  void decoderSentBackward2(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  // package
  void decoderForwardTrain(NNSE::ThreadArg& arg);
  void decoderForwardTest(NNSE::ThreadArg& arg);
  void decoderBackward1(NNSE::ThreadArg& arg, NNSE::Grad& grad);

  void train(NNSE::ThreadArg& arg);
  void valid(NNSE::ThreadArg& arg);
  void test(NNSE::ThreadArg& arg, NNSE::Metrics& targetMetrics);

  void trainOpenMP(const unsigned int epoch, const unsigned int epochEnd);

  void loadInitEmbMat();
  void save(const std::string& file);
  void load(const std::string& file);
  void setMiniBatch(const int beg, const int end);
  void clearMiniBatch(const int beg, const int end);
  void loadData(const MODE mode_);

  static void train(const CHECK check);
  static void test(const CHECK check);

  /* for dropout */
  void dropout(const MODE mode_);
  void clear();
};

class NNSE::Grad{
public:
  /* adam */
  NNSE::AdamGrad* adamGradHist;

  Embed::Grad embedGrad;

  CNNS<CNNS_KERNEL_NUM>::Grad cnnsGrad;

  VecD EODGrad;// for EOD token

  LSTM::Grad encSentGrad;
  LSTM::Grad encParGrad;
  LSTM::Grad encSecGrad;

  VecD zeroSentGrad;// for ZERO token, dummy
  VecD zeroParGrad;// for ZERO token, dummy
  VecD zeroSecGrad;// for ZERO token, dummy

  LSTM::Grad decSentGrad;
  LSTM::Grad decParGrad;
  LSTM::Grad decSecGrad;

  MLP::Grad classifierSentGrad;
  MLP::Grad classifierParGrad;
  MLP::Grad classifierSecGrad;

  Grad():adamGradHist(0){};
  Grad(NNSE& nnse);
  void init();
  Real norm();
  void operator += (const NNSE::Grad& grad);
  void sgd(NNSE& nnse, const Real lr);
  void adam(NNSE& nnse, const Real lr, const Adam::HyperParam& hp);
  void clear();
};

class NNSE::AdamGrad{
public:
  Adam::Grad<VecD> EOD;// EOD token
  AdamGrad(const NNSE& nnse){
    this->EOD = Adam::Grad<VecD>(nnse.EOD);
  }
};

class NNSE::ThreadArg{
public:
  Rand rnd;// for dropout
  // metrics
  NNSE::Metrics metrics;

  Article* doc;// reference
  int sentSeqEndIndex;
  int parSeqEndIndex;
  int secSeqEndIndex;

  std::vector<NNSE::Sentence> orgSentState;
  std::vector<NNSE::Paragraph> orgParState;
  std::vector<NNSE::Section> orgSecState;

  std::vector<LSTM::State*> encSentState;
  std::vector<LSTM::State*> encParState;
  std::vector<LSTM::State*> encSecState;

  LSTM::State* encEndState;
  std::vector<LSTM::State*> decSecState;
  std::vector<LSTM::State*> decParState;
  std::vector<LSTM::State*> decSentState;

  NNSE::Grad grad;

  ThreadArg(NNSE& nnse);
  void mask(const NNSE& nnse);// for dropout
  void init(Article* _doc);
  void clear(const MODE mode_);
  void setZero();
};

class NNSE::Sentence{
public:
  int index;// index of sentence
  // Real weight;
  Article::Sentence* sent;// reference

  Embed::State<Embed::IM>* embedState;
  CNNS<CNNS_KERNEL_NUM>::State* cnnsState;// pseudo sentence vector
  MLP::State* classifierSentState;
  LossFunc::State* lossFuncSentState;

  LSTM::State* encSentState;// reference for classifier
  LSTM::State* decSentState;// reference for classifier

  MLP::State* classifierParState;// reference for classifier

  Sentence(const NNSE& nnse){
    this->embedState = new Embed::State<Embed::IM>(nnse.embed);
    this->cnnsState = new CNNS<CNNS_KERNEL_NUM>::State(nnse.cnns);
    this->classifierSentState = new MLP::State(nnse.classifierSent);
    this->lossFuncSentState = new LossFunc::State;
  }
  void setZero(){
    this->embedState->setZero();
    this->cnnsState->setZero();
    this->classifierSentState->setZero();
    // this->encSentState->setZero();
    // this->decSentState->setZero();
  }
  void clear(){
    this->sent = NULL;
    delete this->embedState;
    delete this->cnnsState;
    delete this->classifierSentState;
    delete this->lossFuncSentState;
    this->encSentState = NULL;
    this->decSentState = NULL;
    this->classifierParState = NULL;
  }
};

class NNSE::Paragraph{
public:
  int index;// index of paragraph
  // Real weight;
  Article::Paragraph* par;// reference

  MLP::State* classifierParState;
  LossFunc::State* lossFuncParState;

  LSTM::State* encSentEndState;// pseudo par vector

  LSTM::State* encParState;// reference for classifier
  LSTM::State* decParState;// reference for classifier

  MLP::State* classifierSecState;// reference for classifier (parent node)

  std::vector<NNSE::Sentence*> orgSentState;// reference
  std::vector<LSTM::State*> encSentState;// reference
  std::vector<LSTM::State*> decSentState;// reference // everytime resized

  size_t sentSeqEnd;// encSentState.size() - 1

  Paragraph(const NNSE& nnse){
    this->classifierParState = new MLP::State(nnse.classifierPar);
    this->lossFuncParState = new LossFunc::State;
  }
  void setZero(){
    this->classifierParState->setZero();
  }
  void clear(){
    this->par = NULL;
    delete this->classifierParState;
    delete this->lossFuncParState;
    this->encParState = NULL;
    this->decParState = NULL;
    for(int i = 0, i_end = this->orgSentState.size(); i < i_end; ++i){
      this->orgSentState[i] = NULL;
      this->encSentState[i] = NULL;
      this->decSentState[i] = NULL;
    }
    this->classifierSecState = NULL;
    this->encSentEndState = NULL;
  }
};

class NNSE::Section{
public:
  int index;// index of paragraph
  // Real weight;
  Article::Section* sec;// reference

  MLP::State* classifierSecState;
  LossFunc::State* lossFuncSecState;

  LSTM::State* encParEndState;// pseudo sec vector

  LSTM::State* encSecState;// reference for classifier
  LSTM::State* decSecState;// reference for classifier

  std::vector<NNSE::Paragraph*> orgParState;// reference
  std::vector<LSTM::State*> encParState;// reference
  std::vector<LSTM::State*> decParState;// reference

  Section(const NNSE& nnse){
    this->classifierSecState = new MLP::State(nnse.classifierSec);
    this->lossFuncSecState = new LossFunc::State;
  }
  void setZero(){
    this->classifierSecState->setZero();
  }
  void clear(){
    this->sec = NULL;
    delete this->classifierSecState;
    delete this->lossFuncSecState;
    this->encSecState = NULL;
    this->decSecState = NULL;
    for(int i = 0, i_end = this->orgParState.size(); i < i_end; ++i){
      this->orgParState[i] = NULL;
      this->encParState[i] = NULL;
      this->decParState[i] = NULL;
    }
    this->encParEndState = NULL;
  }
};

class NNSEGradChecker : public GradChecker{
public:
  NNSE& nnse;

  MaxPooling::GRAD_CHECK flag;

  NNSE::ThreadArg& arg;

  NNSEGradChecker(NNSE& nnse_, NNSE::ThreadArg& arg_):nnse(nnse_), arg(arg_), flag(MaxPooling::CALC_GRAD){};

  Real calcLoss();
  void calcGrad();
  static void test();
};

class NNSE::Config{
public:
  Config(const CHECK check);

  std::string base;
  std::string date;
  std::string trainDataPath, validDataPath, testDataPath;
  std::string validLogPath, trainLogPath, modelSavePath;
  std::string initEmbMatLoadPath;
  std::string modelLoadPath;
  std::string testResultSavePath;
  std::string writerRootPath;

  bool useWord2vec;

  int wordEmbDim;
  int sentEmbDim;
  int docEmbDim;
  Real scale;

  int bodyMaxWordNum, bodyMaxSentNum, bodyMaxParNum, bodyMaxSecNum;

  int cnnsInputDim, cnnsOutputDim;

  int encSentInputDim, encSentOutputDim;
  int encParInputDim, encParOutputDim;
  int encSecInputDim, encSecOutputDim;

  int decSentInputDim, decSentOutputDim;
  int decParInputDim, decParOutputDim;
  int decSecInputDim, decSecOutputDim;

  int classifierHiddenDim;

  unsigned int threadNum;
  unsigned int testThreadNum;
  unsigned int miniBatchSize;
  Real learningRate;
  Real clip;
  Real decay;
  Adam::HyperParam adam;// adam;

  unsigned int itrThreshold;
  unsigned int epochThreshold;

  Real lambda;
  bool useCnnsL2reg;

  Real dropoutRateEncSentX, dropoutRateEncParX, dropoutRateEncSecX;
  Real dropoutRateDecSentX, dropoutRateDecParX, dropoutRateDecSecX;
  Real dropoutRateMLPSent, dropoutRateMLPPar, dropoutRateMLPSec;

  unsigned int tokenFreqThreshold;
  unsigned int nameFreqThreshold;

  unsigned long seed;
};
