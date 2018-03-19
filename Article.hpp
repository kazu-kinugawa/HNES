#pragma once

#include <memory>

#include "Vocabulary.hpp"
#include "XmlParser.hpp"
#include "Matrix.hpp"
#include "ROUGE.hpp"

// edit on 2018/01/13

class Article{
public:
  class Sentence;
  class Paragraph;
  class Section;
  class Kwd;
  class Counter;
  class Arg;

  const std::string fileName;
  const std::string filePath;
  const Real compressibility;// output summary / total body word num
  const bool splWordNum;// limit by word number or not

  /* Vocabulary */
  Vocabulary& voc;
  const int unkIndex;

  /* title */
  bool hasTitle;
  std::vector<int> titleArray;
  std::unordered_set<int> title;

  /* keywords */
  bool hasKwd;
  std::vector<Article::Kwd*> kwdGroup;// -1 indicates separeter
  int kwdNum;

  /* abstract */
  int abstSentNum;
  int abstPrgNum;
  int abstSecNum;

  // int totalAbstUniNum;
  // int totalAbstBiNum;// for evaluation

  Article::Section* abstRoot;

  std::vector<Article::Sentence*> abstSent;
  std::vector<Article::Paragraph*> abstPrg;
  std::vector<Article::Section*> abstSec;

  /* body */
  int bodySentNum;
  int bodyPrgNum;
  int bodySecNum;
  int bodyUsedSecNum;

  int bodyMaxWordNum, bodyMinWordNum;
  int resUpperUniNum;// for evaluation
  int totalBodyUniNum;

  std::vector<Article::Sentence*> bodySent;
  std::vector<Article::Paragraph*> bodyPrg;
  std::vector<Article::Section*> bodySec;
  std::vector<Article::Section*> bodyUsedSec;

  Article::Section* bodyRoot;

  /*  gold standard set of sentences */
  /*
  std::vector<int> goldLabel;
  std::unordered_set<int> goldLabelSet;
  int posLabelNum, negaLabelNum;
  */

  /*  gold standard set of sentences */
  std::vector<int> sentGoldLabel;
  std::unordered_set<int> sentGoldLabelSet;
  int sentPosLabelNum;
  int sentNegaLabelNum;

  /*  gold standard set of paragraphs */
  std::vector<int> prgGoldLabel;
  std::unordered_set<int> prgGoldLabelSet;
  int prgPosLabelNum;
  int prgNegaLabelNum;

  /*  gold standard set of sections */
  std::vector<int> secGoldLabel;
  std::unordered_set<int> secGoldLabelSet;
  int secPosLabelNum;
  int secNegaLabelNum;

  /* result */
  // std::vector<int> res;
  // int resSentNum;
  // int resUniNum;
  // int resBiNum;
  //std::unordered_set<int> resSet;

  std::vector<int> resSent;
  int resSentNum;
  int resUniNum;
  int resBiNum;

  std::vector<int> resPrg;
  int resPrgNum;

  std::vector<int> resSec;
  int resSecNum;

  std::vector<Article::Sentence*> bodySentCopy;

  /* counter */
  Article::Counter* counter; // for evaluation

  /* 逆引きtable */
  // std::vector<int> sent2sec;
  // std::vector<int> sent2prg;
  // std::vector<int> prg2sec;
  std::vector<Article::Section*> sent2sec;
  std::vector<Article::Paragraph*> sent2prg;
  VecI errorNum;
  VecD errorRate;
  VecD errorRegRate;
  
  Article(const std::string& _filePath, Vocabulary& _voc);
  ~Article();
  void buildTree(const XmlElem* elem, Article::Section* node, const bool isChild, Article::Arg& arg);
  void rerank();
  int sweep(const std::vector<int>& word);
  void appEosToken();
  void repUnkToken();
  void repOrgToken();
  void repUnkToken(const std::vector<int>& res, const int len);
  void repOrgToken(const std::vector<int>& res, const int len);
  void print(std::ofstream& fout);
  Real getSentRecall(const std::vector<int>& v, const int len);
  Real getPrgRecall(const std::vector<int>& v, const int len);
  Real getSecRecall(const std::vector<int>& v, const int len);
  static bool cmp(const Article::Sentence* sent1, const Article::Sentence* sent2);
  static void LEAD(const std::string& testPathList, unsigned int threadNum);
  static void optimal(const std::string& testPathList, const unsigned int threadNum);
  static void set(const std::string& pathList, std::vector<Article*>& docs, Vocabulary& voc);
  static void clear(std::vector<Article*>& docs);
  static void appEosToken(std::vector<Article*>& docs);
  static void repUnkToken(std::vector<Article*>& docs);
  static void repOrgToken(std::vector<Article*>& docs);
  static std::string getLowerCase(const char* str){
    std::string res(str);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
  }
  static void getLowerCase(std::string& str){
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  }
  static void delGoldLabelNode(const std::string& pathList, const char* target);
  static void delInfoNode(const std::string& pathList);
  static void appInfoNode(const std::string& pathList);
  static void getAveRate(const std::string& pathList);
  static void all2txt(const std::string& pathList, const std::string& output);
  static void lcs(const std::vector<int>& X, const std::vector<int>& Y, std::vector<int>& res);
  Real getRougeL(const std::vector<int>& res, const int len, ROUGE& eval);
  void evaluate(const std::vector<int>& res, const int len, ROUGE& eval);
  ROUGE evaluate(const std::vector<int>& res, const int len);

  void getSentRegScore();
  void getParRegScore();
  void getSecRegScore();
  void classifyError(const std::vector<int> array, const int array_size);
};

class Article::Kwd{
public:
  std::vector<int> phrase;
  int len;
  int cmpSize;
  Kwd(const XmlElem* elem, Vocabulary& voc);
  int sweep(const std::vector<int>& word);
};

class Article::Sentence{
public:
  int index;

  std::vector<int> word;
  int wordNum;
  int biNum;

  Real label;
  Real score;
  Real regScore;

  std::vector<std::pair<int,int> > unkList;

  /*
  std::unordered_map<int, int> uniCount;
  std::unordered_map<std::pair<int,int>, int> biCount;

  int prgId;
  Article::Paragraph* parentPrg;
  int secId;
  Article::Section* parentSec;
  int secName;
  */

  // Sentence(Article::Paragraph* _parentPrg, const XmlElem* sentElem, Article::Arg& arg);
  Sentence(const XmlElem* sentElem, Article::Arg& arg);
  void repOrgToken();
  void repUnkToken(const int unkIndex);
  void appEosToken(const int eosIndex);
};

class Article::Paragraph{
public:
  int index;

  std::vector<Article::Sentence*> pSent;
  std::vector<int> sent;
  int sentNum;

  Real label;
  Real score;
  Real regScore;

  /*
  int secId;
  Article::Section* parentSec;
  int secName;
  */

  // Paragraph(Article::Section* _parentSec, const XmlElem* prgElem, Article::Arg& arg);
  Paragraph(const XmlElem* prgElem, Article::Arg& arg);
  void disconnect();
  void salvage(const XmlElem* sentElem, Article::Arg& arg);
  void traverse(const XmlElem* elem, Article::Arg& arg);
};

class Article::Section{
public:
  int index;// section index
  int name;// section name

  std::vector<Article::Sentence*> pSent;
  std::vector<int> sent;
  int sentNum;

  std::vector<Article::Paragraph*> pPrg;
  std::vector<int> prg;
  int prgNum;

  Article::Section* left;
  Article::Section* right;

  Real label;
  Real score;
  Real regScore;

  Section():left(NULL), right(NULL), sentNum(0), prgNum(0), index(-1), name(-1) {};// fot root
  Section(const XmlElem* elem, Article::Arg& arg);
  void disconnect();
  void salvage(const XmlElem* prgElem, Article::Arg& arg);
};

class Article::Arg{
  // for decreasing the number of arguments
public:
  std::vector<Article::Sentence*>& sent;
  std::vector<Article::Paragraph*>& prg;
  std::vector<Article::Section*>& sec;

  Vocabulary& voc;

  Arg(std::vector<Article::Sentence*>& _sent, std::vector<Article::Paragraph*>& _prg, std::vector<Article::Section*>& _sec, Vocabulary& _voc):
  sent(_sent), prg(_prg), sec(_sec), voc(_voc){}
};

class Article::Counter{
public:
  std::unordered_map<int, int> uniConverter;
  MatI uniCount;// col(0) indicates abst, col(1) indicates res
  int totalAbstUniNum;

  std::unordered_map<std::pair<int, int>, int> biConverter;
  MatI biCount;// col(0) indicates abst, col(1) indicates res
  int totalAbstBiNum;

  std::vector<int> abstWordSeq;

  Counter(const Article& article);
  ~Counter(){
    this->uniCount = MatI();
    this->biCount = MatI();
  }
};
