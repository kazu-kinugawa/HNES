#include "Article.hpp"
#include <cstdlib>
#include <omp.h>
#include <cassert>

// edit on 2018/01/13

Article::Article(const std::string& _filePath, Vocabulary& _voc):
  filePath(_filePath),
  voc(_voc), unkIndex(_voc.unkIndex),
  // compressibility(0.0640319), // <= path-list 170601
  compressibility(0.06),
  splWordNum(true),
  fileName(_filePath.substr(_filePath.rfind("/") + 1, _filePath.rfind(".xml") - _filePath.rfind("/") - 1))
{
  // std::cout << _filePath << std::endl;
  XmlDoc xml;
  const XmlError loadCheck = xml.LoadFile(_filePath.c_str());
  if(loadCheck != XmlSuccess){
    std::cout << _filePath << " cannot open" << std::endl;
    assert(loadCheck == XmlSuccess);
  }

  XmlElem* article = xml.FirstChildElement("article");// root
  XmlElem* front = article->FirstChildElement("front");

  // title
  if(front->Attribute("has-title", "1")){
    this->hasTitle = true;
    std::string title(front->FirstChildElement("article-title")->GetText());
    Article::getLowerCase(title);
    std::istringstream stream(title);
    std::string token;
    int num;
    while(getline(stream, token, ' ')){
      if(this->voc.token2index.count(token)){
        num = this->voc.token2index.at(token);
      }
      else{
        num = this->voc.unkIndex;
      }
      this->titleArray.push_back(num);// vector
      this->title.insert(num);// unordered_set
    }
  }
  else{
    this->hasTitle = false;
  }

  // kwd
  this->kwdNum = 0;
  if( front->Attribute("has-kwd","1") ){
    this->hasKwd = true;
    XmlElem* kwd_group = front->FirstChildElement("kwd-group");

    for(XmlElem* kwdElem = kwd_group->FirstChildElement("kwd"); kwdElem != NULL; kwdElem = kwdElem->NextSiblingElement("kwd")){
      this->kwdGroup.push_back( new Article::Kwd(kwdElem, this->voc) );
      ++this->kwdNum;
    }
  }
  else{
    this->hasKwd = false;
  }

  Article::Arg abstArg(this->abstSent, this->abstPrg, this->abstSec, _voc);
  this->abstRoot = new Article::Section();//root

  XmlElem* abst = front->FirstChildElement("abstract");

  this->buildTree(abst->FirstChildElement("sec"), this->abstRoot, true, abstArg);

  this->abstSentNum = (int)this->abstSent.size();
  this->abstPrgNum = (int)this->abstPrg.size();
  this->abstSecNum = (int)this->abstSec.size();

  Article::Arg bodyArg(this->bodySent, this->bodyPrg, this->bodySec, _voc);

  this->bodyRoot = new Article::Section();//root

  XmlElem* body = article->FirstChildElement("body");

  this->buildTree(body->FirstChildElement("sec"), this->bodyRoot, true, bodyArg);

  this->bodySentNum = (int)this->bodySent.size();
  this->bodyPrgNum = (int)this->bodyPrg.size();
  this->bodySecNum = (int)this->bodySec.size();

  this->resSent.resize(this->bodySent.size());
  this->bodySentCopy.resize(this->bodySent.size());
  std::copy(this->bodySent.begin(), this->bodySent.end(), this->bodySentCopy.begin());

  this->resPrg.resize(this->bodyPrg.size());
  this->resSec.resize(this->bodySec.size());

  /* setup upper word num for output summary */
  this->totalBodyUniNum = 0;
  for(int i = 0, i_end = this->bodySentNum; i < i_end; ++i){
    this->totalBodyUniNum += this->bodySent[i]->wordNum;
  }
  this->resUpperUniNum = this->totalBodyUniNum * this->compressibility;

  /* copy only used sec pointer */
  this->bodyUsedSecNum = 0;
  for(int i = 0; i < this->bodySecNum; ++i){
    if(this->bodySec[i]->sentNum > 0){
      this->bodyUsedSec.push_back(this->bodySec[i]);
      ++this->bodyUsedSecNum;
    }
  }

  /* making table */
  this->sent2sec.resize(this->bodySentNum);
  this->sent2prg.resize(this->bodySentNum);
  for(int i = 0, is = 0; i < this->bodyUsedSecNum; ++i){
    Article::Section* sec = this->bodyUsedSec[i];
    for(int j = 0, j_end = sec->prgNum; j < j_end; ++j){
      Article::Paragraph* prg = sec->pPrg[j];
      for(int k = 0, k_end = prg->sentNum; k < k_end; ++k, ++is){
          this->sent2prg[is] = prg;
          this->sent2sec[is] = sec;
      }
    }
  }

  /* counter */
  this->counter = new Article::Counter(*this);

  //XmlElem* gold = article->FirstChildElement("info")->FirstChildElement("gold-label-nksk-170603");
  XmlElem* gold = article->FirstChildElement("info")->FirstChildElement("gold-label-nksk-170720");
  // XmlElem* gold = article->FirstChildElement("info")->FirstChildElement("oracle");

  this->sentPosLabelNum = atoi(gold->Attribute("pos-label-num"));
  this->sentNegaLabelNum = atoi(gold->Attribute("nega-label-num"));

  if(this->sentPosLabelNum > 0){
    const std::string line(gold->GetText());
    std::istringstream stream(line);
    int num;
    std::string index;
    std::stringstream ss;
    while(getline(stream, index, ' ')) {
      ss.str(index);
      ss >> num;
      this->bodySent[num]->label = 1.0;
      this->sentGoldLabel.push_back(num);
      this->sentGoldLabelSet.insert(num);
      ss.str("");
      ss.clear(std::stringstream::goodbit);// <- necessary

      const int goldPrgIndex = this->sent2prg[num]->index;
      this->sent2prg[num]->label = 1.0;
      this->prgGoldLabelSet.insert(goldPrgIndex);

      const int goldSecIndex = this->sent2sec[num]->index;
      this->sent2sec[num]->label = 1.0;
      // this->secGoldLabel.push_back(goldSecIndex);
      this->secGoldLabelSet.insert(goldSecIndex);
    }

    this->prgPosLabelNum = (int)this->prgGoldLabelSet.size();
    this->prgNegaLabelNum = this->bodyPrgNum - this->prgPosLabelNum;
    for(auto itr = this->prgGoldLabelSet.begin(), itrEnd = this->prgGoldLabelSet.end(); itr != itrEnd; ++itr){
      this->prgGoldLabel.push_back(*itr);
    }
    std::sort(&(this->prgGoldLabel[0]), &(this->prgGoldLabel[this->prgPosLabelNum]) ); // 0~resNum -1のソート

    this->secPosLabelNum = (int)this->secGoldLabelSet.size();
    this->secNegaLabelNum = this->bodySecNum - this->secPosLabelNum;
    for(auto itr = this->secGoldLabelSet.begin(), itrEnd = this->secGoldLabelSet.end(); itr != itrEnd; ++itr){
      this->secGoldLabel.push_back(*itr);
    }
    std::sort(&(this->secGoldLabel[0]), &(this->secGoldLabel[this->secPosLabelNum]) ); // 0~resNum -1のソート
  }

  this->errorNum = VecI::Zero(4);// correct, pos_sec_pos_par, pos_sec_nega_par, nega_sec_nega_par
  this->errorRate = VecD::Zero(4);// correct, pos_sec_pos_par, pos_sec_nega_par, nega_sec_nega_par
  this->errorRegRate = VecD::Zero(3);// regularized pos_sec_pos_par, pos_sec_nega_par, nega_sec_nega_par  
}

Article::~Article(){
  for(int i = 0; i < this->bodySentNum; ++i){
      this->sent2prg[i] = NULL;
      this->sent2sec[i] = NULL;
  }
  for(int i = 0; i < this->bodyUsedSecNum; ++i){
    this->bodyUsedSec[i] = NULL;
  }
  for(int i = 0; i < this->bodySentNum; ++i){
    this->bodySentCopy[i] = NULL;
  }
  for(int i = 0; i < this->bodyPrgNum; ++i){
    this->bodyPrg[i]->disconnect();
  }
  for(int i = 0; i < this->bodySecNum; ++i){
    this->bodySec[i]->disconnect();
  }
  this->bodyRoot->left = NULL;

  for(int i = 0; i < this->bodySentNum; ++i){
    delete this->bodySent[i];
  }
  for(int i = 0; i < this->bodyPrgNum; ++i){
    delete this->bodyPrg[i];
  }
  for(int i = 0; i < this->bodySecNum; ++i){
    delete this->bodySec[i];
  }
  delete this->bodyRoot;

  for(int i = 0; i < this->abstPrgNum; ++i){
    this->abstPrg[i]->disconnect();
  }
  for(int i = 0; i < this->abstSecNum; ++i){
    this->abstSec[i]->disconnect();
  }
  this->abstRoot->left = NULL;

  for(int i = 0; i < this->abstSentNum; ++i){
    delete this->abstSent[i];
  }
  for(int i = 0; i < this->abstPrgNum; ++i){
    delete this->abstPrg[i];
  }
  for(int i = 0; i < this->abstSecNum; ++i){
    delete this->abstSec[i];
  }
  delete this->abstRoot;

  for(int i = 0; i < this->kwdNum; ++i){
    delete this->kwdGroup[i];
  }
  delete this->counter;
}

void Article::buildTree(const XmlElem* elem, Article::Section* node, const bool isChild, Article::Arg& arg){
  // elem <- read, sp <- write
  if(elem == NULL){
    return;
  }

  if(isChild){
    node->left = new Article::Section(elem, arg);
    arg.sec.push_back(node->left);
  }
  else{
    node->right = new Article::Section(elem, arg);
    arg.sec.push_back(node->right);
  }

  this->buildTree(elem->FirstChildElement("sec"), node, true, arg);// write child node
  this->buildTree(elem->NextSiblingElement("sec"), node, false, arg);// write colleage node
}

bool Article::cmp(const Article::Sentence* sent1, const Article::Sentence* sent2){
  return sent1->score > sent2->score;
}

void Article::rerank(){
  std::sort(this->bodySentCopy.begin(), this->bodySentCopy.end(), this->cmp);
  if(!this->splWordNum) {
    // sent_num
    for(int i = 0; i < this->sentPosLabelNum; ++i) {
      this->resSent[i] = this->bodySentCopy[i]->index;
    }
    this->resSentNum = this->sentPosLabelNum;
  }
  else{
    // word_num
    this->resSentNum = 0;
    int sum = 0;
    for(int i = 0; i < this->bodySentNum; ++i) {
      int id = this->bodySentCopy[i]->index;
      int tmp = this->bodySent[id]->wordNum;
      if(this->resUpperUniNum >= sum + tmp) {
        sum += tmp;
        this->resSent[this->resSentNum++] = id;
      }
      else{
        break;
      }
    }
  }
  std::sort(&(this->resSent[0]), &(this->resSent[this->resSentNum]) ); // 0~resNum -1のソート

  this->resPrgNum = this->resSecNum = 0;
  std::unordered_set<int> tmpPrg, tmpSec;
  for(int i = 0; i < this->resSentNum; ++i){
    const int index = this->resSent[i];
    tmpPrg.insert(this->sent2prg[index]->index);
    tmpSec.insert(this->sent2sec[index]->index);
  }
  for(auto itr = tmpPrg.begin(), itrEnd = tmpPrg.end(); itr != itrEnd; ++itr){
    this->resPrg[this->resPrgNum++] = (*itr);
  }
  std::sort(&(this->resPrg[0]), &(this->resPrg[this->resPrgNum]) ); // 0~resNum -1のソート

  for(auto itr = tmpSec.begin(), itrEnd = tmpSec.end(); itr != itrEnd; ++itr){
    this->resSec[this->resSecNum++] = (*itr);
  }
  std::sort(&(this->resSec[0]), &(this->resSec[this->resSecNum]) ); // 0~resNum -1のソート
}

int Article::sweep(const std::vector<int>& word){
  // return
  int res = 0;
  for(size_t i = 0, i_end = this->kwdGroup.size(); i < i_end; ++i){
    res += this->kwdGroup[i]->sweep(word);
  }
  return res;
}

void Article::appEosToken(){
  const int eosIndex = this->voc.eosIndex;
  for(size_t i = 0, i_end = this->bodySent.size(); i < i_end; ++i){
    this->bodySent[i]->appEosToken(eosIndex);
  }
}

void Article::repUnkToken(){
  const int unkIndex = this->voc.unkIndex;
  for(size_t i = 0, i_end = this->bodySent.size(); i < i_end; ++i){
    this->bodySent[i]->repUnkToken(unkIndex);
  }
}

void Article::repOrgToken(){
  for(size_t i = 0, i_end = this->bodySent.size(); i < i_end; ++i){
    this->bodySent[i]->repOrgToken();
  }
}

void Article::repUnkToken(const std::vector<int>& res, const int len){
  const int unkIndex = this->voc.unkIndex;
  for(int i = 0; i < len; ++i){
    this->bodySent[ res[i] ]->repUnkToken(unkIndex);
  }
}

void Article::repOrgToken(const std::vector<int>& res, const int len){
  for(int i = 0; i < len; ++i){
    this->bodySent[ res[i] ]->repOrgToken();
  }
}

Real Article::getSentRecall(const std::vector<int>& v, const int len){
  int num = 0;
  for(int i = 0; i < len; ++i){
    if(this->sentGoldLabelSet.count(v[i])){
      ++num;
    }
  }
  return num*1.0/this->sentPosLabelNum;
}

Real Article::getPrgRecall(const std::vector<int>& v, const int len){
  int num = 0;
  for(int i = 0; i < len; ++i){
    if(this->prgGoldLabelSet.count(v[i])){
      ++num;
    }
  }
  return num*1.0/this->prgPosLabelNum;
}
Real Article::getSecRecall(const std::vector<int>& v, const int len){
  int num = 0;
  for(int i = 0; i < len; ++i){
    if(this->secGoldLabelSet.count(v[i])){
      ++num;
    }
  }
  return num*1.0/this->secPosLabelNum;
}
void Article::getSentRegScore(){
  Real sum = 0;
  for(int i = 0; i < this->bodySentNum; ++i){
    sum += this->bodySent[i]->score;
  }
  for(int i = 0; i < this->bodySentNum; ++i){
    this->bodySent[i]->regScore = this->bodySent[i]->score/sum;
  }
}
void Article::getParRegScore(){
  Real sum = 0;
  for(int i = 0; i < this->bodyPrgNum; ++i){
    sum += this->bodyPrg[i]->score;
  }
  for(int i = 0; i < this->bodyPrgNum; ++i){
    this->bodyPrg[i]->regScore = this->bodyPrg[i]->score/sum;
  }
}
void Article::getSecRegScore(){
  Real sum = 0;
  for(int i = 0; i < this->bodyUsedSecNum; ++i){
    sum += this->bodyUsedSec[i]->score;
  }
  for(int i = 0; i < this->bodyUsedSecNum; ++i){
    this->bodyUsedSec[i]->regScore = this->bodyUsedSec[i]->score/sum;
  }
}
void Article::classifyError(const std::vector<int> array, const int array_size){
  this->errorNum.setZero();
  this->errorRate.setZero();
  this->errorRegRate.setZero();

  int TotalErrorNum = 0;
  
  for(int i = 0; i < array_size; ++i){
    const int index = array[i];
    const Article::Sentence* sent = this->bodySent[index];
    const Article::Paragraph* par = this->sent2prg[index];
    const Article::Section* sec = this->sent2sec[index];

    if(sent->label > 0.5){
      // correct
      this->errorNum.coeffRef(0,0) += 1;
    }
    else{
      if(par->label > 0.5 && sec->label > 0.5){
        // pos_sec_pos_par
        this->errorNum.coeffRef(1,0) += 1;
	TotalErrorNum += 1;
      }
      else if(par->label < 0.5 && sec->label > 0.5){
        // pos_sec_nega_par
        this->errorNum.coeffRef(2,0) += 1;
	TotalErrorNum += 1;
      }
      else if(par->label < 0.5 && sec->label < 0.5){
        // nega_sec_nega_par
        this->errorNum.coeffRef(3,0) += 1;
	TotalErrorNum += 1;
      }
    }
  }
  
  const Real tmp = 1.0/this->errorNum.sum();
  this->errorRate.coeffRef(0,0) = this->errorNum.coeffRef(0,0)*tmp;
  this->errorRate.coeffRef(1,0) = this->errorNum.coeffRef(1,0)*tmp;
  this->errorRate.coeffRef(2,0) = this->errorNum.coeffRef(2,0)*tmp;
  this->errorRate.coeffRef(3,0) = this->errorNum.coeffRef(3,0)*tmp;
  
  const Real tmp_ = 1.0/TotalErrorNum;
  this->errorRegRate.coeffRef(0,0) = this->errorNum.coeffRef(1,0)*tmp_;
  this->errorRegRate.coeffRef(1,0) = this->errorNum.coeffRef(2,0)*tmp_;
  this->errorRegRate.coeffRef(2,0) = this->errorNum.coeffRef(3,0)*tmp_;
}

Article::Kwd::Kwd(const XmlElem* elem, Vocabulary& voc):
  len(0)
{
  std::string str(elem->GetText());
  Article::getLowerCase(str);// lower casing
  std::istringstream stream(str);
  std::string token;
  int num;
  while(getline(stream, token, ' ')){
    if(voc.token2index.count(token)){
      num = voc.token2index.at(token);
    }
    else{
      num = voc.unkIndex;
    }
    this->phrase.push_back(num);
    ++this->len;
  }
  this->cmpSize = sizeof(int) * this->len;
}

int Article::Kwd::sweep(const std::vector<int>& word){
  // return the number of matching among a sentence
  int res = 0;
  for(int i = 0, i_end = (int)word.size() - this->len; i <= i_end; ++i){
    if( !memcmp( &this->phrase[0], &word[i], this->cmpSize ) ){
      ++res;
    }
  }
  return res;
}

Article::Sentence::Sentence(const XmlElem* sentElem, Article::Arg& arg):
  score(0), label(0), regScore(0)
{
  this->index = atoi(sentElem->Attribute("id"));

  this->wordNum = 0;

  std::string line(sentElem->GetText());
  Article::getLowerCase(line);// lower casing
  std::istringstream stream(line);
  std::string token;

  int num;

  while(getline(stream, token, ' ')) {
    if(arg.voc.token2index.count(token)){
      num = arg.voc.token2index.at(token);
    }
    else if(arg.voc.unk2index.count(token)){
      num = arg.voc.unk2index.at(token);
      this->unkList.push_back(std::make_pair(this->wordNum, num));
    }
    else{
      num = arg.voc.unkCounter++;
      arg.voc.unk2index.insert(std::make_pair(token, num));
      this->unkList.push_back(std::make_pair(this->wordNum, num));
    }
    ++this->wordNum;
    this->word.push_back(num);
  }
  this->biNum = this->wordNum - 1;
}

void Article::Sentence::repUnkToken(const int unkIndex){
  // replace orginal token as *UNK* in this sentence
  for(size_t i = 0, i_end = this->unkList.size(); i < i_end; ++i){
    this->word[this->unkList[i].first] = unkIndex;
  }
}

void Article::Sentence::repOrgToken(){
  // replace *UNK* as orginal token in this sentence
  for(size_t i = 0, i_end = this->unkList.size(); i < i_end; ++i){
    this->word[this->unkList[i].first] = this->unkList[i].second;
  }
}

void Article::Sentence::appEosToken(const int eosIndex){
  // append *EOS* into this sentence
  this->word.push_back(eosIndex);
  ++this->wordNum;
}

Article::Paragraph::Paragraph(const XmlElem* prgElem, Article::Arg& arg):
score(0), label(0), regScore(0)
{
  this->index = atoi(prgElem->Attribute("id"));

  // this->salvage(prgElem->FirstChildElement("s"), arg);
  this->traverse(prgElem->FirstChildElement(), arg);

  this->sentNum = (int)this->pSent.size();
}

void Article::Paragraph::disconnect(){
  for(size_t i = 0, i_end = this->pSent.size(); i < i_end; ++i){
    this->pSent[i] = NULL;
  }
}

void Article::Paragraph::traverse(const XmlElem* elem, Article::Arg& arg){

  if(elem == NULL){
    return;
  }

  const char* str = elem->Name();
  if( !strcmp("s", str) ){
    arg.sent.push_back( new Article::Sentence(elem, arg) );
    this->pSent.push_back( arg.sent.back() );
    this->sent.push_back( arg.sent.back()->index );
  }

  this->traverse(elem->FirstChildElement(), arg);
  this->traverse(elem->NextSiblingElement(), arg);
}

void Article::Paragraph::salvage(const XmlElem* sentElem, Article::Arg& arg){
  if(sentElem == NULL){
    return;
  }

  arg.sent.push_back( new Article::Sentence(sentElem, arg) );
  this->pSent.push_back( arg.sent.back() );
  this->sent.push_back( arg.sent.back()->index );

  this->salvage(sentElem->NextSiblingElement("s"), arg);
}

Article::Section::Section(const XmlElem* secElem, Article::Arg& arg):
score(0), label(0), left(NULL), right(NULL), regScore(0)
{
  this->index = atoi(secElem->Attribute("id"));

  std::string tmp(secElem->Attribute("sec-type"));
  if(arg.voc.name2index.count(tmp)){
    this->name = arg.voc.name2index.at(tmp);
  }
  else{
    this->name = arg.voc.unkNameIndex;
  }

  this->salvage(secElem->FirstChildElement("p"), arg);

  for(size_t i = 0, i_end = this->pPrg.size(); i < i_end; ++i){
    const Article::Paragraph* pp = this->pPrg[i];
    for(size_t j = 0, j_end = pp->pSent.size(); j < j_end; ++j){
      this->pSent.push_back(pp->pSent[j]);
      this->sent.push_back(pp->sent[j]);
    }
  }

  this->sentNum = (int)this->pSent.size();
  this->prgNum = (int)this->pPrg.size();
}

inline void Article::Section::disconnect(){
  for(size_t i = 0, i_end = this->pSent.size(); i < i_end; ++i){
    this->pSent[i] = NULL;
  }
  for(size_t i = 0, i_end = this->pPrg.size(); i < i_end; ++i){
    this->pPrg[i] = NULL;
  }
  this->left = NULL;
  this->right = NULL;
}

void Article::Section::salvage(const XmlElem* prgElem, Article::Arg& arg){
  if(prgElem == NULL) {
    return;
  }
  arg.prg.push_back(new Article::Paragraph(prgElem, arg));

  this->pPrg.push_back(arg.prg.back());
  this->prg.push_back(arg.prg.back()->index);

  this->salvage(prgElem->NextSiblingElement("p"), arg);
}

void Article::LEAD(const std::string& testPathList, unsigned int threadNum){

  Vocabulary voc(testPathList, 0, 0);

  std::vector<Article*> docs;
  Article::set(testPathList, docs, voc);

  std::cout << "testing Data size = " << docs.size() << std::endl;

  ROUGE eval;
  std::vector<ROUGE> evalTmp;
  for (unsigned int i = 0; i < threadNum; ++i) {
    evalTmp.push_back(ROUGE());
    evalTmp.back().init();
  }

  std::cout << "testing Data : OPTIMAL" << std::endl;
  eval.init();
#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(eval, evalTmp)
  for (size_t i = 0; i < docs.size(); ++i) {
    unsigned int id = omp_get_thread_num();
    docs[i]->evaluate(docs[i]->sentGoldLabel, docs[i]->sentPosLabelNum, evalTmp[id]);
  }
  for(unsigned int id = 0; id < threadNum; ++id) {
    eval += evalTmp[id];
    evalTmp[id].init();
  }
  eval *= 1.0 / docs.size();
  eval.print();

  std::cout << "testing Data : LEAD" << std::endl;
  eval.init();
  std::vector<Real> recall(threadNum, 0);
#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(eval, evalTmp)
  for (size_t i = 0; i < docs.size(); ++i) {
    unsigned int id = omp_get_thread_num();

    docs[i]->resSentNum = 0;
    int sum = 0;
    for(int j = 0; j < docs[i]->bodySentNum; ++j) {
      int tmp = docs[i]->bodySent[j]->wordNum;
      if(docs[i]->resUpperUniNum >= sum + tmp) {
        sum += tmp;
        docs[i]->resSent[docs[i]->resSentNum++] = j;
      }
      else{
        break;
      }
    }

    std::sort(&(docs[i]->resSent[0]), &(docs[i]->resSent[docs[i]->resSentNum])); // 0~resNum -1のソート
    docs[i]->evaluate(docs[i]->resSent, docs[i]->resSentNum, evalTmp[id]);
    recall[id] += docs[i]->getSentRecall(docs[i]->resSent, docs[i]->resSentNum);
  }

  Real totalRecall = 0;
  for(unsigned int id = 0; id < threadNum; ++id) {
    eval += evalTmp[id];
    evalTmp[id].init();
    totalRecall += recall[id];
  }
  eval *= 1.0/docs.size();
  eval.print();

  std::cout << "Recall = " << totalRecall / (1.0*docs.size()) << std::endl;

  Article::clear(docs);
}

void Article::optimal(const std::string& testPathList, const unsigned int threadNum){
  Vocabulary voc(testPathList, 0, 0);

  std::vector<Article*> docs;
  Article::set(testPathList, docs, voc);

  std::cout << "testing Data size = " << docs.size() << std::endl;

  ROUGE eval;
  std::vector<ROUGE> evalTmp;
  for (unsigned int i = 0; i < threadNum; ++i) {
    evalTmp.push_back(ROUGE());
    evalTmp.back().init();
  }

  std::cout << "testing Data : OPTIMAL" << std::endl;
  eval.init();
#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(eval, evalTmp)
  for (size_t i = 0; i < docs.size(); ++i) {
    unsigned int id = omp_get_thread_num();
    docs[i]->evaluate(docs[i]->sentGoldLabel, docs[i]->sentPosLabelNum, evalTmp[id]);
  }
  for(unsigned int id = 0; id < threadNum; ++id) {
    eval += evalTmp[id];
    evalTmp[id].init();
  }
  eval *= 1.0 / docs.size();
  eval.print();
}

void Article::set(const std::string& pathList, std::vector<Article*>& docs, Vocabulary& voc){
  std::ifstream fin(pathList);
  if(!fin){
    std::cout << pathList << " cannot open" << std::endl;
    return;
  }

  std::string line;
  while(getline(fin, line, '\n')){
    //    std::cout << line << std::endl;
    docs.push_back(new Article(line, voc));
  }
}

void Article::clear(std::vector<Article*>& docs){
  for(size_t i = 0, i_end = docs.size(); i < i_end; ++i){
    delete docs[i]; // destructor of Document is called here
    docs[i] = NULL;
  }
}

void Article::appEosToken(std::vector<Article*>& docs){
  for(size_t i = 0, i_end = docs.size(); i < i_end; ++i){
    docs[i]->appEosToken();
  }
}

void Article::repUnkToken(std::vector<Article*>& docs){
  for(size_t i = 0, i_end = docs.size(); i < i_end; ++i){
    docs[i]->repUnkToken();
  }
}

void Article::repOrgToken(std::vector<Article*>& docs){
  for(size_t i = 0, i_end = docs.size(); i < i_end; ++i){
    docs[i]->repOrgToken();
  }
}

void Article::print(std::ofstream& fout){
  // for checking
  fout << "\n\nTitle : ";
  for(size_t k = 0, k_end = this->titleArray.size(); k < k_end; ++k){
    fout << this->voc.index2token.at(this->titleArray[k]) << " ";
  }
  fout << std::endl;
  for(int i = 0; i < this->bodySecNum; ++i){
    const Article::Section* sec = this->bodySec[i];
    fout << "Section No." << sec->index << std::endl;
    for(int j = 0, j_end = sec->sentNum; j < j_end; ++j){
      const Article::Sentence* sent = sec->pSent[j];
      fout << "\tSentence No." << sent->index << std::endl;
      fout << "\t";
      for(int k = 0, k_end = sent->wordNum; k < k_end; ++k){
        fout << this->voc.index2token.at(sent->word[k]) << " ";
      }
      fout << std::endl;
    }
  }
}

void Article::delGoldLabelNode(const std::string& pathList, const char* target){
  std::ifstream fin(pathList);
  if(!fin){
    std::cout << pathList << " cannot open" << std::endl;
    return;
  }

  std::string line;
  std::vector<std::string> lines;
  while(getline(fin, line, '\n')){
    lines.push_back(line);
  }

  for(size_t i = 0, i_end = lines.size(); i < i_end; ++i){
    const std::string _filePath = lines[i];

    XmlDoc xml;
    xml.LoadFile(_filePath.c_str());

    XmlElem* article = xml.FirstChildElement("article");// root
    XmlElem* info = article->FirstChildElement("info");
    XmlElem* elem = info->FirstChildElement(target);
    info->DeleteChild(elem);

    if( remove(_filePath.c_str()) ){
      std::cout << _filePath << " cannot removed" << std::endl;
      exit(1);
    }

    xml.SaveFile(_filePath.c_str());
  }
}

void Article::delInfoNode(const std::string& pathList){
  std::ifstream fin(pathList);
  if(!fin){
    std::cout << pathList << " cannot open" << std::endl;
    return;
  }

  std::string line;
  std::vector<std::string> lines;
  while(getline(fin, line, '\n')){
    lines.push_back(line);
  }

  for(size_t i = 0, i_end = lines.size(); i < i_end; ++i){
    const std::string _filePath = lines[i];

    XmlDoc xml;
    xml.LoadFile(_filePath.c_str());

    XmlElem* article = xml.FirstChildElement("article");// root
    XmlElem* info = article->FirstChildElement("info");
    article->DeleteChild(info);

    if( remove(_filePath.c_str()) ){
      std::cout << _filePath << " cannot removed" << std::endl;
      exit(1);
    }

    xml.SaveFile(_filePath.c_str());
  }
}

void Article::appInfoNode(const std::string& pathList){
  std::ifstream fin(pathList);
  if(!fin){
    std::cout << pathList << " cannot open" << std::endl;
    return;
  }

  std::string line;
  std::vector<std::string> lines;
  while(getline(fin, line, '\n')){
    lines.push_back(line);
  }

  for(size_t i = 0, i_end = lines.size(); i < i_end; ++i){
    const std::string _filePath = lines[i];

    XmlDoc xml;
    xml.LoadFile(_filePath.c_str());

    XmlElem* article = xml.FirstChildElement("article");// root
    XmlElem* info = xml.NewElement("info");
    article->InsertEndChild(info);

    if( remove(_filePath.c_str()) ){
      std::cout << _filePath << " cannot removed" << std::endl;
      exit(1);
    }

    xml.SaveFile(_filePath.c_str());
  }
}

void Article::getAveRate(const std::string& pathList){
  Vocabulary voc(pathList, 0, 0);// path, tokenFreqThreshold, nameFreqThreshold

  std::vector<Article*> docs;
  Article::set(pathList, docs, voc);

  VecD rate = VecD(docs.size());
  for(size_t i = 0, i_end = docs.size(); i < i_end; ++i){
    if(docs[i]->totalBodyUniNum < 1 || docs[i]->counter->totalAbstUniNum < 1){
      rate.coeffRef(i,0) = 0;
      std::cout << docs[i]->filePath << std::endl;
      std::cout << "totalBodyUniNum = " << docs[i]->totalBodyUniNum << std::endl;
      std::cout << "totalAbstUniNum = " << docs[i]->counter->totalAbstUniNum << std::endl;
    }
    else{
      rate.coeffRef(i,0) = docs[i]->counter->totalAbstUniNum*1.0 / docs[i]->totalBodyUniNum;
    }
  }
  const Real mean = rate.mean();
  std::cout << "mean = " << mean << std::endl;
  /*
    rate -= mean;
    const Real dev = std::sqrt( (rate.array()*rate.array()).mean() );
    std::cout << "dev = " << dev << std::endl;
  */
}

void Article::all2txt(const std::string& pathList, const std::string& output){
  std::ofstream fout(output);
  if(!fout){
    std::cout << output << " cannot open" << std::endl;
    return;
  }

  Vocabulary voc(pathList, 0, 0);// path, tokenFreqThreshold, nameFreqThreshold
  voc.setInverter();

  std::vector<Article*> docs;
  Article::set(pathList, docs, voc);

  for(size_t i = 0, i_end = docs.size(); i < i_end; ++i){
    const Article* doc = docs[i];
    for(int j = 0, j_end = doc->bodySentNum; j < j_end; ++j){
      const Article::Sentence* sent = doc->bodySent[j];
      for(int k = 0, k_end = sent->wordNum; k < k_end; ++k){
        fout << voc.index2token.at(sent->word[k]) << " ";
      }
      fout << std::endl;
    }
  }
  fout.close();
}

Article::Counter::Counter(const Article& article):
  totalAbstUniNum(0), totalAbstBiNum(0)
{
  std::unordered_map<int, int> uniCounter;// for evaluation
  std::unordered_map<std::pair<int, int>, int> biCounter;// for evaluation

  for(int i = 0, i_end = article.abstSentNum; i < i_end; ++i){
    const Article::Sentence* sp = article.abstSent[i];
    std::pair<int,int> bigram(-1,-1);
    for(int j = 0, j_end = sp->wordNum; j < j_end; ++j){
      const int num =  sp->word[j];
      if( !uniCounter.count(num) ){
        uniCounter.insert( std::make_pair(num, 1) );
      }
      else{
        ++uniCounter.at(num);
      }

      bigram.first = bigram.second;
      bigram.second = num;

      if(bigram.first != -1) {
        if( !biCounter.count(bigram) ){
          biCounter.insert( std::make_pair(bigram, 1) );
        }
        else{
          ++biCounter.at(bigram);
        }
      }

      this->abstWordSeq.push_back(num);
    }
    this->totalAbstUniNum += sp->wordNum;
    this->totalAbstBiNum += sp->biNum;
  }

  this->uniCount = MatI::Zero(uniCounter.size(), 2);
  int counter = 0;
  for(auto itr = uniCounter.begin(), itr_end = uniCounter.end(); itr != itr_end; ++itr){
    this->uniConverter.insert(std::make_pair(itr->first, counter));
    this->uniCount.coeffRef(counter, 0) = itr->second;// set abst unigram
    ++counter;
  }

  this->biCount = MatI::Zero(biCounter.size(), 2);
  counter = 0;
  for(auto itr = biCounter.begin(), itr_end = biCounter.end(); itr != itr_end; ++itr){
    this->biConverter.insert(std::make_pair(itr->first, counter));
    this->biCount.coeffRef(counter, 0) = itr->second;// set abst bigram
    ++counter;
  }
}

void Article::evaluate(const std::vector<int>& res, const int len, ROUGE& eval){

  MatI bodyUniCount = MatI::Zero(this->counter->uniCount.rows(), len);
  MatI bodyBiCount = MatI::Zero(this->counter->biCount.rows(), len);
  this->resUniNum = 0;
  this->resBiNum = 0;
  for(int i = 0; i < len; ++i){
    const int index = res[i];
    const Article::Sentence* sp = this->bodySent[index];
    std::pair<int,int> bigram(-1,-1);
    for(int j = 0, j_end = sp->wordNum; j < j_end; ++j){
      const int num =  sp->word[j];

      if(this->counter->uniConverter.count(num)){
        ++bodyUniCount.coeffRef(this->counter->uniConverter.at(num), i);
      }

      bigram.first = bigram.second;
      bigram.second = num;

      if(bigram.first != -1) {
        if(this->counter->biConverter.count(bigram)){
          ++bodyBiCount.coeffRef(this->counter->biConverter.at(bigram), i);
        }
      }
    }
    this->resUniNum += sp->wordNum;
    this->resBiNum += sp->biNum;
  }

  this->counter->uniCount.col(1) = bodyUniCount.rowwise().sum();
  const int uniProdNum = this->counter->uniCount.rowwise().minCoeff().sum();

  //if(resUniNum == 0) std::cout << "resUniNum == 0 : " << doc.filePath << std::endl;
  eval.rouge1.coeffRef(0,0) += uniProdNum * 1.0 / this->counter->totalAbstUniNum;
  eval.rouge1.coeffRef(1,0) += uniProdNum * 1.0 / this->resUniNum;
  eval.rouge1.coeffRef(2,0) += uniProdNum * 2.0 / (this->counter->totalAbstUniNum + this->resUniNum);

  this->counter->biCount.col(1) = bodyBiCount.rowwise().sum();
  const int biProdNum = this->counter->biCount.rowwise().minCoeff().sum();

  //if(resBiNum == 0) std::cout << "resBiNum == 0 : " << doc.filePath << std::endl;
  eval.rouge2.coeffRef(0,0) += biProdNum * 1.0 / this->counter->totalAbstBiNum;
  eval.rouge2.coeffRef(1,0) += biProdNum * 1.0 / this->resBiNum;
  eval.rouge2.coeffRef(2,0) += biProdNum * 2.0 / (this->counter->totalAbstBiNum + this->resBiNum);

  Real llcs = 0.0;

    // summry-level LCS
    for(int i = 0; i < this->abstSentNum; ++i){
      std::unordered_set<int> lcsUnion;
      size_t combinedLcsLength = 0;
      for(int j = 0; j < len; ++j){
        const int index = res[j];
        std::vector<int> array;
        this->lcs(this->abstSent[i]->word, this->bodySent[index]->word, array);
        combinedLcsLength += array.size();
        for(size_t k = 0, k_end = array.size(); k < k_end; ++k){
          lcsUnion.insert(array[k]);
        }
      }
      // llcs += lcsUnion.size()*1.0/combinedLcsLength;
      llcs += lcsUnion.size()*1.0;
    }

    const Real r_lcs = llcs / this->counter->totalAbstUniNum;
    const Real p_lcs = llcs / this->resUniNum;
    const Real beta = p_lcs / (r_lcs + 1.0e-12);
    const Real num = (1 + (beta*beta)) * r_lcs * p_lcs;
    const Real denom = r_lcs + ((beta*beta) * p_lcs);
    const Real f_lcs = num / (denom + 1.0e-12);

    eval.rougeL.coeffRef(0,0) += r_lcs;
    eval.rougeL.coeffRef(1,0) += p_lcs;
    eval.rougeL.coeffRef(2,0) += f_lcs;
}

ROUGE Article::evaluate(const std::vector<int>& res, const int len){

  ROUGE eval;

  MatI bodyUniCount = MatI::Zero(this->counter->uniCount.rows(), len);
  MatI bodyBiCount = MatI::Zero(this->counter->biCount.rows(), len);
  this->resUniNum = 0;
  this->resBiNum = 0;
  for(int i = 0; i < len; ++i){
    const int index = res[i];
    const Article::Sentence* sp = this->bodySent[index];
    std::pair<int,int> bigram(-1,-1);
    for(int j = 0, j_end = sp->wordNum; j < j_end; ++j){
      const int num =  sp->word[j];

      if(this->counter->uniConverter.count(num)){
        ++bodyUniCount.coeffRef(this->counter->uniConverter.at(num), i);
      }

      bigram.first = bigram.second;
      bigram.second = num;

      if(bigram.first != -1) {
        if(this->counter->biConverter.count(bigram)){
          ++bodyBiCount.coeffRef(this->counter->biConverter.at(bigram), i);
        }
      }
    }
    this->resUniNum += sp->wordNum;
    this->resBiNum += sp->biNum;
  }

  this->counter->uniCount.col(1) = bodyUniCount.rowwise().sum();
  const int uniProdNum = this->counter->uniCount.rowwise().minCoeff().sum();

  //if(resUniNum == 0) std::cout << "resUniNum == 0 : " << doc.filePath << std::endl;
  eval.rouge1.coeffRef(0,0) = uniProdNum * 1.0 / this->counter->totalAbstUniNum;
  eval.rouge1.coeffRef(1,0) = uniProdNum * 1.0 / this->resUniNum;
  eval.rouge1.coeffRef(2,0) = uniProdNum * 2.0 / (this->counter->totalAbstUniNum + this->resUniNum);

  this->counter->biCount.col(1) = bodyBiCount.rowwise().sum();
  const int biProdNum = this->counter->biCount.rowwise().minCoeff().sum();

  //if(resBiNum == 0) std::cout << "resBiNum == 0 : " << doc.filePath << std::endl;
  eval.rouge2.coeffRef(0,0) = biProdNum * 1.0 / this->counter->totalAbstBiNum;
  eval.rouge2.coeffRef(1,0) = biProdNum * 1.0 / this->resBiNum;
  eval.rouge2.coeffRef(2,0) = biProdNum * 2.0 / (this->counter->totalAbstBiNum + this->resBiNum);

  Real llcs = 0.0;

    // summry-level LCS
    for(int i = 0; i < this->abstSentNum; ++i){
      std::unordered_set<int> lcsUnion;
      size_t combinedLcsLength = 0;
      for(int j = 0; j < len; ++j){
        const int index = res[j];
        std::vector<int> array;
        this->lcs(this->abstSent[i]->word, this->bodySent[index]->word, array);
        combinedLcsLength += array.size();
        for(size_t k = 0, k_end = array.size(); k < k_end; ++k){
          lcsUnion.insert(array[k]);
        }
      }
      // llcs += lcsUnion.size()*1.0/combinedLcsLength;
      llcs += lcsUnion.size()*1.0;
    }

    const Real r_lcs = llcs / this->counter->totalAbstUniNum;
    const Real p_lcs = llcs / this->resUniNum;
    const Real beta = p_lcs / (r_lcs + 1.0e-12);
    const Real num = (1 + (beta*beta)) * r_lcs * p_lcs;
    const Real denom = r_lcs + ((beta*beta) * p_lcs);
    const Real f_lcs = num / (denom + 1.0e-12);

    eval.rougeL.coeffRef(0,0) = r_lcs;
    eval.rougeL.coeffRef(1,0) = p_lcs;
    eval.rougeL.coeffRef(2,0) = f_lcs;

    return eval;
}

void Article::lcs(const std::vector<int>& X, const std::vector<int>& Y, std::vector<int>& res){
  const size_t m = X.size();
  const size_t n = Y.size();
  MatI L = MatI::Zero(m+1,n+1);

  /* Following steps build L[m+1][n+1] in bottom up fashion. Note
     that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] */
  for (size_t i = 0; i <= m; ++i){
    for (size_t j = 0; j <= n; ++j){
      if (i == 0 || j == 0){
        L.coeffRef(i,j) = 0;
      }
      else if (X[i-1] == Y[j-1]){
        L.coeffRef(i,j) = L.coeffRef(i-1,j-1) + 1;
      }
      else{
        L.coeffRef(i,j) = std::max(L.coeffRef(i-1,j), L.coeffRef(i,j-1));
      }
    }
  }

  // Following code is used to print LCS
  int index = L.coeffRef(m,n);

  // Create a character array to store the lcs string
  //char lcs[index+1];
  //lcs[index] = '\0'; // Set the terminating character
  res.resize(index);

  // Start from the right-most-bottom-most corner and
  // one by one store characters in lcs[]
  int i = m, j = n;
  while (i > 0 && j > 0){
    // If current character in X[] and Y are same, then
    // current character is part of LCS
    if (X[i-1] == Y[j-1]){
      res[index-1] = X[i-1]; // Put current character in result
      --i;
      --j;
      --index;     // reduce values of i, j and index
    }

    // If not same, then find the larger of two and
    // go in the direction of larger value
    else if (L.coeffRef(i-1,j) > L.coeffRef(i,j-1)){
      --i;
    }
    else{
      --j;
    }
  }
  std::sort(res.begin(), res.end());
  // Print the lcs
  /*
    std::cout << std::endl;
    std::cout << "LCS is " << std::endl;
    for(size_t i = 0, i_end = res.size(); i < i_end; ++i){
    std::cout << res[i] << " ";
    }
    std::cout << std::endl;
  */
}

Real Article::getRougeL(const std::vector<int>& res, const int len, ROUGE& eval){
  Real llcs = 0.0;

  // summry-level LCS
  for(int i = 0; i < this->abstSentNum; ++i){
    std::unordered_set<int> lcsUnion;
    size_t combinedLcsLength = 0;
    for(int j = 0; j < len; ++j){
      const int index = res[j];
      std::vector<int> array;
      this->lcs(this->abstSent[i]->word, this->bodySent[index]->word, array);
      combinedLcsLength += array.size();
      for(size_t k = 0, k_end = array.size(); k < k_end; ++k){
        lcsUnion.insert(array[k]);
      }
    }
    // llcs += lcsUnion.size()*1.0/combinedLcsLength;
    llcs += lcsUnion.size()*1.0;
  }

  /*
  // sentece-level LCS
  std::vector<int> resWordSeq;
  for(int i = 0; i < len; ++i){
  const int index = res[i];
  Article::Sentence* sp = this->bodySent[index];
  sp->repOrgToken();
  for(int j = 0, j_end = sp->wordNum; j < j_end; ++j){
  resWordSeq.push_back(sp->word[j]);
  }
  sp->repUnkToken(this->voc.unkIndex);
  }
  std::vector<int> array;
  this->lcs(this->counter->abstWordSeq, resWordSeq, array);
  llcs = (Real)array.size();
  */

  const Real r_lcs = llcs / this->counter->totalAbstUniNum;
  const Real p_lcs = llcs / this->resUniNum;
  const Real beta = p_lcs / (r_lcs + 1.0e-12);
  const Real num = (1 + (beta*beta)) * r_lcs * p_lcs;
  const Real denom = r_lcs + ((beta*beta) * p_lcs);
  const Real f_lcs = num / (denom + 1.0e-12);

  eval.rougeL.coeffRef(0,0) += r_lcs;
  eval.rougeL.coeffRef(1,0) += p_lcs;
  eval.rougeL.coeffRef(2,0) += f_lcs;
}
