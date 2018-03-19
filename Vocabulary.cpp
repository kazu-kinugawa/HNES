#include "Vocabulary.hpp"

Vocabulary::Vocabulary(const std::string& pathList, const int tokenFreqThreshold, const int nameFreqThreshold){

  std::cout << "Making Dictionary ... " << std::flush;

  std::ifstream ifs(pathList.c_str());
  if(!ifs){
    std::cout << pathList << " cannot open" << std::endl;
    exit(1);
  }
  ifs.unsetf(std::ios::skipws);

  std::unordered_map<std::string, int> tokenCount;
  Vocabulary::tokenCounter func1(tokenCount);

  std::unordered_map<std::string, int> nameCount;
  Vocabulary::nameCounter func2(nameCount);

  std::string path;

  while(getline(ifs, path, '\n')){
    XmlDoc xml;
    XmlError loadCheck = xml.LoadFile(path.c_str());
    if(loadCheck != XmlSuccess){
      std::cout << path << " cannot open" << std::endl;
      continue;
    }
    this->traverse(xml.FirstChildElement("article"), func1, func2);
  }

  ifs.close();

  /* token */
  int counter = 0;
  for (auto it = tokenCount.begin(), itEnd = tokenCount.end(); it != itEnd; ++it){
    if (it->second >= tokenFreqThreshold) {
      this->token2index.insert(std::make_pair(it->first, counter++));
    }
  }
  //語彙サイズVとしたら,"eos"のIDはV,"unk"のIDはV+1
  this->eosIndex = this->token2index.size();
  this->unkIndex = this->eosIndex + 1;

  this->token2index.insert(std::make_pair("*EOS*", this->eosIndex));
  this->token2index.insert(std::make_pair("*UNK*", this->unkIndex));

  this->unkCounter = this->unkIndex + 1;

  /* section title */
  counter = 0;
  for (auto it = nameCount.begin(), itEnd = nameCount.end(); it != itEnd; ++it){
    if (it->second >= nameFreqThreshold){
      this->name2index.insert(std::make_pair(it->first, counter++));
    }
  }
  //語彙サイズVとしたら,"eos"のIDはV,"unk"のIDはV+1
  this->unkNameIndex = this->name2index.size();
  this->name2index.insert(std::make_pair("*UNK*", this->unkNameIndex));

  std::cout << "End" << std::endl;
  std::cout << "Vocabulary size (including *EOS* and *UNK*) = " << this->token2index.size() << std::endl;
  std::cout << "# of Section (including *UNK*) = " << this->name2index.size() << std::endl;
};

void Vocabulary::loadStopWords(const std::string& path){

  std::cout << "Loading stopWords ..." << std::endl;

  std::ifstream fin(path.c_str());
  if(!fin) {
    std::cout << path << std::endl;
    return;
  }
  fin.unsetf(std::ios::skipws);

  std::string token;
  while(getline(fin, token, '\n')){
    this->stopWords.insert(token);
  }

  fin.close();

  std::cout << "stopWords num = " << this->stopWords.size() << std::endl;
};

void Vocabulary::setInverter(){
  for(auto itr = this->token2index.begin(), itrEnd = this->token2index.end(); itr != itrEnd; ++itr){
    this->index2token.insert(std::make_pair(itr->second, itr->first));
  }
  for(auto itr = this->unk2index.begin(), itrEnd = this->unk2index.end(); itr != itrEnd; ++itr){
    this->index2unk.insert(std::make_pair(itr->second, itr->first));
  }
  for(auto itr = this->name2index.begin(), itrEnd = this->name2index.end(); itr != itrEnd; ++itr){
    this->index2name.insert(std::make_pair(itr->second, itr->first));
  }
};

template<class Func1, class Func2>
void Vocabulary::traverse(const XmlElem* elem, Func1 func1, Func2 func2){

  if(elem == NULL){
    return;
  }

  const char* str = elem->Name();
  if( !strcmp("s", str) ){
    func1(elem);
  }
  else if( !strcmp("sec", str) ){
    func2(elem);
  }

  this->traverse(elem->FirstChildElement(), func1, func2);
  this->traverse(elem->NextSiblingElement(), func1, func2);
};
