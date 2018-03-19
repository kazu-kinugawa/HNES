#pragma once

#include <string>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "XmlParser.hpp"

class Vocabulary{
public:

  class tokenCounter;
  class nameCounter;

  std::unordered_map<std::string, int> token2index;
  std::unordered_map<std::string, int> unk2index;
  // std::unordered_map<std::string, int> outOfVoc2index;
  std::unordered_map<std::string, int> name2index;

  std::unordered_map<int, std::string> index2token;
  std::unordered_map<int, std::string> index2unk;
  std::unordered_map<int, std::string> index2name;
  // std::unordered_map<int, std::string> index2outOfVoc;

  std::unordered_set<std::string> stopWords;

  int eosIndex;
  int unkIndex;
  // int outOfVocCounter;
  int unkCounter;

  int unkNameIndex;

  Vocabulary(const std::string& pathList, const int tokenFreqThreshold, const int nameFreqThreshold);
  void loadStopWords(const std::string& path);
  void setInverter();

  template<class Func1, class Func2> void traverse(const XmlElem* elem, Func1 func1, Func2 func2);
  static void getLowerCase(std::string& str){
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  }
  static std::string getLowerCase(const char* str){
    std::string res(str);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
  }
};

class Vocabulary::tokenCounter {
public:
  std::unordered_map<std::string, int>& tokenCount;
  tokenCounter(std::unordered_map<std::string, int>& _tokenCount):tokenCount(_tokenCount){}
  void operator()(const XmlElem* elem) {
      std::string tmp(elem->GetText());
      Vocabulary::getLowerCase(tmp);
      std::istringstream stream(tmp);
      std::string token;
      while(getline(stream, token, ' ')){
        if( !this->tokenCount.count(token) ){
          this->tokenCount.insert(std::make_pair(token, 1));
        }
        else{
          ++this->tokenCount.at(token);
        }
      }
    }
};

class Vocabulary::nameCounter {
public:
  std::unordered_map<std::string, int>& nameCount;
  nameCounter(std::unordered_map<std::string, int>& _nameCount):nameCount(_nameCount){}
  void operator()(const XmlElem* elem) {
      std::string str(elem->Attribute("sec-type"));
      Vocabulary::getLowerCase(str);
      if( !this->nameCount.count(str) ){
        this->nameCount.insert(std::make_pair(str, 1));
      }
      else {
      ++this->nameCount.at(str);
      }
    }
};
