#pragma once

#include "Article.hpp"

class Writer{
  // save text in xml file as ROUGE1.5.5 format
public:
  class Config;
  const std::string root;
  Vocabulary& voc;
  Writer(const std::string& root_, Vocabulary& voc_);
  void save(const Article* doc);
  void save(const Article* doc, const std::vector<int>& res, const int len);
  void write(const std::string& fileName, const std::vector<std::string>& doc);
  std::string compose(const Article::Sentence* sent);
  static void save();
  static void saveGoldStandardSet();
  static void searchDir(const std::string root, std::vector<std::string>& fileNames);
  static void formatFile(const std::string path);
  static void formatDir();
  static void saveProperty();
  static void saveLeadSimple();
  static void saveLeadSection();
};

class Writer::Config{
public:
  std::string fileNamePath;
  unsigned int threadNum;
  std::string outputPath;
};
