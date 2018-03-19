#include "Writer.hpp"
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <dirent.h>
#include <string.h>

Writer::Writer(const std::string& root_, Vocabulary& voc_): root(root_), voc(voc_)
{
  voc_.setInverter();

  struct stat st;

  if(stat(root_.c_str(), &st) != 0){
    // root directory does not exist
    if(!mkdir(root_.c_str(), 0775)){// for linux
    // if(!mkdir(root_.c_str())){// for windows
      std::cout << root_ << " has been made" << std::endl;
    }
    else{
      std::cout << root_ << " can't be made" << std::endl;
      exit(0);
    }
  }
}

void Writer::save(const Article* doc){

  // write abstract as xml file

  std::vector<std::string> text;

  for(int i = 0, i_end = doc->abstSentNum; i < i_end; ++i){
    text.push_back(this->compose(doc->abstSent[i]));
  }

  this->write(doc->fileName, text);
}

void Writer::save(const Article* doc, const std::vector<int>& res, const int len){

  std::vector<std::string> text;

  for(int i = 0, i_end = len; i < i_end; ++i){
    const int index = res[i];
    text.push_back(this->compose(doc->bodySent[index]));
  }

  this->write(doc->fileName, text);
}

void Writer::write(const std::string& fileName, const std::vector<std::string>& doc){

  XmlDoc xml;

  XmlElem* html = xml.NewElement("html");
  xml.InsertEndChild(html);

  XmlElem* head = html->GetDocument()->NewElement("head");
  html->InsertEndChild(head);

  XmlElem* title = head->GetDocument()->NewElement("title");
  head->InsertEndChild(title);
  title->InsertEndChild( title->GetDocument()->NewText( fileName.c_str() ) );

  XmlElem* body = html->GetDocument()->NewElement("body");
  body->SetAttribute("bgcolor", "white");
  html->InsertEndChild(body);

  for(int i = 0, i_end = doc.size(); i < i_end; ++i){

    const int id = 1 + i;
    const std::string str(std::to_string(id));

    XmlElem* a1 = body->GetDocument()->NewElement("a");
    a1->SetAttribute("name", str.c_str());
    a1->InsertEndChild( a1->GetDocument()->NewText( ("["+str+"]").c_str() ) );
    body->InsertEndChild(a1);

    XmlElem* a2 = body->GetDocument()->NewElement("a");
    a2->SetAttribute("href", ("#"+str).c_str() );
    // a2->SetAttribute("id", id);
    const std::string dummy = "*DUMMY-" + str + "-DUMMY*";
    a2->SetAttribute("id", dummy.c_str());
    a2->InsertEndChild( a2->GetDocument()->NewText( doc[i].c_str() ) );
    body->InsertEndChild(a2);
  }

  const std::string path = this->root + "/" + fileName + ".html";

  const XmlError loadCheck = xml.SaveFile(path.c_str());
  if(loadCheck != XmlSuccess){
    std::cout << path << " cannot open" << std::endl;
    assert(loadCheck == XmlSuccess);
  }

  Writer::formatFile(path);
}

std::string Writer::compose(const Article::Sentence* sent){

  std::string line;

  for(int i = 0, i_end = sent->wordNum - 1; i <= i_end; ++i){

    const int token = sent->word[i];

    if(i == i_end){
      if(this->voc.index2token.count(token)){
	line += this->voc.index2token[ token ];
      }
      else{
	line += this->voc.index2unk[ token ];
      }
    }
    else{
      if(this->voc.index2token.count(token)){
	line += this->voc.index2token[ token ] + " ";
      }
      else{
	line += this->voc.index2unk[ token ] + " ";
      }
    }
  }

  return line;
}

void Writer::searchDir(const std::string root, std::vector<std::string>& fileNames){

  const std::string dir = root + '/';

  struct stat stat_buf;
  struct dirent** nameList = NULL;

  const int dirElements = scandir(dir.c_str(), &nameList, NULL, NULL);// for linux
  // const int dirElements = -1;// for windows

  if(dirElements == -1){
    std::cout << "ERROR BEFORE ROOP" << std::endl;
    return;
  }

  for(int i = 0; i < dirElements; ++i){

    char* ret;
    if( ( ret = strstr(nameList[i]->d_name, ".html") ) == NULL){
      continue;
    }

    std::string search_path = dir;
    search_path += std::string(nameList[i]->d_name);

    if(!stat(search_path.c_str(), &stat_buf)){
      fileNames.push_back(search_path);
    }
    else{
      std::cout << "ERROR WITHIN ROOP" << std::endl;
      return;
    }
  }
}

void Writer::formatFile(const std::string path){

  std::ifstream fin(path);

  std::string text;

  std::string line;

  const std::string flag0= "<";

  const std::string flag1 = "<a name=\"";
  const std::string flag2 = "\">[";
  const std::string flag3 = "]</a>";

  const std::string dummyL = "\"*DUMMY-";
  const std::string dummyR = "-DUMMY*\"";
  const int dummyLen = 8;

  // const std::string flag4 = "<a href=\"#";
  // const std::string flag5 = "\" id=\"";
  // const std::string flag6 = "</a>";

  while(getline(fin, line)){
    // getline does not put '\n' into line ?

    // remove forefornt spaces
    line.replace(0, line.find(flag0), "");

    if( (line.find(dummyL) != std::string::npos) && (line.find(dummyR) != std::string::npos) ){
      line.replace(line.find(dummyL), dummyLen, "");
      line.replace(line.find(dummyR), dummyLen, "");
    }

    if( (line.find(flag1) != std::string::npos) && (line.find(flag2) != std::string::npos) && (line.find(flag3) != std::string::npos) ){
      line += ' ';
    }
    else{
      line += '\n';
    }

    text += line;
  }

  fin.close();

  if(remove(path.c_str()) != 0){
    std::cout << path << " cannot removed" << std::endl;
    exit(0);
  }

  std::ofstream fout(path);
  fout << text;
  fout.close();
}

void Writer::formatDir(){
  // const std::string root = "/data/local/kinugawa/Eval/170721.30000/system/GOLD-STANDARD-SET";
  const std::string root = "/data/local/kinugawa/Eval/170721.30000/reference";
  const unsigned int thread_num = 8;

  std::vector<std::string> fileNames;
  Writer::searchDir(root, fileNames);

  const size_t i_end = fileNames.size();

  #pragma omp parallel for num_threads(thread_num) schedule(dynamic)
  for (size_t i = 0; i < i_end; ++i) {
    // std::cout << fileNames[i] << std::endl;
    Writer::formatFile(fileNames[i]);
  }
}

void Writer::save(){
  // for abstract

  const std::string path = "/data/local/kinugawa/path-list/170721/170721.30000.test.path.org";
  const std::string root = "/data/local/kinugawa/Eval/170721.30000/reference";
  const unsigned int thread_num = 8;

  Vocabulary voc(path, 0, 0);

  std::vector<Article*> docs;
  Article::set(path, docs, voc);

  Writer writer(root, voc);

#pragma omp parallel for num_threads(thread_num) schedule(dynamic) shared(writer)
  for (size_t i = 0; i < docs.size(); ++i) {
    writer.save(docs[i]);
  }

  Article::clear(docs);
}

void Writer::saveGoldStandardSet(){
  // for gold-standard-set

  const std::string path = "/data/local/kinugawa/path-list/170721/170721.30000.test.path.org";
  const std::string root = "/data/local/kinugawa/Eval/170721.30000/system/GOLD-STANDARD-SET";

  const unsigned int thread_num = 8;

  Vocabulary voc(path, 0, 0);

  std::vector<Article*> docs;
  Article::set(path, docs, voc);

  Writer writer(root, voc);

#pragma omp parallel for num_threads(thread_num) schedule(dynamic) shared(writer)
  for (size_t i = 0; i < docs.size(); ++i) {
    writer.save(docs[i], docs[i]->sentGoldLabel, docs[i]->sentGoldLabel.size());
  }

  Article::clear(docs);
}

void Writer::saveLeadSimple(){
  // for lead-simple

  const std::string path = "/data/local/kinugawa/path-list/170721/170721.30000.test.path.org";
  const std::string root = "/data/local/kinugawa/Eval/170721.30000/system/LEAD-SIMPLE";

  const unsigned int thread_num = 8;

  Vocabulary voc(path, 0, 0);

  std::vector<Article*> docs;
  Article::set(path, docs, voc);

  Writer writer(root, voc);

  const size_t i_end = docs.size();

#pragma omp parallel for num_threads(thread_num) schedule(dynamic) shared(writer)
  for (size_t i = 0; i < i_end; ++i) {

    const Article* doc = docs[i];
    std::vector<int> res;
    int sum = 0;
    for(int j = 0, j_end = doc->bodySentNum; j < j_end; ++j) {
      const Article::Sentence* sent = doc->bodySent[j];
      const int tmp = sent->wordNum;
      if(doc->resUpperUniNum >= sum + tmp) {
        sum += tmp;
        res.push_back(j);
      }
      else{
        break;
      }
    }

    writer.save(doc, res, res.size());
  }

  Article::clear(docs);
}

void Writer::saveProperty(){

  const std::string path = "/data/local/kinugawa/path-list/170721/170721.30000.test.path.org";
  const std::string output = "/data/local/kinugawa/Eval/170721.30000/property.csv";

  std::ofstream fout(output);

  Vocabulary voc(path, 0, 0);

  std::vector<Article*> docs;
  Article::set(path, docs, voc);

  fout << "file_name\t# of sentence\t# of paragraph\t# of section\t" << std::endl;;
  for(size_t i = 0, i_end = docs.size(); i < i_end; ++i){
    fout << docs[i]->fileName << '\t';
    fout << docs[i]->bodySentNum << '\t';
    fout << docs[i]->bodyPrgNum << '\t';
    fout << docs[i]->bodyUsedSecNum << '\t' << std::endl;;
  }

  Article::clear(docs);
}

void Writer::saveLeadSection(){
  // for lead-section

  const std::string path = "/data/local/kinugawa/path-list/170721/170721.30000.test.path.org";
  const std::string root = "/data/local/kinugawa/Eval/170721.30000/system/LEAD-SECTION";

  const unsigned int thread_num = 8;

  Vocabulary voc(path, 0, 0);

  std::vector<Article*> docs;
  Article::set(path, docs, voc);

  Writer writer(root, voc);

  const size_t i_end = docs.size();

#pragma omp parallel for num_threads(thread_num) schedule(dynamic) shared(writer)
  for (size_t i = 0; i < i_end; ++i) {

    const Article* doc = docs[i];
    std::vector<int> res;
    int sum = 0;
    for(int j = 0, j_end = doc->bodyUsedSecNum; j < j_end; ++j) {
      const Article::Section* sec = doc->bodyUsedSec[j];

      const int index = sec->sent[0];

      const Article::Sentence* sent = doc->bodySent[index];

      const int tmp = sent->wordNum;
      if(doc->resUpperUniNum >= sum + tmp) {
        sum += tmp;
        res.push_back(index);
      }
      else{
        break;
      }
    }

    writer.save(doc, res, res.size());
  }

  Article::clear(docs);
}
