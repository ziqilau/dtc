/*
 * corpus.h
 *
 *  Created on: 2013-3-4
 *      Author: ziqilau
 */

#ifndef CORPUS_H_
#define CORPUS_H_

#include <vector>
#include <map>
using namespace std;

class Document
{
public:
  Document();
  Document(int num_participants, int len_doc);
  ~Document();
  void InitAuthors(int num_participants);
  void InitWords(int len_doc);

public:
  int id_;               //document id
  int epoch_;            //document's epoch, must be consecutive integer begin from 0.
  int num_participants_;       //number of participants
  int* participants_;
  int length_;           //number of terms
  int total_;            //number of tokens
  int* words_;          //term
  int* counts_;         //term's count in this document
};

class Corpus
{
public:
  Corpus();
  ~Corpus();
  void ReadData(const char* filename);

public:
  int num_epoches_;             //consecutive epoches start from 0

  int total_participants_;
  int size_participants_;

  int size_vocab_;
  int total_tokens_;
  map<int, int> total_tokens_by_e_;

  int num_docs_;
  map<int, int> num_docs_by_e_;
  vector<Document*> docs_;
};

#endif /* CORPUS_H_ */
