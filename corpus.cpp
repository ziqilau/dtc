/*
 * corpus.cpp
 *
 *  Created on: 2013-3-4
 *      Author: ziqilau
 */

#include "corpus.h"
#include "utils.h"

Document::Document()
{
  id_ = -1;
  epoch_ = -1;
  num_participants_ = 0;
  participants_ = NULL;
  length_ = 0;
  total_ = 0;
  words_ = NULL;
  counts_ = NULL;
}

Document::Document(int num_participants, int len_doc)
{
  id_ = -1;
  epoch_ = -1;
  num_participants_ = num_participants;
  participants_ = new int [num_participants];
  length_ = len_doc;
  words_ = new int [len_doc];
  counts_ = new int [len_doc];
  total_ = 0;
}

Document::~Document()
{
  id_ = -1;
  epoch_ = -1;
  if(words_ != NULL)
  {
    delete [] words_;
    delete [] counts_;
    words_ = NULL;
    counts_ = NULL;
    length_ = 0;
    total_ = 0;
  }
  if(participants_ != NULL)
  {
    delete [] participants_;
    participants_ = NULL;
    num_participants_ = 0;
  }
}

void Document::InitAuthors(int num_participants)
{
  num_participants_ = num_participants;
  participants_ = new int [num_participants];
}

void Document::InitWords(int len_doc)
{
  length_ = len_doc;
  words_ = new int [len_doc];
  counts_ = new int [len_doc];
}

Corpus::Corpus()
{
  num_epoches_ = 0;

  total_participants_ = 0;
  size_participants_ = 0;

  size_vocab_ = 0;
  total_tokens_ = 0;
  total_tokens_by_e_.clear();

  num_docs_ = 0;
  num_docs_by_e_.clear();
  docs_.clear();
}

Corpus::~Corpus()
{
  int i = 0;
  Document * doc = NULL;
  for(i = 0; i < num_docs_; i++)
  {
    doc = docs_[i];
    if(doc != NULL)
    {
      delete doc;
      doc = NULL;
    }
  }
  docs_.clear();

  num_epoches_ = 0;
  total_participants_ = 0;
  size_participants_ = 0;
  size_vocab_ = 0;
  total_tokens_ = 0;
  total_tokens_by_e_.clear();
  num_docs_ = 0;
  num_docs_by_e_.clear();
}

void Corpus::ReadData(const char* filename)
{
  FILE * file_ptr;
  int length, count, word, n, nd, nw, epoch, num_partis, parti, epoch_start, epoch_end;

  // reading the data
  printf("\nreading data from %s\n", filename);

  file_ptr = fopen(filename, "r");
  nd = 0;
  nw = 0;
  epoch_start = epoch_end = -1;
  while((fscanf(file_ptr, "%d", &epoch) != EOF))
  {
    if(num_docs_by_e_.count(epoch))
      num_docs_by_e_[epoch]++;
    else
      num_docs_by_e_[epoch] = 1;
    Document * doc = new Document();
    doc->id_ = nd;
    doc->epoch_ = epoch;
    if(epoch_start > epoch || epoch_start == -1)
      epoch_start = epoch;
    if(epoch_end < epoch || epoch_end == -1)
      epoch_end = epoch;

    fscanf(file_ptr, "%d", &num_partis);
    doc->InitAuthors(num_partis);
    total_participants_ += num_partis;
    for(n = 0; n < num_partis; n++)
    {
      fscanf(file_ptr, "%d,", &parti);
      doc->participants_[n] = parti;
      if(parti >= size_participants_)
        size_participants_ = parti + 1;
    }

    fscanf(file_ptr, "%d", &length);
    doc->InitWords(length);
    for(n = 0; n < length; n++)
    {
      fscanf(file_ptr, "%d:%d", &word, &count);
      doc->words_[n] = word;
      doc->counts_[n] = count;
      doc->total_ += count;
      if(word >= nw)
        nw = word + 1;
    }

    if(total_tokens_by_e_.count(epoch))
      total_tokens_by_e_[epoch] += doc->total_;
    else
      total_tokens_by_e_[epoch] = doc->total_;

    total_tokens_ += doc->total_;
    docs_.push_back(doc);
    nd++;
  }
  num_docs_ = nd;
  size_vocab_ = nw;
  num_epoches_= epoch_end - epoch_start + 1;

  fclose(file_ptr); // close the file

  printf("number of epoches             :  %d\n", num_epoches_);
  printf("total size of participants    :  %d\n", size_participants_);
  printf("total number of docs          :  %d\n", nd);
  printf("total number of terms         :  %d\n", nw);
  printf("total number of total tokens  :  %d\n", total_tokens_);
}
