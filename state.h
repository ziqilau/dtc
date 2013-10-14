/*
 * state.h
 *
 *  Created on: 2013-3-4
 *      Author: ziqilau
 */

#ifndef STATE_H_
#define STATE_H_

#include "corpus.h"

class DtcState;

class DtcHyperPara
{
public:
  DtcHyperPara();
  void SetPara(double alpha, double beta, double gamma,
              double tau, double zeta, double lambda, double lambda_con,
              int max_iter, int save_lag, int delta, int burnin,
              bool sample_hyperparameter, bool metropolis);

public:
  double zeta_;
  double tau_;
  double alpha_;
  double beta_;
  double gamma_;
  double lambda_;               //bigger the lambda, smoother the epoch weights, default is 0.5~1.0
  double lambda_con_;
  int delta_;                   //default epoches number is 4, this value must greater than 0
  int save_lag_;
  bool metropolis_hastings_;
  bool sample_hyperparameter_;
  int max_iter_;
  int burnin_;
};

struct TokenState
{
  int token_index_;
  int table_assignment_;
};

class DocState
{
public:
  explicit DocState(Document* doc);
  explicit DocState(DocState* d_state);
  ~DocState();
  void UpdateTokenState(DocState* d);

public:
  int doc_id_;
  int len_doc_;

  TokenState* tokens_;
  int comm_assignment_;

  int num_participants_;
  int* participants_;   //cannot be the same
};

class CommState
{
public:
  CommState();
  explicit CommState(int num_tables);
  CommState(int num_tables, int size_participants);
  CommState(CommState* c_state, int size_participants);
  ~CommState();
  int* CompactTables(int* k_to_new_k);
  void AddParti(DocState* d);
  void RemoveParti(DocState* d);
  void UpdateCommSS(CommState* c_state);

public:
  int num_docs_;
  int num_tables_;
  int num_tables_init_;
  bool* alive_b_;

  int word_counts_;
  vector<int> tables_to_topics_;
  vector<int> word_counts_by_b_;
  int parti_counts_;
  int* parti_counts_by_r_;

  double decay_kernel_word_counts_;
  double decay_kernel_num_docs_;
  vector<double> decay_kernel_word_counts_by_b_;
  double decay_kernel_zeta_;
  double* decay_kernel_zeta_by_r_;
};

class EpochState
{
public:
  EpochState(int e_id, int size_vocab, int size_participants);
  EpochState(Corpus* c, int e_id);
  EpochState(EpochState* e, DocState* d, int comm);
  EpochState();
  ~EpochState();
  void RemoveWord(DocState* doc_state, int word_index);
  void AddWord(DocState* doc_state, int word_index, int k);
  void SampleWordAssignment(bool remove, DocState* doc_state, int word_index, vector<double>& q_k,
      vector<double>& q_b, vector<double>& f, DtcHyperPara* hyper);
  void SampleTableAssignment(CommState* c_state, int b, vector<int>& words, vector<double>& q_k,
      vector<double>& f, DtcHyperPara* hyper);
  void BuildWordsByCommsTables(vector<vector<vector<int> > >& words_by_cb);
  void SampleTables(vector<double>& q_k, vector<double>& f, DtcHyperPara* hyper);
  double ComputeWordsLogLikelihood(DocState* doc_state, EpochState* e_new,
      int comm, DtcHyperPara* hyper, vector<double>& q_k, vector<double>& q_b, vector<double>& f);
  void BuildCandidateStateSS(EpochState** e_states_new, DocState* doc_state);
  void UpdateEpochSS(EpochState* e);
  double ComputePartiLogLikelihood(DocState* d, int comm, DtcHyperPara* hyper);
  void SampleCommunities(vector<double>& q, vector<double>& q_k, vector<double>& q_b,
      vector<double>& f, DtcHyperPara* hyper);
  double SampleCommunitiesMH(vector<double>& q, vector<double>& q_k, vector<double>& q_b,
      vector<double>& f, DtcHyperPara* hyper);
  void CompactEpochStates();
  void InitEpochGibbsState(DtcHyperPara* hyper);
  double NextEpochGibbsSweep(bool permute, DtcHyperPara* hyper);
  void ReshuffleEpoch();
  double DocPartitionLikelihood(DtcHyperPara* hyper);
  double CommPartitionLikelihood(CommState* c_state, DtcHyperPara* hyper);
  double TablePartitionLikelihood(DtcHyperPara* hyper);
  double DataLikelihood(DtcHyperPara* hyper);
  double JointLikelihood(DtcHyperPara* hyper);
  double GetPerplexity(DtcHyperPara* hyper);
  double GetPerplexityNosum(DtcHyperPara* hyper);
  double GetPartiLhood(DocState* d, int comm, DtcHyperPara* hyper);
  double GetWordLhood(DocState* d, int comm, DtcHyperPara* hyper);
  void CommPropGivenDoc(DocState* d_state, double* q_c, DtcHyperPara* hyper);
  void SaveState(const char* directory);
  void SaveState(const char* directory, int iter);
  void SaveModel(const char* directory);
  void LoadModel(const char* directory, int epoch);
  bool StateCheckSum();
  void EpochDebug();

public:
  //epoch-level info
  int epoch_id_;
  int num_docs_;
  int total_tokens_;
  int size_participants_;
  int size_vocab_;
  //states
  DocState** doc_states_;
  vector<CommState*> comm_states_;
  //ss
  bool* alive_c_;       //length equals num_comms_init_
  bool* alive_z_;       //length equals num_topics_init_
  int num_comms_;
  int num_topics_;
  int num_comms_init_;
  int num_topics_init_;

  int total_num_tables_;
  vector<int> num_tables_by_z_;
  vector<int> word_counts_by_z_;
  vector<int*> word_counts_by_zw_;

  double decay_kernel_num_tables_;
  vector<double> decay_kernel_num_tables_by_z_;
  vector<double> decay_kernel_tau_by_z_;
  vector<double*> decay_kernel_tau_by_zw_;
  double decay_kernel_num_docs_;
};

class DtcState
{
public:
  DtcState(int size_vocab, int total_tokens, int num_epoches);
  ~DtcState();
  void Reshuffle();
  double ComputeKernel(double lambda, int delta, int v1, double v2 = 0.0, int v3 = 0);
  void InitNextEpoch(int epoch, double lambda, double lambda_con, int delta);
  void InitNextEpochInfer(int epoch, double lambda, double lambda_con, int delta);
  //void NextGibbsSweep(bool permute, DtcHyperPara* hyper);

public:
  //corpus-level info
  int size_vocab_;
  int total_tokens_;
  int num_epoches_;
  //states
  EpochState** epoch_states_;
};

#endif /* STATE_H_ */
