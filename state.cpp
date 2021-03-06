/*
 * state.cpp
 *
 *  Created on: 2013-3-5
 *      Author: ziqilau
 */

#include "state.h"
#include "corpus.h"
#include "utils.h"

#define INIT_Z_SIZE 50
#define INIT_C_SIZE 50
#define INIT_B_SIZE 100
#define EPSILON 1e-6

DtcHyperPara::DtcHyperPara()
{

}

void DtcHyperPara::SetPara(double alpha, double beta, double gamma,
                          double tau, double zeta, double lambda, double lambda_con,
                          int max_iter, int save_lag, int delta, int burnin,
                          bool sample_hyperparameter, bool metropolis)
{
  zeta_ = zeta;
  tau_ = tau;
  alpha_ = alpha;
  beta_ = beta;
  gamma_ = gamma;
  lambda_ = lambda;
  lambda_con_ = lambda_con;
  delta_ = delta;
  save_lag_ = save_lag;
  metropolis_hastings_ = metropolis;
  sample_hyperparameter_ = sample_hyperparameter;
  max_iter_ = max_iter;
  burnin_ = burnin;

  printf("\nHyper-parameter setting:\nalpha = %.2f\nbeta = %.2f\ngamma = %.2f\ntau = %.2f\nzeta = %.2f\nlambda = %.1f\nlambda_concentration = %.1f\nDelta = %2d\nsave_lag = %d\nmetropolis_hastings = %d\nsample_hyperparameter = %d\nmax_iter = %d\nburnin = %d\n\n",
      alpha_, beta_, gamma_,
      tau_, zeta_, lambda_, lambda_con_, delta_,
      save_lag_, metropolis_hastings_,
      sample_hyperparameter_, max_iter_, burnin_);
}


DocState::DocState(DocState* d_state)
{
  doc_id_ = d_state->doc_id_;
  len_doc_ = d_state->len_doc_;
  comm_assignment_ = d_state->comm_assignment_;
  tokens_ = new TokenState[len_doc_];
  memcpy(tokens_, d_state->tokens_, sizeof(TokenState)*len_doc_);

  participants_ = NULL;
}

DocState::DocState(Document* doc)
{
  doc_id_ = doc->id_;
  len_doc_ = doc->total_;
  comm_assignment_ = -1;
  tokens_ = new TokenState [len_doc_];
  int i, j, m;
  m = 0;
  for(i = 0; i < doc->length_; i++)
  {
    for(j = 0; j < doc->counts_[i]; j++)
    {
      tokens_[m].table_assignment_ = -1;
      tokens_[m].token_index_ = doc->words_[i];
      m++;
    }
  }
  num_participants_ = doc->num_participants_;
  participants_ = new int [num_participants_];
  memcpy(participants_, doc->participants_, sizeof(int)*doc->num_participants_);
}

DocState::~DocState()
{
  delete [] tokens_;
  tokens_ = NULL;
  delete [] participants_;
  participants_ = NULL;
}

void DocState::UpdateTokenState(DocState* d)
{
  assert(d->len_doc_ == len_doc_);
  memcpy(tokens_, d->tokens_, sizeof(TokenState)*len_doc_);
}


CommState::CommState()
{
  num_docs_ = 0;
  num_tables_ = 0;
  num_tables_init_ = 0;
  alive_b_ = NULL;

  word_counts_ = 0;
  tables_to_topics_.clear();
  word_counts_by_b_.clear();

  decay_kernel_word_counts_by_b_.clear();
  decay_kernel_num_docs_ = 0.0;
  decay_kernel_word_counts_ = 0.0;

  parti_counts_ = 0;
  parti_counts_by_r_ = NULL;
  decay_kernel_zeta_ = 0.0;
  decay_kernel_zeta_by_r_ = NULL;
}


//Do not consider participants info.
CommState::CommState(int num_tables)
{
  num_docs_ = 0;
  num_tables_ = 0;
  num_tables_init_ = 0;
  alive_b_ = NULL;

  word_counts_ = 0;
  tables_to_topics_.clear();
  tables_to_topics_.resize(num_tables, -1);
  word_counts_by_b_.clear();
  word_counts_by_b_.resize(num_tables, 0);

  decay_kernel_word_counts_by_b_.clear();
  decay_kernel_num_docs_ = 0.0;
  decay_kernel_word_counts_ = 0.0;

  parti_counts_ = 0;
  parti_counts_by_r_ = NULL;
  decay_kernel_zeta_ = 0.0;
  decay_kernel_zeta_by_r_ = NULL;
}

CommState::CommState(int num_tables, int size_participants)
{
  num_docs_ = 0;
  num_tables_ = 0;
  num_tables_init_ = 0;
  alive_b_ = NULL;

  word_counts_ = 0;
  tables_to_topics_.clear();
  tables_to_topics_.resize(num_tables, -1);
  word_counts_by_b_.clear();
  word_counts_by_b_.resize(num_tables, 0);

  decay_kernel_word_counts_by_b_.clear();
  decay_kernel_num_docs_ = 0.0;
  decay_kernel_word_counts_ = 0.0;

  parti_counts_ = 0;
  parti_counts_by_r_ = new int [size_participants];
  memset(parti_counts_by_r_, 0, sizeof(int)*size_participants);
  decay_kernel_zeta_ = 0.0;
  decay_kernel_zeta_by_r_ = NULL;
}

//build comm_state from last epoch's corresponding community state
CommState::CommState(CommState* c_state, int size_participants)
{
  num_docs_ = 0;
  num_tables_ = c_state->num_tables_;
  num_tables_init_ = c_state->num_tables_;
  alive_b_ = new bool[num_tables_init_];

  bool b_init = num_tables_init_ < INIT_B_SIZE;

  word_counts_ = 0;
  tables_to_topics_.clear();
  //table assignment states from previous epoch is fixed, and cannot be sampled in consecutive epoches.
  tables_to_topics_ = c_state->tables_to_topics_;
  tables_to_topics_.resize(b_init?INIT_B_SIZE:(num_tables_init_ + 1), -1);
  word_counts_by_b_.clear();
  word_counts_by_b_.resize(b_init?INIT_B_SIZE:(num_tables_init_ + 1), 0);

  decay_kernel_word_counts_ = 0.0;
  decay_kernel_num_docs_ = 0.0;
  decay_kernel_word_counts_by_b_.clear();
  decay_kernel_word_counts_by_b_.resize(num_tables_init_, 0.0);

  parti_counts_ = 0;
  parti_counts_by_r_ = new int [size_participants];
  memset(parti_counts_by_r_, 0, sizeof(int)*size_participants);
  decay_kernel_zeta_ = 0.0;
  decay_kernel_zeta_by_r_ = new double [size_participants];
  memset(decay_kernel_zeta_by_r_, 0.0, sizeof(double)*size_participants);
}

CommState::~CommState()
{
  tables_to_topics_.clear();
  word_counts_by_b_.clear();
  decay_kernel_word_counts_by_b_.clear();
  delete [] alive_b_;
  alive_b_ = NULL;
  delete [] parti_counts_by_r_;
  parti_counts_by_r_ = NULL;
  delete [] decay_kernel_zeta_by_r_;
  decay_kernel_zeta_by_r_ = NULL;
}

int* CommState::CompactTables(int* k_to_new_k)
{
  int* b_to_new_b = new int[num_tables_];

  int b, new_b, k;
  for (b = 0, new_b = 0; b < num_tables_; b++)
  {
    if (b < num_tables_init_ || word_counts_by_b_[b] > 0)
    {
      b_to_new_b[b] = new_b;
      if(!(b < num_tables_init_ && !alive_b_[b]))
      {
        k = tables_to_topics_[b];
        tables_to_topics_[new_b] = k_to_new_k[k];
      }
      swap_vec_element(word_counts_by_b_, new_b, b);
      new_b ++;
    }
    else
      tables_to_topics_[b] = -1;
  }
  num_tables_ = new_b;
  return b_to_new_b;
}

void CommState::AddParti(DocState* d)
{
  parti_counts_ += d->num_participants_;
  for(int r = 0; r < d->num_participants_; r++)
    parti_counts_by_r_[d->participants_[r]]++;
}

void CommState::RemoveParti(DocState* d)
{
  parti_counts_ -= d->num_participants_;
  for(int r = 0; r < d->num_participants_; r++)
    parti_counts_by_r_[d->participants_[r]]--;
}

void CommState::UpdateCommSS(CommState* c_state)
{
  num_docs_ = c_state->num_docs_;
  num_tables_ = c_state->num_tables_;
  word_counts_ = c_state->word_counts_;
  tables_to_topics_ = c_state->tables_to_topics_;
  word_counts_by_b_ = c_state->word_counts_by_b_;
}

EpochState::EpochState(int e_id, int size_vocab, int size_participants)
{
  epoch_id_ = e_id;
  num_docs_ = 0;
  total_tokens_ = 0;
  size_participants_ = size_participants;
  size_vocab_ = size_vocab;

  doc_states_ = NULL;
  comm_states_.clear();

  alive_c_ = NULL;
  alive_z_ = NULL;
  num_comms_ = 0;
  num_topics_ = 0;
  num_comms_init_ = 0;
  num_topics_init_ = 0;

  total_num_tables_ = 0;
  num_tables_by_z_.clear();
  word_counts_by_z_.clear();
  word_counts_by_zw_.clear();

  decay_kernel_num_tables_ = 0.0;
  decay_kernel_num_tables_by_z_.clear();
  decay_kernel_tau_by_z_.clear();
  decay_kernel_tau_by_zw_.clear();
  decay_kernel_num_docs_ = 0.0;
}


EpochState::EpochState(Corpus* c, int e_id)
{
  epoch_id_ = e_id;
  num_docs_ = c->num_docs_by_e_[e_id];
  total_tokens_ = c->total_tokens_by_e_[e_id];
  size_participants_ = c->size_participants_;
  size_vocab_ = c->size_vocab_;

  doc_states_ = new DocState* [num_docs_];
  memset(doc_states_, NULL, sizeof(DocState*)*num_docs_);
  comm_states_.clear();

  alive_c_ = NULL;
  alive_z_ = NULL;
  num_comms_ = 0;
  num_topics_ = 0;
  num_comms_init_ = 0;
  num_topics_init_ = 0;

  total_num_tables_ = 0;
  num_tables_by_z_.clear();
  word_counts_by_z_.clear();
  word_counts_by_zw_.clear();

  decay_kernel_num_tables_ = 0.0;
  decay_kernel_num_tables_by_z_.clear();
  decay_kernel_tau_by_z_.clear();
  decay_kernel_tau_by_zw_.clear();
  decay_kernel_num_docs_ = 0.0;
}

//copy epoch state, build one community state, one doc state, but decay_kernel informations.
EpochState::EpochState(EpochState* e, DocState* d, int comm)
{
  //copy epoch_state
  epoch_id_ = e->epoch_id_;
  num_docs_ = e->num_docs_;
  total_tokens_ = e->total_tokens_;
  size_participants_ = e->size_participants_;
  size_vocab_ = e->size_vocab_;

  num_comms_ = e->num_comms_;
  num_comms_init_ = e->num_comms_init_;
  if(e->num_comms_init_ > 0)
  {
    alive_c_ = new bool [e->num_comms_init_];
    memcpy(alive_c_, e->alive_c_, sizeof(bool)*e->num_comms_init_);
  }
  else
    alive_c_ = NULL;

  num_topics_ = e->num_topics_;
  num_topics_init_ = e->num_topics_init_;
  if(e->num_topics_init_ > 0)
  {
    alive_z_ = new bool [e->num_topics_init_];
    memcpy(alive_z_, e->alive_z_, sizeof(bool)*e->num_topics_init_);
  }
  else
    alive_z_ = NULL;

  total_num_tables_ = e->total_num_tables_;
  num_tables_by_z_ = e->num_tables_by_z_;
  word_counts_by_z_ = e->word_counts_by_z_;
  int size = e->word_counts_by_zw_.size();
  if(static_cast<int>(word_counts_by_zw_.size()) < size)
    word_counts_by_zw_.resize(size, NULL);
  for(int k = 0; k < size; k++)
  {
    if(e->word_counts_by_zw_[k])
    {
      word_counts_by_zw_[k] = new int[size_vocab_];
      memcpy(word_counts_by_zw_[k], e->word_counts_by_zw_[k], sizeof(int)*size_vocab_);
    }
  }

  decay_kernel_num_tables_ = e->decay_kernel_num_tables_;
  decay_kernel_num_tables_by_z_ = e->decay_kernel_num_tables_by_z_;
  decay_kernel_tau_by_z_ = e->decay_kernel_tau_by_z_;
  size = static_cast<int>(e->decay_kernel_tau_by_zw_.size());
  decay_kernel_tau_by_zw_.resize(size, NULL);
  for(int k = 0; k < size; k++)
  {
    if(e->decay_kernel_tau_by_zw_[k])
    {
      double* p = new double [size_vocab_];
      memcpy(p, e->decay_kernel_tau_by_zw_[k], sizeof(double)*size_vocab_);
      decay_kernel_tau_by_zw_[k] = p;
    }
  }
  decay_kernel_num_docs_ = e->decay_kernel_num_docs_;  //no use for sampling of words

  //copy doc_state
  doc_states_ = new DocState* [num_docs_];
  memset(doc_states_, NULL, sizeof(DocState*)*num_docs_);
  doc_states_[0] = new DocState(d);
  doc_states_[0]->comm_assignment_ = comm;      //modify the community assignment

  //copy comm_state
  comm_states_.resize(e->num_comms_ + 1, NULL);
  comm_states_[comm] = new CommState(INIT_B_SIZE);
  if(comm < e->num_comms_)
  {
    comm_states_[comm]->num_docs_ = e->comm_states_[comm]->num_docs_;
    comm_states_[comm]->num_tables_ = e->comm_states_[comm]->num_tables_;
    comm_states_[comm]->num_tables_init_ = e->comm_states_[comm]->num_tables_init_;
    if(comm_states_[comm]->num_tables_init_ > 0)
    {
      comm_states_[comm]->alive_b_ = new bool [e->comm_states_[comm]->num_tables_init_];
      memcpy(comm_states_[comm]->alive_b_, e->comm_states_[comm]->alive_b_, sizeof(bool)*(e->comm_states_[comm]->num_tables_init_));
    }

    comm_states_[comm]->word_counts_ = e->comm_states_[comm]->word_counts_;
    comm_states_[comm]->tables_to_topics_ = e->comm_states_[comm]->tables_to_topics_;
    comm_states_[comm]->word_counts_by_b_ = e->comm_states_[comm]->word_counts_by_b_;

    comm_states_[comm]->decay_kernel_word_counts_ = e->comm_states_[comm]->decay_kernel_word_counts_;
    comm_states_[comm]->decay_kernel_num_docs_ = e->comm_states_[comm]->decay_kernel_num_docs_;
    comm_states_[comm]->decay_kernel_word_counts_by_b_ = e->comm_states_[comm]->decay_kernel_word_counts_by_b_;
  }
}



EpochState::~EpochState()
{
  int c, j;

  delete [] alive_c_;
  alive_c_ = NULL;

  delete [] alive_z_;
  alive_z_ = NULL;

  num_tables_by_z_.clear();
  word_counts_by_z_.clear();

  free_vec_ptr(word_counts_by_zw_);

  decay_kernel_num_tables_by_z_.clear();
  decay_kernel_tau_by_z_.clear();

  free_vec_ptr(decay_kernel_tau_by_zw_);

  for(c = 0; c < static_cast<int>(comm_states_.size()); c++)
    delete comm_states_[c];
  comm_states_.clear();

  for(j = 0; j < num_docs_; j++)
  {
    delete doc_states_[j];
    doc_states_[j] = NULL;
  }
  delete [] doc_states_;
  doc_states_ = NULL;
}


void EpochState::RemoveWord(DocState* doc_state, int word_index)
{
  int w, b, k, comm;
  CommState* c_state;
  w = doc_state->tokens_[word_index].token_index_;
  b = doc_state->tokens_[word_index].table_assignment_;
  comm = doc_state->comm_assignment_;
  c_state = comm_states_[comm];
  k = c_state->tables_to_topics_[b];

  c_state->word_counts_--;
  c_state->word_counts_by_b_[b]--;
  word_counts_by_z_[k]--;
  word_counts_by_zw_[k][w]--;

  if(c_state->word_counts_by_b_[b] == 0)
  {
    total_num_tables_--;
    num_tables_by_z_[k]--;
    if(b >= c_state->num_tables_init_)
      c_state->tables_to_topics_[b] = -1;
  }
}

void EpochState::AddWord(DocState* doc_state, int word_index, int k)
{
  int w, b, comm;
  CommState* c_state;
  w = doc_state->tokens_[word_index].token_index_;
  b = doc_state->tokens_[word_index].table_assignment_;
  comm = doc_state->comm_assignment_;
  c_state = comm_states_[comm];

  c_state->word_counts_++;
  c_state->word_counts_by_b_[b]++;
  word_counts_by_z_[k]++;
  word_counts_by_zw_[k][w]++;

  if(c_state->word_counts_by_b_[b] == 1)
  {
    total_num_tables_++;
    num_tables_by_z_[k]++;
    c_state->tables_to_topics_[b] = k;
    if(b >= c_state->num_tables_init_)  //new table
    {
      assert(b == c_state->num_tables_);
      c_state->num_tables_++;
    }
    if(static_cast<int>(c_state->tables_to_topics_.size()) < c_state->num_tables_+1)
    {
      c_state->tables_to_topics_.push_back(-1);
      c_state->word_counts_by_b_.push_back(0);
    }

    if(k == num_topics_)        //new topic
    {
      assert(word_counts_by_z_[k] == 1);
      num_topics_++;
      if(static_cast<int>(num_tables_by_z_.size()) < num_topics_+1)
      {
        num_tables_by_z_.push_back(0);
        word_counts_by_z_.push_back(0);
        int* p = new int[size_vocab_];
        memset(p, 0, sizeof(int)*size_vocab_);
        word_counts_by_zw_.push_back(p);
      }
    }
  }
}

void EpochState::SampleWordAssignment(bool remove, DocState* doc_state, int word_index, vector<double>& q_k,
    vector<double>& q_b, vector<double>& f, DtcHyperPara* hyper)
{
  if(remove)
    RemoveWord(doc_state, word_index);

  int comm = doc_state->comm_assignment_;
  CommState* c_state = comm_states_[comm];
  int num_tables_c = c_state->num_tables_;

  if (static_cast<int>(q_k.size()) < num_topics_ + 1)
    q_k.resize(2 * num_topics_ + 1, 0.0);
  if (static_cast<int>(q_b.size()) < num_tables_c + 1)
    q_b.resize(2 * num_tables_c + 1, 0.0);
  if (static_cast<int>(f.size()) < num_topics_)
    f.resize(2 * num_topics_ + 1, 0.0);

  int k, b, w;
  w = doc_state->tokens_[word_index].token_index_;

  double f_new = 0.0;
  for(k = 0; k < num_topics_; k++)
  {
    if(k < num_topics_init_ && alive_z_[k])
    {
      f[k] = (hyper->tau_ + word_counts_by_zw_[k][w] + decay_kernel_tau_by_zw_[k][w])
                                          /(size_vocab_*hyper->tau_ + word_counts_by_z_[k] + decay_kernel_tau_by_z_[k]);
      f_new += (num_tables_by_z_[k] + decay_kernel_num_tables_by_z_[k]) * f[k];
    }
    else if(k >= num_topics_init_)
    {
      f[k] = (hyper->tau_ + word_counts_by_zw_[k][w])
                                          /(size_vocab_*hyper->tau_ + word_counts_by_z_[k]);
      f_new += (num_tables_by_z_[k]) * f[k];
    }
    else
      f[k] = 0.0;
    q_k[k] = f_new;
  }
  f_new += hyper->gamma_ / size_vocab_;
  q_k[num_topics_] = f_new;
  f_new = f_new / (total_num_tables_ + decay_kernel_num_tables_ + hyper->gamma_);

  double total_q = 0.0, f_k = 0.0;
  for(b = 0; b < num_tables_c; b++)
  {
    if(b < c_state->num_tables_init_ && c_state->alive_b_[b])
    {
      k = c_state->tables_to_topics_[b];
      f_k = f[k];
      total_q += (c_state->word_counts_by_b_[b] + c_state->decay_kernel_word_counts_by_b_[b]) * f_k;
    }
    if(b >= c_state->num_tables_init_)
    {
      if(c_state->word_counts_by_b_[b] > 0)
      {
        k = c_state->tables_to_topics_[b];
        f_k = f[k];
      }
      else f_k = 0.0;

      total_q += (c_state->word_counts_by_b_[b]) * f_k;
    }
    q_b[b] = total_q;
  }
  total_q += hyper->beta_ * f_new;
  q_b[num_tables_c] = total_q;

  double u = runiform() * total_q;
  for(b = 0; b < num_tables_c+1; b++)
    if(u < q_b[b]) break;

  doc_state->tokens_[word_index].table_assignment_ = b;
  if(b == num_tables_c)
  {
    u = runiform() * q_k[num_topics_];
    for(k = 0; k < num_topics_+1; k++)
      if(u < q_k[k]) break;

    AddWord(doc_state, word_index, k);
  }
  else
    AddWord(doc_state, word_index, c_state->tables_to_topics_[b]);
}

void EpochState::SampleTableAssignment(CommState* c_state, int b, vector<int>& words, vector<double>& q_k,
    vector<double>& f, DtcHyperPara* hyper)
{
  int i, w, k, k_old;
  int* counts = new int[size_vocab_];
  int* counts_copy = new int[size_vocab_];
  memset(counts_copy, 0, sizeof(int)*size_vocab_);
  int count_sum = c_state->word_counts_by_b_[b];
  for(i = 0; i < c_state->word_counts_by_b_[b]; i++)
  {
    w = words[i];
    counts_copy[w]++;
  }
  memcpy(counts, counts_copy, sizeof(int)*size_vocab_);

  double f_new = lgamma(size_vocab_*hyper->tau_) - lgamma(count_sum + size_vocab_*hyper->tau_);
  for(i = 0; i < c_state->word_counts_by_b_[b]; i++)
  {
    w = words[i];
    if (counts[w] > 0)
    {
      f_new += lgamma(counts[w]+hyper->tau_) - lgamma(hyper->tau_);
      counts[w] = 0;
    }
  }

  if(static_cast<int>(q_k.size()) < num_topics_ + 1)
    q_k.resize(2 * num_topics_+1, 0.0);

  if(static_cast<int>(f.size()) < num_topics_)
    f.resize(2 * num_topics_+1, 0.0);

  q_k[num_topics_] = log(hyper->gamma_) + f_new;

  k_old = c_state->tables_to_topics_[b];

  for(k = 0; k < num_topics_; k++)
  {
    if(k == k_old)
    {
      if(num_tables_by_z_[k] == 1 && k >= num_topics_init_)  q_k[k] = log(0);
      else
      {
        if(k < num_topics_init_)
        {
          assert(alive_z_[k]);
          f[k] = lgamma(size_vocab_*hyper->tau_ + decay_kernel_tau_by_z_[k] + word_counts_by_z_[k] - count_sum) -
              lgamma(size_vocab_*hyper->tau_ + decay_kernel_tau_by_z_[k] + word_counts_by_z_[k]);

          memcpy(counts, counts_copy, sizeof(int)*size_vocab_);
          for(i = 0; i < c_state->word_counts_by_b_[b]; i++)
          {
            w = words[i];
            if(counts[w] > 0)
            {
              f[k] += lgamma(hyper->tau_ + decay_kernel_tau_by_zw_[k][w] + word_counts_by_zw_[k][w]) -
                  lgamma(hyper->tau_ + decay_kernel_tau_by_zw_[k][w] + word_counts_by_zw_[k][w] - counts[w]);
              counts[w] = 0;
            }
          }
          q_k[k] = log(num_tables_by_z_[k] - 1 + decay_kernel_num_tables_by_z_[k]) + f[k];
        }
        else
        {
          f[k] = lgamma(size_vocab_*hyper->tau_ + word_counts_by_z_[k] - count_sum) -
              lgamma(size_vocab_*hyper->tau_ + word_counts_by_z_[k]);

          memcpy(counts, counts_copy, sizeof(int)*size_vocab_);
          for(i = 0; i < c_state->word_counts_by_b_[b]; i++)
          {
            w = words[i];
            if(counts[w] > 0)
            {
              f[k] += lgamma(hyper->tau_ + word_counts_by_zw_[k][w]) -
                  lgamma(hyper->tau_ + word_counts_by_zw_[k][w] - counts[w]);
              counts[w] = 0;
            }
          }
          q_k[k] = log(num_tables_by_z_[k] - 1) + f[k];
        }
      }
    }
    else
    {
      if(k < num_topics_init_ && alive_z_[k])
      {
        f[k] = lgamma(size_vocab_*hyper->tau_ + decay_kernel_tau_by_z_[k] + word_counts_by_z_[k]) -
            lgamma(size_vocab_*hyper->tau_ + decay_kernel_tau_by_z_[k] + word_counts_by_z_[k] + count_sum);

        memcpy(counts, counts_copy, sizeof(int)*size_vocab_);
        for(i = 0; i < c_state->word_counts_by_b_[b]; i++)
        {
          w = words[i];
          if(counts[w] > 0)
          {
            f[k] += lgamma(hyper->tau_ + decay_kernel_tau_by_zw_[k][w] + word_counts_by_zw_[k][w] + counts[w]) -
                lgamma(hyper->tau_ + decay_kernel_tau_by_zw_[k][w] + word_counts_by_zw_[k][w]);
            counts[w] = 0;
          }
        }
        q_k[k] = log(num_tables_by_z_[k] + decay_kernel_num_tables_by_z_[k]) + f[k];
      }
      else if(k >= num_topics_init_)
      {
        f[k] = lgamma(size_vocab_*hyper->tau_ + word_counts_by_z_[k]) -
            lgamma(size_vocab_*hyper->tau_ + word_counts_by_z_[k] + count_sum);

        memcpy(counts, counts_copy, sizeof(int)*size_vocab_);
        for(i = 0; i < c_state->word_counts_by_b_[b]; i++)
        {
          w = words[i];
          if(counts[w] > 0)
          {
            f[k] += lgamma(hyper->tau_ + word_counts_by_zw_[k][w] + counts[w]) -
                lgamma(hyper->tau_ + word_counts_by_zw_[k][w]);
            counts[w] = 0;
          }
        }
        q_k[k] = log(num_tables_by_z_[k]) + f[k];
      }
      else
        q_k[k] = log(0);
    }
  }

  //normalizing in log space for sampling
  log_normalize(q_k, num_topics_ + 1);
  q_k[0] = exp(q_k[0]);
  double total_q = q_k[0];
  for(k = 1; k < num_topics_+1; k++)
  {
    total_q += exp(q_k[k]);
    q_k[k] = total_q;
  }

  double u = runiform() * total_q;
  for(k = 0; k < num_topics_+1; k++)
    if(u < q_k[k]) break;

  if(k != k_old)
  {
    /// reassign the topic to current table
    c_state->tables_to_topics_[b] = k;

    /// update the statistics by removing the table b from topic k_old
    num_tables_by_z_[k_old]--;
    word_counts_by_z_[k_old] -= count_sum;

    /// update the statistics by adding the table t to topic k
    num_tables_by_z_[k]++;
    word_counts_by_z_[k] += count_sum;

    for(int i = 0; i < c_state->word_counts_by_b_[b]; i++)
    {
      w = words[i];
      word_counts_by_zw_[k_old][w]--;
      word_counts_by_zw_[k][w]++;
    }
    if(k == num_topics_) // a new topic is created
    {
      num_topics_++; // create a new topic
      if(static_cast<int>(num_tables_by_z_.size()) < num_topics_+1)
      {
        num_tables_by_z_.push_back(0);
        word_counts_by_z_.push_back(0);

        int* q = new int[size_vocab_];
        memset(q, 0, sizeof(int)*size_vocab_);
        word_counts_by_zw_.push_back(q);
      }
    }
  }

  delete [] counts;
  delete [] counts_copy;
}


void EpochState::BuildWordsByCommsTables(vector<vector<vector<int> > >& words_by_cb)
{
  int j, i, c, b, w, comm;
  vector<vector<int> > i_by_cb;
  words_by_cb.resize(num_comms_);
  i_by_cb.resize(num_comms_);
  for (c = 0; c < num_comms_; c++)
  {
    if((c < num_comms_init_ && alive_c_[c]) || (c >= num_comms_init_ && comm_states_[c]->num_docs_ > 0))
    {
      words_by_cb[c].resize(comm_states_[c]->num_tables_);        //maybe an alive zero num_tables_ community
      i_by_cb[c].resize(comm_states_[c]->num_tables_, 0);
      for (b = 0; b < comm_states_[c]->num_tables_; b++)
        words_by_cb[c][b].resize(comm_states_[c]->word_counts_by_b_[b], -1);
    }
  }

  for (j = 0; j < num_docs_; j++)
  {
    DocState* doc_state = doc_states_[j];
    comm = doc_state->comm_assignment_;
    for (i = 0; i < doc_state->len_doc_; i++)
    {
      w = doc_state->tokens_[i].token_index_;
      b = doc_state->tokens_[i].table_assignment_;
      words_by_cb[comm][b][i_by_cb[comm][b]] = w;
      i_by_cb[comm][b]++;
    }
  }
}


void EpochState::SampleTables(vector<double>& q_k, vector<double>& f, DtcHyperPara* hyper)
{
  int c, b;
  CommState* c_state;
  vector<vector<vector<int> > > words_by_cb;  //Hold all the words from current epoch by their communities and tables in.
  BuildWordsByCommsTables(words_by_cb);

  for(c = 0; c < num_comms_; c++)
  {
    if((c < num_comms_init_ && alive_c_[c]) || (c >= num_comms_init_ && comm_states_[c]->num_docs_ > 0))
    {
      c_state = comm_states_[c];
      for(b = 0; b < c_state->num_tables_; b++)
        if(b >= c_state->num_tables_init_ && c_state->word_counts_by_b_[b] > 0)
          SampleTableAssignment(c_state, b, words_by_cb[c][b], q_k, f, hyper);
    }
  }
}

double EpochState::ComputeWordsLogLikelihood(DocState* doc_state, EpochState* e_new,
    int comm, DtcHyperPara* hyper, vector<double>& q_k, vector<double>& q_b, vector<double>& f)
{
  int i, b, k;
  double words_loglikelihood = 0.0;

  CommState* c_state = e_new->comm_states_[comm];

  if(static_cast<int>(f.size()) < e_new->num_topics_)
    f.resize(2 * e_new->num_topics_ + 1, 0.0);

  for(i = 0; i < doc_state->len_doc_; i++)
  {
    int w = doc_state->tokens_[i].token_index_;

    double f_new = 0.0;
    for(k = 0; k < e_new->num_topics_; k++)
    {
      if(k < e_new->num_topics_init_ && e_new->alive_z_[k])
      {
        f[k] = (hyper->tau_ + e_new->word_counts_by_zw_[k][w] + e_new->decay_kernel_tau_by_zw_[k][w])
                       /(size_vocab_*hyper->tau_ + e_new->word_counts_by_z_[k] + e_new->decay_kernel_tau_by_z_[k]);
        f_new += f[k] * (e_new->num_tables_by_z_[k] + e_new->decay_kernel_num_tables_by_z_[k]);
      }
      else if(k >= e_new->num_topics_init_)
      {
        f[k] = (hyper->tau_ + e_new->word_counts_by_zw_[k][w])
                       /(size_vocab_*hyper->tau_ + e_new->word_counts_by_z_[k]);
        f_new += f[k] * (e_new->num_tables_by_z_[k]);
      }
      else
        f[k] = 0.0;
    }
    f_new += hyper->gamma_ / size_vocab_;
    f_new = f_new / (hyper->gamma_ + e_new->total_num_tables_ + e_new->decay_kernel_num_tables_);
    double p_new = 0.0;
    for(b = 0; b < c_state->num_tables_; b++)
    {
      k = c_state->tables_to_topics_[b];
      if(b < c_state->num_tables_init_ && c_state->alive_b_[b])
        p_new += f[k] * (c_state->word_counts_by_b_[b] + c_state->decay_kernel_word_counts_by_b_[b]);
      if(b >= c_state->num_tables_init_)
        p_new += f[k] * (c_state->word_counts_by_b_[b]);
    }
    p_new += hyper->beta_ * f_new;
    p_new = log(p_new) - log(c_state->word_counts_ + c_state->decay_kernel_word_counts_ + hyper->beta_);
    words_loglikelihood += p_new;

    if(doc_state->comm_assignment_ != comm)
      e_new->SampleWordAssignment(false, e_new->doc_states_[0], i, q_k, q_b, f, hyper);
    else        //update c_old'ss
    {
      int table_assign, topic_assign, w;
      table_assign = doc_state->tokens_[i].table_assignment_;
      w = doc_state->tokens_[i].token_index_;
      topic_assign = comm_states_[comm]->tables_to_topics_[table_assign];

      if(c_state->word_counts_by_b_[table_assign] == 0)
      {
        e_new->total_num_tables_++;
        e_new->num_tables_by_z_[topic_assign]++;
        c_state->tables_to_topics_[table_assign] = topic_assign;
      }
      c_state->word_counts_++;
      c_state->word_counts_by_b_[table_assign]++;
      e_new->word_counts_by_z_[topic_assign]++;
      e_new->word_counts_by_zw_[topic_assign][w]++;
    }
  }

  return words_loglikelihood;
}

/***
 * for each candidate epoch state:
 * 1. update its epoch level info/ss
 * 2. update its community level info/ss
 */
void EpochState::BuildCandidateStateSS(EpochState** e_states_new, DocState* doc_state)
{
  int i, c, table_assign, topic_assign, w, c_old;
  EpochState* e_new;
  c_old = doc_state->comm_assignment_;

  for(c = 0; c < num_comms_ + 1; c++)
  {
    if(e_states_new[c])
    {
      if(c!= c_old)
        e_states_new[c]->comm_states_[c]->num_docs_++;  //no use for sampling,  but for final state update.
      else
        e_states_new[c]->comm_states_[c]->num_docs_--;
    }
  }

  for(i = 0; i < doc_state->len_doc_; i++)
  {
    table_assign = doc_state->tokens_[i].table_assignment_;
    topic_assign = comm_states_[c_old]->tables_to_topics_[table_assign];
    w = doc_state->tokens_[i].token_index_;

    e_new = e_states_new[c_old];
    assert(e_new->comm_states_[c_old]->word_counts_by_b_[table_assign] > 0);
    e_new->comm_states_[c_old]->word_counts_--;
    e_new->comm_states_[c_old]->word_counts_by_b_[table_assign]--;

    if(e_new->comm_states_[c_old]->word_counts_by_b_[table_assign] == 0)
    {
      if(table_assign >= e_new->comm_states_[c_old]->num_tables_init_)
        e_new->comm_states_[c_old]->tables_to_topics_[table_assign] = -1;

      for(c = 0; c < num_comms_ + 1; c++)
      {
        if(e_states_new[c])
        {
          e_states_new[c]->total_num_tables_--;
          e_states_new[c]->num_tables_by_z_[topic_assign]--;
        }
      }
    }

    for(c = 0; c < num_comms_ + 1; c++)
    {
      if(e_states_new[c])
      {
        e_new = e_states_new[c];
        assert(e_new->word_counts_by_z_[topic_assign] > 0);
        e_new->word_counts_by_z_[topic_assign]--;
        e_new->word_counts_by_zw_[topic_assign][w]--;
      }
    }
  }
}




void EpochState::UpdateEpochSS(EpochState* e)
{
  num_comms_ = e->num_comms_;
  num_topics_ = e->num_topics_;
  total_num_tables_ = e->total_num_tables_;
  num_tables_by_z_ = e->num_tables_by_z_;
  word_counts_by_z_ = e->word_counts_by_z_;
  int size_e = (int)e->word_counts_by_zw_.size();
  int size = (int)word_counts_by_zw_.size();
  assert(size_e == (int)e->num_tables_by_z_.size());
  assert(size_e >= size);
  if(size < size_e)
    word_counts_by_zw_.resize(size_e, NULL);
  for(int k = 0; k < size_e; k++)
  {
    if(e->word_counts_by_zw_[k])
    {
      if(k >= size)
        word_counts_by_zw_[k] = new int[size_vocab_];
      memcpy(word_counts_by_zw_[k], e->word_counts_by_zw_[k], sizeof(int)*size_vocab_);
    }
    else
      assert(word_counts_by_zw_[k] == NULL);
  }
}


double EpochState::ComputePartiLogLikelihood(DocState* d, int comm, DtcHyperPara* hyper)
{
  double ret = 0.0;
  int r, parti;

  if(comm < num_comms_init_ && alive_c_[comm])
  {
    if(d->comm_assignment_ == comm)       //old one
    {
      ret = lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_ + comm_states_[comm]->parti_counts_ - d->num_participants_) -
          lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_ + comm_states_[comm]->parti_counts_);

      for(r = 0; r < d->num_participants_; r++)
      {
        parti = d->participants_[r];
        ret += lgamma(hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_by_r_[parti] + comm_states_[comm]->parti_counts_by_r_[parti]) -
            lgamma(hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_by_r_[parti] + comm_states_[comm]->parti_counts_by_r_[parti] - 1);
      }
    }
    else
    {
      ret = lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_ + comm_states_[comm]->parti_counts_) -
          lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_ + comm_states_[comm]->parti_counts_ + d->num_participants_);

      for(r = 0; r < d->num_participants_; r++)
      {
        parti = d->participants_[r];
        ret += lgamma(hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_by_r_[parti] + comm_states_[comm]->parti_counts_by_r_[parti] + 1) -
            lgamma(hyper->zeta_ + comm_states_[comm]->decay_kernel_zeta_by_r_[parti] + comm_states_[comm]->parti_counts_by_r_[parti]);
      }
    }
  }
  else if(comm >= num_comms_init_)
  {
    if(d->comm_assignment_ == comm)       //old one
    {
      ret = lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->parti_counts_ - d->num_participants_) -
          lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->parti_counts_);

      for(r = 0; r < d->num_participants_; r++)
      {
        parti = d->participants_[r];
        ret += lgamma(hyper->zeta_ + comm_states_[comm]->parti_counts_by_r_[parti]) -
            lgamma(hyper->zeta_ + comm_states_[comm]->parti_counts_by_r_[parti] - 1);
      }
    }
    else
    {
      ret = lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->parti_counts_) -
          lgamma(size_participants_*hyper->zeta_ + comm_states_[comm]->parti_counts_ + d->num_participants_);

      for(r = 0; r < d->num_participants_; r++)
      {
        parti = d->participants_[r];
        ret += lgamma(hyper->zeta_ + comm_states_[comm]->parti_counts_by_r_[parti] + 1) -
            lgamma(hyper->zeta_ + comm_states_[comm]->parti_counts_by_r_[parti]);
      }
    }
  }
  else
    ret = log(0);
  return ret;
}


void EpochState::SampleCommunities(vector<double>& q, vector<double>& q_k,
                                  vector<double>& q_b, vector<double>& f, DtcHyperPara* hyper)
{
  int j, c, comm_old;
  EpochState** e_states_new;

  for(j = 0; j < num_docs_; j++)
  {
    DocState* doc_state = doc_states_[j];
    comm_old = doc_state->comm_assignment_;
    const int num_comms_old = num_comms_;

    if(static_cast<int>(q.size()) < num_comms_ + 1)
      q.resize(2 * num_comms_ + 1, 0.0);

    e_states_new = new EpochState* [num_comms_ + 1];
    memset(e_states_new, NULL, sizeof(EpochState*)*(num_comms_+1));
    for(c = 0; c < num_comms_ + 1; c++)
      if((c < num_comms_init_ && alive_c_[c]) || (c == num_comms_) || (c >= num_comms_init_ && comm_states_[c]->num_docs_ > 0))
        e_states_new[c] = new EpochState(this, doc_state, c);

    BuildCandidateStateSS(e_states_new, doc_state);

    CommState* c_state_old = new CommState(INIT_B_SIZE);
    c_state_old->UpdateCommSS(e_states_new[comm_old]->comm_states_[comm_old]);

    double f_w, f_p;
    for(c = 0; c < num_comms_ + 1; c++)
    {
      if(c < num_comms_init_ && alive_c_[c])
      {
        f_w = ComputeWordsLogLikelihood(doc_state, e_states_new[c], c, hyper, q_k, q_b, f);
        f_p = ComputePartiLogLikelihood(doc_state, c, hyper);
        if(c == comm_old)
          q[c] = log(comm_states_[c]->num_docs_ - 1 + comm_states_[c]->decay_kernel_num_docs_) + f_w + f_p;
        else
          q[c] = log(comm_states_[c]->num_docs_ + comm_states_[c]->decay_kernel_num_docs_) + f_w + f_p;
      }
      else if(c == num_comms_)
      {
        f_w = ComputeWordsLogLikelihood(doc_state, e_states_new[c], c, hyper, q_k, q_b, f);
        f_p = ComputePartiLogLikelihood(doc_state, c, hyper);
        q[c] = log(hyper->alpha_) + f_w + f_p;
      }
      else if(c >= num_comms_init_ && comm_states_[c]->num_docs_ > 0)
      {
        f_w = ComputeWordsLogLikelihood(doc_state, e_states_new[c], c, hyper, q_k, q_b, f);
        f_p = ComputePartiLogLikelihood(doc_state, c, hyper);
        if(c == comm_old)
          q[c] = log(comm_states_[c]->num_docs_ - 1) + f_w + f_p;
        else
          q[c] = log(comm_states_[c]->num_docs_) + f_w + f_p;
      }
      else
        q[c] = log(0);
    }

    //sample comm
    log_normalize(q, num_comms_ + 1);
    q[0] = exp(q[0]);
    double total_q = q[0];
    for (c = 1; c < num_comms_ + 1; c++)
    {
      total_q += exp(q[c]);
      q[c] = total_q;
    }
    double u = runiform() * total_q;
    for(c = 0; c < num_comms_ + 1; c++)
      if (u < q[c]) break;

    //update states
    doc_state->comm_assignment_ = c;
    if(comm_old != c)
    {
      doc_state->UpdateTokenState(e_states_new[c]->doc_states_[0]);
      UpdateEpochSS(e_states_new[c]);
      comm_states_[c]->UpdateCommSS(e_states_new[c]->comm_states_[c]);
      comm_states_[comm_old]->UpdateCommSS(c_state_old);
      comm_states_[c]->AddParti(doc_state);
      comm_states_[comm_old]->RemoveParti(doc_state);
      if(c == num_comms_)
      {
        num_comms_++;
        if(static_cast<int>(comm_states_.size()) < num_comms_ + 1)
          comm_states_.push_back(new CommState(INIT_B_SIZE, size_participants_));
      }
    }

    //delete e_states_new
    for(c = 0; c < num_comms_old + 1; c++)
    {
      delete e_states_new[c];
      e_states_new[c] = NULL;
    }
    delete [] e_states_new;
    e_states_new = NULL;
    delete c_state_old;
    c_state_old = NULL;
  }
}


double EpochState::SampleCommunitiesMH(vector<double>& q, vector<double>& q_k,
                                  vector<double>& q_b, vector<double>& f, DtcHyperPara* hyper)
{
  int j, c, comm_old, proposal_count = 0, accpt_count = 0;
  double f_w, f_w_new, f_p, ratio;

  for(j = 0; j < num_docs_; j++)
  {
    DocState* doc_state = doc_states_[j];
    comm_old = doc_state->comm_assignment_;
    if(static_cast<int>(q.size()) < num_comms_ + 1)
      q.resize(2 * num_comms_ + 1, 0.0);

    for(c = 0; c < num_comms_ + 1; c++)
    {
      if(c < num_comms_init_ && alive_c_[c])
      {
        f_p = ComputePartiLogLikelihood(doc_state, c, hyper);
        if(c == comm_old)
          q[c] = log(comm_states_[c]->num_docs_ - 1 + comm_states_[c]->decay_kernel_num_docs_) + f_p;
        else
          q[c] = log(comm_states_[c]->num_docs_ + comm_states_[c]->decay_kernel_num_docs_) + f_p;
      }
      else if(c == num_comms_)
      {
        f_p = ComputePartiLogLikelihood(doc_state, c, hyper);
        q[c] = log(hyper->alpha_) + f_p;
      }
      else if(c >= num_comms_init_ && comm_states_[c]->num_docs_ > 0)
      {
        f_p = ComputePartiLogLikelihood(doc_state, c, hyper);
        if(c == comm_old)
          q[c] = log(comm_states_[c]->num_docs_ - 1) + f_p;
        else
          q[c] = log(comm_states_[c]->num_docs_) + f_p;
      }
      else
        q[c] = log(0);
    }

    //sample candidate comm
    log_normalize(q, num_comms_ + 1);
    q[0] = exp(q[0]);
    double total_q = q[0];
    for (c = 1; c < num_comms_ + 1; c++)
    {
      total_q += exp(q[c]);
      q[c] = total_q;
    }
    double u = runiform() * total_q;
    for(c = 0; c < num_comms_ + 1; c++)
      if (u < q[c]) break;

    if(comm_old == c)   //candidate community is the old community, do nothing.
      continue;
    else        //judge whether this candidate community could be a new sample.
    {
      proposal_count++;
      EpochState** e_states_new;
      e_states_new = new EpochState* [num_comms_ + 1];
      memset(e_states_new, NULL, sizeof(EpochState*)*(num_comms_+1));
      e_states_new[comm_old] = new EpochState(this, doc_state, comm_old);
      e_states_new[c] = new EpochState(this, doc_state, c);
      int num_comms_old = num_comms_;

      BuildCandidateStateSS(e_states_new, doc_state);
      CommState* c_state_old = new CommState(INIT_B_SIZE);
      c_state_old->UpdateCommSS(e_states_new[comm_old]->comm_states_[comm_old]);

      f_w = ComputeWordsLogLikelihood(doc_state, e_states_new[comm_old], comm_old, hyper, q_k, q_b, f);
      f_w_new = ComputeWordsLogLikelihood(doc_state, e_states_new[c], c, hyper, q_k, q_b, f);
      ratio = exp(f_w_new - f_w);

      if(ratio >= 1.0)
        doc_state->comm_assignment_ = c;
      else
      {
        u = runiform();
        if(u < ratio)
          doc_state->comm_assignment_ = c;
      }

      if(doc_state->comm_assignment_ != comm_old)
      {
        accpt_count++;
        doc_state->UpdateTokenState(e_states_new[c]->doc_states_[0]);
        UpdateEpochSS(e_states_new[c]);
        comm_states_[c]->UpdateCommSS(e_states_new[c]->comm_states_[c]);
        comm_states_[comm_old]->UpdateCommSS(c_state_old);
        comm_states_[c]->AddParti(doc_state);
        comm_states_[comm_old]->RemoveParti(doc_state);
        if(c == num_comms_)
        {
          num_comms_++;
          if(static_cast<int>(comm_states_.size()) < num_comms_ + 1)
            comm_states_.push_back(new CommState(INIT_B_SIZE, size_participants_));
        }
      }

      //delete e_states_new
      for(c = 0; c < num_comms_old + 1; c++)
      {
        delete e_states_new[c];
        e_states_new[c] = NULL;
      }
      delete [] e_states_new;
      e_states_new = NULL;
      delete c_state_old;
      c_state_old = NULL;
    }
  }
  return ((double)accpt_count)/proposal_count;
}


void EpochState::CompactEpochStates()
{
  //compact topics
  int* k_to_new_k = new int[num_topics_];
  int k, new_k;
  for(k = 0, new_k = 0; k < num_topics_; k++)
  {
    if(k < num_topics_init_ || word_counts_by_z_[k] > 0)
    {
      k_to_new_k[k] = new_k;
      swap_vec_element(word_counts_by_z_,  new_k, k);
      swap_vec_element(num_tables_by_z_,   new_k, k);
      swap_vec_element(word_counts_by_zw_, new_k, k);
      new_k ++;
    }
  }
  num_topics_ = new_k;

  //compact communities
  int* c_to_new_c = new int[num_comms_];
  int c, new_c;
  for(c = 0, new_c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ || comm_states_[c]->num_docs_ > 0)
    {
      c_to_new_c[c] = new_c;
      swap_vec_element(comm_states_,  new_c, c);
      new_c ++;
    }
  }
  num_comms_ = new_c;

  //compact tables
  vector<int*> b_to_new_b_by_c;
  b_to_new_b_by_c.resize(num_comms_, NULL);
  for(c = 0; c < num_comms_; c++)
  {
    CommState* c_state = NULL;
    c_state = comm_states_[c];
    if(c >= num_comms_init_ || alive_c_[c])
      b_to_new_b_by_c[c] = c_state->CompactTables(k_to_new_k);
  }

  //update docs
  int i, j, b, new_b;
  for(j = 0; j < num_docs_; j++)
  {
    DocState* d_state = doc_states_[j];
    c = d_state->comm_assignment_;
    new_c = c_to_new_c[c];
    d_state->comm_assignment_ = new_c;
    for(i = 0; i < d_state->len_doc_; i++)
    {
      b = d_state->tokens_[i].table_assignment_;
      new_b = b_to_new_b_by_c[new_c][b];
      d_state->tokens_[i].table_assignment_ = new_b;
    }
  }

  delete [] c_to_new_c;
  c_to_new_c = NULL;
  delete [] k_to_new_k;
  k_to_new_k = NULL;
  for(c = 0; c < num_comms_; c++)
    delete [] b_to_new_b_by_c[c];
  b_to_new_b_by_c.clear();
}

void EpochState::ReshuffleEpoch()
{
  DocState* doc_state;
  rshuffle(doc_states_, num_docs_, sizeof(DocState*));
  for(int j = 0; j < num_docs_; j++)
  {
    doc_state = doc_states_[j];
    rshuffle(doc_state->tokens_, doc_state->len_doc_, sizeof(TokenState));
  }
}

double EpochState::NextEpochGibbsSweep(bool permute, DtcHyperPara* hyper)
{
  if(permute)
    ReshuffleEpoch();

  int j, i;
  double mh_accpt_ratio = -1.0;
  vector<double> q_k, q_b, q_c, f;
  DocState* doc_state;

  //Sample a community indicator for each document in current epoch.
  if(hyper->metropolis_hastings_)
    mh_accpt_ratio = SampleCommunitiesMH(q_c, q_k, q_b, f, hyper);
  else
    SampleCommunities(q_c, q_k, q_b, f, hyper);

  //Sample a table assignment for each word in current epoch.
  for(j = 0; j < num_docs_; j++)
  {
    doc_state = doc_states_[j];
    for(i = 0; i < doc_state->len_doc_; i++)
      SampleWordAssignment(true, doc_state, i, q_k, q_b, f, hyper);
  }

  //Sample a topic assignment for each table in each community in current epoch.
  SampleTables(q_k, f, hyper);

  CompactEpochStates();

  //EpochDebug();

  return mh_accpt_ratio;
}

void EpochState::InitEpochGibbsState(DtcHyperPara* hyper)
{
  int i, j, c, r, parti;
  vector<double> q, q_k, f;

  ReshuffleEpoch();

  for(j = 0; j < num_docs_; j++)
  {
    if(static_cast<int>(q.size()) < num_comms_ + 1)
      q.resize(2 * num_comms_ + 1, 0.0);

    double total = 0.0;
    for(c = 0; c < num_comms_; c++)
    {
      if(c < num_comms_init_ && alive_c_[c])
        q[c] = comm_states_[c]->num_docs_ + comm_states_[c]->decay_kernel_num_docs_;
      else if(c >= num_comms_init_)
        q[c] = comm_states_[c]->num_docs_;
      else
        q[c] = 0.0;
      total += q[c];
      q[c] = total;
    }
    q[num_comms_] = hyper->alpha_ + total;

    double u = runiform() * q[num_comms_];
    for(c = 0; c < num_comms_ + 1; c++)
      if (u < q[c]) break;

    doc_states_[j]->comm_assignment_ = c;
    comm_states_[c]->num_docs_++;
    comm_states_[c]->parti_counts_ += doc_states_[j]->num_participants_;
    for(r = 0; r < doc_states_[j]->num_participants_; r++)
    {
      parti = doc_states_[j]->participants_[r];
      comm_states_[c]->parti_counts_by_r_[parti]++;
    }

    if(c == num_comms_)
    {
      num_comms_++;
      if(static_cast<int>(comm_states_.size()) < num_comms_ + 1)
        comm_states_.push_back(new CommState(INIT_B_SIZE, size_participants_));
    }

    for(i = 0; i < doc_states_[j]->len_doc_; i++)
      SampleWordAssignment(false, doc_states_[j], i, q_k, q, f, hyper);
  }
}

double EpochState::DocPartitionLikelihood(DtcHyperPara* hyper)
{
  int c, num_alive_comms = num_comms_;

  for(c = 0; c < num_comms_init_; c++)
    if(!alive_c_[c])
      num_alive_comms--;

  assert(num_alive_comms > 0);
  double likelihood = num_alive_comms * log(hyper->alpha_) - log_factorial(num_docs_ + decay_kernel_num_docs_, hyper->alpha_);
  for(c = 0; c < num_comms_; c++)
  {
    if(c >= num_comms_init_)
    {
      assert(comm_states_[c]->num_docs_ > 0);
      likelihood += lgamma(comm_states_[c]->num_docs_);
    }
    if(c < num_comms_init_ && alive_c_[c])
      likelihood += lgamma(comm_states_[c]->num_docs_ + comm_states_[c]->decay_kernel_num_docs_);
  }
  return likelihood;
}


double EpochState::CommPartitionLikelihood(CommState* c_state, DtcHyperPara* hyper)
{
  int b, num_alive_tables = c_state->num_tables_;

  for(b = 0; b < c_state->num_tables_init_; b++)
    if(!c_state->alive_b_[b])
      num_alive_tables--;

  assert(num_alive_tables > 0);
  double likelihood = num_alive_tables * log(hyper->beta_) - log_factorial(c_state->word_counts_ + c_state->decay_kernel_word_counts_, hyper->beta_);
  /// use n! = Gamma(n+1), that is log(n!) = lgamma(n+1)
  for(b = 0; b < c_state->num_tables_; b++)
  {
    if(b >= c_state->num_tables_init_)
    {
      assert(c_state->word_counts_by_b_[b] > 0);
      likelihood += lgamma(c_state->word_counts_by_b_[b]);
    }
    if(b < c_state->num_tables_init_ && c_state->alive_b_[b])
      likelihood += lgamma(c_state->word_counts_by_b_[b] + c_state->decay_kernel_word_counts_by_b_[b]);
  }
  return likelihood;
}

double EpochState::TablePartitionLikelihood(DtcHyperPara* hyper)
{
  int k, num_alive_topics = num_topics_;

  for(k = 0; k < num_topics_init_; k++)
    if(!alive_z_[k])
      num_alive_topics--;

  double likelihood = num_alive_topics * log(hyper->gamma_) - log_factorial(total_num_tables_ + decay_kernel_num_tables_, hyper->gamma_);
  /// use n! = Gamma(n+1), that is log(n!) = lgamma(n+1)
  for(k = 0; k < num_topics_; k++)
  {
    if(k >= num_topics_init_)
    {
      assert(num_tables_by_z_[k] > 0);
      likelihood += lgamma(num_tables_by_z_[k]);
    }
    if(k < num_topics_init_ && alive_z_[k])
      likelihood += lgamma(num_tables_by_z_[k] + decay_kernel_num_tables_by_z_[k]);
  }
  return likelihood;
}


double EpochState::DataLikelihood(DtcHyperPara* hyper)
{
  int k, w, c, p;
  double likelihood = 0.0;
  double lgamma_tau = lgamma(hyper->tau_);
  double lgamma_zeta = lgamma(hyper->zeta_);

  for(k = 0; k < num_topics_; k++)      //P(W|Z)
  {
    if(k >= num_topics_init_)
    {
      likelihood += lgamma(size_vocab_ * hyper->tau_);
      likelihood -= lgamma(size_vocab_ * hyper->tau_ + word_counts_by_z_[k]);
      for(w = 0; w < size_vocab_; w++)
        if(word_counts_by_zw_[k][w] > 0)
          likelihood += lgamma(word_counts_by_zw_[k][w] + hyper->tau_) - lgamma_tau;
    }
    if(k < num_topics_init_ && alive_z_[k])
    {
      likelihood += lgamma(size_vocab_ * hyper->tau_ + decay_kernel_tau_by_z_[k]);
      likelihood -= lgamma(size_vocab_ * hyper->tau_ + word_counts_by_z_[k] + decay_kernel_tau_by_z_[k]);
      for(w = 0; w < size_vocab_; w++)
        if(word_counts_by_zw_[k][w] > 0)
          likelihood += lgamma(word_counts_by_zw_[k][w] + hyper->tau_ + decay_kernel_tau_by_zw_[k][w])
            - lgamma(hyper->tau_ + decay_kernel_tau_by_zw_[k][w]);
    }
  }

  for(c = 0; c < num_comms_; c++)       //P(P|C)
  {
    CommState* c_state = comm_states_[c];
    if(c >= num_comms_init_)
    {
      likelihood += lgamma(size_participants_ * hyper->zeta_);
      likelihood -= lgamma(size_participants_ * hyper->zeta_ + c_state->parti_counts_);
      for(p = 0; p < size_participants_; p++)
        if(c_state->parti_counts_by_r_[p] > 0)
          likelihood += lgamma(c_state->parti_counts_by_r_[p] + hyper->zeta_) - lgamma_zeta;
    }
    if(c < num_comms_init_ && alive_c_[c])
    {
      likelihood += lgamma(size_participants_ * hyper->zeta_ + c_state->decay_kernel_zeta_);
      likelihood -= lgamma(size_participants_ * hyper->zeta_ + c_state->parti_counts_ + c_state->decay_kernel_zeta_);
      for(p = 0; p < size_participants_; p++)
        if(c_state->parti_counts_by_r_[p] > 0)
          likelihood += lgamma(c_state->parti_counts_by_r_[p] + hyper->zeta_ + c_state->decay_kernel_zeta_by_r_[p])
            - lgamma(hyper->zeta_ + c_state->decay_kernel_zeta_by_r_[p]);
    }
  }

  return likelihood;
}



double EpochState::JointLikelihood(DtcHyperPara* hyper)
{
  double likelihood = 0.0;
  likelihood += DocPartitionLikelihood(hyper);                          //doc partition
  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])  continue;
    likelihood += CommPartitionLikelihood(comm_states_[c], hyper);      //comm partition
  }
  likelihood += TablePartitionLikelihood(hyper);                        //table partition
  likelihood += DataLikelihood(hyper);                                  //P(W|Z)*P(P|C), Dirichlet-multinomial

  return likelihood;
}


double EpochState::GetPartiLhood(DocState* d, int comm, DtcHyperPara* hyper)
{
  double ret = 0.0;
  int parti;

  if(comm < num_comms_init_ && alive_c_[comm])
  {
    for(int r = 0; r < d->num_participants_; r++)
    {
      parti = d->participants_[r];
      ret += log( (hyper->zeta_ + comm_states_[comm]->parti_counts_by_r_[parti] + comm_states_[comm]->decay_kernel_zeta_by_r_[parti])
          / (hyper->zeta_*size_participants_ + comm_states_[comm]->parti_counts_ + comm_states_[comm]->decay_kernel_zeta_) );
    }
  }

  if(comm >= num_comms_init_)
  {
    for(int r = 0; r < d->num_participants_; r++)
    {
      parti = d->participants_[r];
      ret += log( (hyper->zeta_ + comm_states_[comm]->parti_counts_by_r_[parti])
          / (hyper->zeta_*size_participants_ + comm_states_[comm]->parti_counts_) );
    }
  }

  return ret;
}

double EpochState::GetWordLhood(DocState* d, int comm, DtcHyperPara* hyper)
{
  double ret = 0.0;
  CommState* c_state = comm_states_[comm];
  double* f = new double [num_topics_];
  memset(f, 0.0, sizeof(double)*num_topics_);

  for(int i = 0; i < d->len_doc_; i++)
  {
    int w = d->tokens_[i].token_index_;

    double f_new = 0.0;
    for(int k = 0; k < num_topics_; k++)
    {
      if(k < num_topics_init_ && alive_z_[k])
      {
        f[k] = (hyper->tau_ + word_counts_by_zw_[k][w] + decay_kernel_tau_by_zw_[k][w])
                                            /(size_vocab_*hyper->tau_ + word_counts_by_z_[k] + decay_kernel_tau_by_z_[k]);
        f_new += f[k] * (num_tables_by_z_[k] + decay_kernel_num_tables_by_z_[k]);
      }
      else if(k >= num_topics_init_)
      {
        f[k] = (hyper->tau_ + word_counts_by_zw_[k][w])
                                            /(size_vocab_*hyper->tau_ + word_counts_by_z_[k]);
        f_new += f[k] * (num_tables_by_z_[k]);
      }
      else
        f[k] = 0.0;
    }
    f_new += hyper->gamma_ / size_vocab_;
    f_new = f_new / (hyper->gamma_ + total_num_tables_ + decay_kernel_num_tables_);

    double p_new = 0.0;
    for(int b = 0; b < c_state->num_tables_; b++)
    {
      int k = c_state->tables_to_topics_[b];
      if(b < c_state->num_tables_init_ && c_state->alive_b_[b])
        p_new += f[k] * (c_state->word_counts_by_b_[b] + c_state->decay_kernel_word_counts_by_b_[b]);
      if(b >= c_state->num_tables_init_)
        p_new += f[k] * (c_state->word_counts_by_b_[b]);
    }
    p_new += hyper->beta_ * f_new;
    p_new = log(p_new) - log(c_state->word_counts_ + c_state->decay_kernel_word_counts_ + hyper->beta_);
    ret += p_new;
  }
  delete [] f;
  return ret;
}

void EpochState::CommPropGivenDoc(DocState* d_state, double* q_c, DtcHyperPara* hyper)
{
  for(int c = 0; c < num_comms_; c++)
  {
    CommState* c_state = comm_states_[c];
    if(c < num_comms_init_ && !alive_c_[c])
    {
      q_c[c] = log(0);
      continue;
    }

    double f = 0.0;
    f += GetPartiLhood(d_state, c, hyper);
    //f += GetWordLhood(d_state, c, hyper);
    f += log( (hyper->alpha_ + c_state->num_docs_) / (hyper->alpha_*num_comms_ + num_docs_) );
    q_c[c] = f;
  }
  log_normalize(q_c, num_comms_);
  q_c[0] = exp(q_c[0]);
  double total_q = q_c[0];
  for(int c = 1; c < num_comms_; c++)
  {
    q_c[c] = exp(q_c[c]);
    total_q += q_c[c];
  }
}
/*
double EpochState::GetPerplexity(DtcHyperPara* hyper)
{
  int j, i, c, b, k, w;
  int num_alive_comms = num_comms_;
  double perp = 0.0;

  double* q_c = new double [num_comms_];
  double* f = new double [num_topics_];
  memset(q_c, 0.0, sizeof(double)*num_comms_);
  memset(f, 0.0, sizeof(double)*num_topics_);

  for(j = 0; j < num_docs_; j++)
  {
    DocState* doc_state = doc_states_[j];
    CommPropGivenDoc(doc_state, q_c, hyper);
    for(i = 0; i < doc_state->len_doc_; i++)
    {
      w = doc_state->tokens_[i].token_index_;

      double f_new = 0.0;
      for(k = 0; k < num_topics_; k++)
      {
        if(k < num_topics_init_ && alive_z_[k])
        {
          f[k] = (hyper->tau_ + word_counts_by_zw_[k][w] + decay_kernel_tau_by_zw_[k][w])
                        /(size_vocab_*hyper->tau_ + word_counts_by_z_[k] + decay_kernel_tau_by_z_[k]);
          f_new += f[k] * (num_tables_by_z_[k] + decay_kernel_num_tables_by_z_[k]);
        }
        else if(k >= num_topics_init_)
        {
          f[k] = (hyper->tau_ + word_counts_by_zw_[k][w])
                        /(size_vocab_*hyper->tau_ + word_counts_by_z_[k]);
          f_new += f[k] * (num_tables_by_z_[k]);
        }
        else
          f[k] = 0.0;
      }
      f_new += hyper->gamma_ / size_vocab_;
      f_new = f_new / (hyper->gamma_ + total_num_tables_ + decay_kernel_num_tables_);

      double p_w = 0.0;
      for(c = 0; c < num_comms_; c++)       // sum_c sum_z sum_b P(c)P(b|c)P(w|z)
      {
        if((c < num_comms_init_ && alive_c_[c]) || c >= num_comms_init_)
        {
          CommState* c_state = comm_states_[c];

          double p_new = 0.0;
          for(b = 0; b < c_state->num_tables_; b++)
          {
            k = c_state->tables_to_topics_[b];
            if(b < c_state->num_tables_init_ && c_state->alive_b_[b])
              p_new += f[k] * (c_state->word_counts_by_b_[b] + c_state->decay_kernel_word_counts_by_b_[b]);
            if(b >= c_state->num_tables_init_)
              p_new += f[k] * (c_state->word_counts_by_b_[b]);
          }
          p_new += hyper->beta_ * f_new;
          //p_new = log(p_new) - log(c_state->word_counts_ + c_state->decay_kernel_word_counts_ + hyper->beta_);
          p_new = (p_new) / (c_state->word_counts_ + c_state->decay_kernel_word_counts_ + hyper->beta_);

          //p_w += exp(p_new + q_c[c]);
          p_w += p_new * q_c[c];
        }
      }
      perp += log(p_w);
    }
  }

  delete [] q_c;
  delete [] f;

  perp = exp(-perp/total_tokens_);
  return perp;
}*/
/*
double EpochState::GetPerplexity(DtcHyperPara* hyper)
{
  int j, i, c, b, k, w;
  double perp = 0.0;

  double* q_k = new double [num_topics_];
  double* f = new double [num_topics_];
  memset(f, 0.0, sizeof(double)*num_topics_);

  for(j = 0; j < num_docs_; j++)
  {
    DocState* doc_state = doc_states_[j];
    memset(q_k, 0.0, sizeof(double)*num_topics_);
    for(i = 0; i < doc_state->len_doc_; i++)
    {
      c = doc_state->comm_assignment_;
      CommState* c_state = comm_states_[c];
      w = doc_state->tokens_[i].token_index_;
      b = doc_state->tokens_[i].table_assignment_;
      k = c_state->tables_to_topics_[b];
      q_k[k] += 1.0;
    }
    for(k = 0; k < num_topics_; k++)
    {
      q_k[k] = q_k[k]/doc_state->len_doc_;
    }

    for(i = 0; i < doc_state->len_doc_; i++)
    {
      w = doc_state->tokens_[i].token_index_;

      double f_new = 0.0;
      for(k = 0; k < num_topics_; k++)
      {
        if(k < num_topics_init_ && alive_z_[k])
        {
          f[k] = (hyper->tau_ + word_counts_by_zw_[k][w] + decay_kernel_tau_by_zw_[k][w])
                            /(size_vocab_*hyper->tau_ + word_counts_by_z_[k] + decay_kernel_tau_by_z_[k]);
          f_new += f[k] * (num_tables_by_z_[k] + decay_kernel_num_tables_by_z_[k]);
        }
        else if(k >= num_topics_init_)
        {
          f[k] = (hyper->tau_ + word_counts_by_zw_[k][w])
                            /(size_vocab_*hyper->tau_ + word_counts_by_z_[k]);
          f_new += f[k] * (num_tables_by_z_[k]);
        }
        else
          f[k] = 0.0;
      }
      f_new += hyper->gamma_ / size_vocab_;
      f_new = f_new / (hyper->gamma_ + total_num_tables_ + decay_kernel_num_tables_);

      double p_w = 0.0;
      for(k = 0; k < num_topics_; k++)
      {
        p_w += f[k] * q_k[k];
      }
      perp += log(p_w);
    }
  }

  delete [] q_k;
  delete [] f;

  perp = exp(-perp/total_tokens_);
  return perp;
}*/



double EpochState::GetPerplexity(DtcHyperPara* hyper)
{
  int j, i, b, k, w;
  double perp = 0.0;

  double* f = new double [num_topics_];
  memset(f, 0.0, sizeof(double)*num_topics_);

  for(j = 0; j < num_docs_; j++)
  {
    double p_d = 0.0;
    DocState* doc_state = doc_states_[j];
    for(i = 0; i < doc_state->len_doc_; i++)
    {
      w = doc_state->tokens_[i].token_index_;

      double f_new = 0.0;
      for(k = 0; k < num_topics_; k++)
      {
        if(k < num_topics_init_ && alive_z_[k])
        {
          f[k] = (hyper->tau_ + word_counts_by_zw_[k][w] + decay_kernel_tau_by_zw_[k][w])
                        /(size_vocab_*hyper->tau_ + word_counts_by_z_[k] + decay_kernel_tau_by_z_[k]);
          f_new += f[k] * (num_tables_by_z_[k] + decay_kernel_num_tables_by_z_[k]);
        }
        else if(k >= num_topics_init_)
        {
          f[k] = (hyper->tau_ + word_counts_by_zw_[k][w])
                        /(size_vocab_*hyper->tau_ + word_counts_by_z_[k]);
          f_new += f[k] * (num_tables_by_z_[k]);
        }
        else
          f[k] = 0.0;
      }
      f_new += hyper->gamma_ / size_vocab_;
      f_new = f_new / (hyper->gamma_ + total_num_tables_ + decay_kernel_num_tables_);


      CommState* c_state = comm_states_[doc_states_[j]->comm_assignment_];

      double p_new = 0.0;
      for(b = 0; b < c_state->num_tables_; b++)
      {
        k = c_state->tables_to_topics_[b];
        if(b < c_state->num_tables_init_ && c_state->alive_b_[b])
          p_new += f[k] * (c_state->word_counts_by_b_[b] + c_state->decay_kernel_word_counts_by_b_[b]);
        if(b >= c_state->num_tables_init_)
          p_new += f[k] * (c_state->word_counts_by_b_[b]);
      }
      p_new += hyper->beta_ * f_new;
      p_new = log(p_new) - log(c_state->word_counts_ + c_state->decay_kernel_word_counts_ + hyper->beta_);
      perp += p_new;
      p_d += p_new;
    }
  }

  delete [] f;

  return exp(-perp/total_tokens_);
}


double EpochState::GetPerplexityNosum(DtcHyperPara* hyper)
{
  int j, i, b, k, w;
  double perp = 0.0;

  double f;

  for(j = 0; j < num_docs_; j++)
  {
    DocState* doc_state = doc_states_[j];
    CommState* c_state = comm_states_[doc_state->comm_assignment_];
    for(i = 0; i < doc_state->len_doc_; i++)
    {
      w = doc_state->tokens_[i].token_index_;
      b = doc_state->tokens_[i].table_assignment_;
      k = c_state->tables_to_topics_[b];
      f = (hyper->tau_ + word_counts_by_zw_[k][w])/(size_vocab_*hyper->tau_ + word_counts_by_z_[k]);

      perp += log(f);
    }
  }

  return exp(-perp/total_tokens_);
}

void EpochState::SaveState(const char* directory)
{
  int comm, w, k, b, j, i, p;
  CommState* c_state;
  DocState* d_state;

  char filename[500];

  sprintf(filename, "%s/model-topic-epoch%d.dat", directory, epoch_id_);
  FILE* file = fopen(filename, "w");
  for(k = 0; k < num_topics_; k++)
  {
    if(k >= num_topics_init_ || (k < num_topics_init_ && alive_z_[k]))
    {
      for(w = 0; w < size_vocab_; w++)
        fprintf(file, "%d ", word_counts_by_zw_[k][w]);
      fprintf(file, "\n");
    }
    else
      fprintf(file, "\n");
  }
  fclose(file);

  sprintf(filename, "%s/model-parti-epoch%d.dat", directory, epoch_id_);
  file = fopen(filename, "w");
  for(comm = 0; comm < num_comms_; comm++)
  {
    c_state = comm_states_[comm];
    if(comm >= num_comms_init_ || (comm < num_comms_init_ && alive_c_[comm]))
    {
      for(p = 0; p < size_participants_; p++)
        fprintf(file, "%d ", c_state->parti_counts_by_r_[p]);
      fprintf(file, "\n");
    }
    else
      fprintf(file, "\n");
  }
  fclose(file);

  sprintf(filename, "%s/model-state-epoch%d.dat", directory, epoch_id_);
  file = fopen(filename, "w");
  for (j = 0; j < num_docs_; j++)
  {
    d_state = doc_states_[j];
    int doc_id = d_state->doc_id_;
    for(i = 0; i < d_state->len_doc_; i++)
    {
      comm = d_state->comm_assignment_;
      c_state = comm_states_[comm];
      w = d_state->tokens_[i].token_index_;
      b = d_state->tokens_[i].table_assignment_;
      k = c_state->tables_to_topics_[b];
      fprintf(file, "%d %d %d %d %d\n", doc_id, comm, w, b, k);
    }
  }
  fclose(file);
}


void EpochState::SaveState(const char* directory, int iter)
{
  int comm, w, k, b, j, i, p;
  CommState* c_state;
  DocState* d_state;

  char filename[500];

  sprintf(filename, "%s/model-topic-epoch%d-%d.dat", directory, epoch_id_, iter);
  FILE* file = fopen(filename, "w");
  for(k = 0; k < num_topics_; k++)
  {
    if(k >= num_topics_init_ || (k < num_topics_init_ && alive_z_[k]))
    {
      for(w = 0; w < size_vocab_; w++)
        fprintf(file, "%d ", word_counts_by_zw_[k][w]);
      fprintf(file, "\n");
    }
    else
      fprintf(file, "\n");
  }
  fclose(file);

  sprintf(filename, "%s/model-parti-epoch%d-%d.dat", directory, epoch_id_, iter);
  file = fopen(filename, "w");
  for(comm = 0; comm < num_comms_; comm++)
  {
    c_state = comm_states_[comm];
    if(comm >= num_comms_init_ || (comm < num_comms_init_ && alive_c_[comm]))
    {
      for(p = 0; p < size_participants_; p++)
        fprintf(file, "%d ", c_state->parti_counts_by_r_[p]);
      fprintf(file, "\n");
    }
    else
      fprintf(file, "\n");
  }
  fclose(file);

  sprintf(filename, "%s/model-state-epoch%d-%d.dat", directory, epoch_id_, iter);
  file = fopen(filename, "w");
  for (j = 0; j < num_docs_; j++)
  {
    d_state = doc_states_[j];
    int doc_id = d_state->doc_id_;
    for(i = 0; i < d_state->len_doc_; i++)
    {
      comm = d_state->comm_assignment_;
      c_state = comm_states_[comm];
      w = d_state->tokens_[i].token_index_;
      b = d_state->tokens_[i].table_assignment_;
      k = c_state->tables_to_topics_[b];
      fprintf(file, "%d %d %d %d %d\n", doc_id, comm, w, b, k);
    }
  }
  fclose(file);
}


void EpochState::SaveModel(const char* directory)
{

  char name[500];
  sprintf(name, "%s/e%d_model", directory, epoch_id_);
  FILE * file = fopen(name, "wb");

  sprintf(name, "%s/save%d", directory, epoch_id_);
  FILE * f = fopen(name, "w");

  fwrite(&num_topics_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_topics_);
  fwrite(&num_comms_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_comms_);
  fwrite(&num_topics_init_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_topics_init_);
  fwrite(&num_comms_init_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_comms_init_);
  if(num_topics_init_ > 0)
    fwrite(alive_z_, sizeof(bool), num_topics_init_, file);
  if(num_comms_init_ > 0)
    fwrite(alive_c_, sizeof(bool), num_comms_init_, file);

  fwrite(&total_num_tables_, sizeof(int), 1, file);
  fprintf(f, "%d ", total_num_tables_);
  fwrite(&decay_kernel_num_tables_, sizeof(double), 1, file);
  fprintf(f, "%f ", decay_kernel_num_tables_);
  fwrite(&decay_kernel_num_docs_, sizeof(double), 1, file);
  fprintf(f, "%f ", decay_kernel_num_docs_);

  for(int k = 0; k < num_topics_; k++)
  {
    if(k < num_topics_init_ && !alive_z_[k])    continue;
    fwrite(&(num_tables_by_z_[k]), sizeof(int), 1, file);
    fwrite(&(word_counts_by_z_[k]), sizeof(int), 1, file);
    fwrite(word_counts_by_zw_[k], sizeof(int), size_vocab_, file);
    if(num_topics_init_ > 0 && k < num_topics_init_)
    {
      fwrite(&(decay_kernel_num_tables_by_z_[k]), sizeof(double), 1, file);
      fwrite(&(decay_kernel_tau_by_z_[k]), sizeof(double), 1, file);
      fwrite(decay_kernel_tau_by_zw_[k], sizeof(double), size_vocab_, file);
    }
  }

  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])     continue;
    fwrite(&(comm_states_[c]->num_docs_), sizeof(int), 1, file);
    fwrite(&(comm_states_[c]->num_tables_), sizeof(int), 1, file);
    fwrite(&(comm_states_[c]->num_tables_init_), sizeof(int), 1, file);
    fwrite(&(comm_states_[c]->word_counts_), sizeof(int), 1, file);
    fwrite(&(comm_states_[c]->parti_counts_), sizeof(int), 1, file);
    fwrite(&(comm_states_[c]->decay_kernel_word_counts_), sizeof(double), 1, file);
    fwrite(&(comm_states_[c]->decay_kernel_num_docs_), sizeof(double), 1, file);
    fwrite(&(comm_states_[c]->decay_kernel_zeta_), sizeof(double), 1, file);
    fwrite(comm_states_[c]->parti_counts_by_r_, sizeof(int), size_participants_, file);

    fprintf(f, "%d %d %d %d ", comm_states_[c]->num_docs_, comm_states_[c]->num_tables_, comm_states_[c]->num_tables_init_, comm_states_[c]->word_counts_);


    if(comm_states_[c]->num_tables_init_ > 0)
    {
      fwrite(comm_states_[c]->alive_b_, sizeof(bool), comm_states_[c]->num_tables_init_, file);
      fwrite(comm_states_[c]->decay_kernel_zeta_by_r_, sizeof(double), size_participants_, file);
      for(int b = 0; b < comm_states_[c]->num_tables_init_; b++)
      {
        if(!comm_states_[c]->alive_b_[b])  continue;
        fwrite(&(comm_states_[c]->decay_kernel_word_counts_by_b_[b]), sizeof(double), 1, file);
      }
    }
    for(int b = 0; b < comm_states_[c]->num_tables_; b++)
    {
      if(b < comm_states_[c]->num_tables_init_ && !comm_states_[c]->alive_b_[b])        continue;
      fwrite(&(comm_states_[c]->tables_to_topics_[b]), sizeof(int), 1, file);
      fwrite(&(comm_states_[c]->word_counts_by_b_[b]), sizeof(int), 1, file);
    }
  }

  fclose(file);
  fclose(f);
}

void EpochState::LoadModel(const char* directory, int epoch)
{
  char name[500];
  sprintf(name, "%s/e%d_model", directory, epoch);
  FILE * file = fopen(name, "rb");

  sprintf(name, "%s/load%d", directory, epoch);
  FILE * f = fopen(name, "w");

  fread(&num_topics_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_topics_);
  fread(&num_comms_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_comms_);
  fread(&num_topics_init_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_topics_init_);
  fread(&num_comms_init_, sizeof(int), 1, file);
  fprintf(f, "%d ", num_comms_init_);
  if(num_topics_init_ > 0)
  {
    alive_z_ = new bool [num_topics_init_];
    fread(alive_z_, sizeof(bool), num_topics_init_, file);
  }
  if(num_comms_init_ > 0)
  {
    alive_c_ = new bool [num_comms_init_];
    fread(alive_c_, sizeof(bool), num_comms_init_, file);
  }

  fread(&total_num_tables_, sizeof(int), 1, file);
  fprintf(f, "%d ", total_num_tables_);
  fread(&decay_kernel_num_tables_, sizeof(double), 1, file);
  fprintf(f, "%f ", decay_kernel_num_tables_);
  fread(&decay_kernel_num_docs_, sizeof(double), 1, file);
  fprintf(f, "%f ", decay_kernel_num_docs_);

  num_tables_by_z_.resize(num_topics_, 0);
  word_counts_by_z_.resize(num_topics_, 0);
  word_counts_by_zw_.resize(num_topics_, NULL);
  decay_kernel_num_tables_by_z_.resize(num_topics_init_, 0.0);
  decay_kernel_tau_by_z_.resize(num_topics_init_, 0.0);
  decay_kernel_tau_by_zw_.resize(num_topics_init_, NULL);
  for(int k = 0; k < num_topics_; k++)
  {
    if(k < num_topics_init_ && !alive_z_[k])    continue;
    fread(&(num_tables_by_z_[k]), sizeof(int), 1, file);
    fread(&(word_counts_by_z_[k]), sizeof(int), 1, file);
    word_counts_by_zw_[k] = new int [size_vocab_];
    fread(word_counts_by_zw_[k], sizeof(int), size_vocab_, file);
    if(num_topics_init_ > 0 && k < num_topics_init_)
    {
      fread(&(decay_kernel_num_tables_by_z_[k]), sizeof(double), 1, file);
      fread(&(decay_kernel_tau_by_z_[k]), sizeof(double), 1, file);
      decay_kernel_tau_by_zw_[k] = new double [size_vocab_];
      fread(decay_kernel_tau_by_zw_[k], sizeof(double), size_vocab_, file);
    }
  }

  comm_states_.resize(num_comms_, NULL);
  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])     continue;
    comm_states_[c] = new CommState();
    fread(&(comm_states_[c]->num_docs_), sizeof(int), 1, file);
    fread(&(comm_states_[c]->num_tables_), sizeof(int), 1, file);
    fread(&(comm_states_[c]->num_tables_init_), sizeof(int), 1, file);
    fread(&(comm_states_[c]->word_counts_), sizeof(int), 1, file);
    fread(&(comm_states_[c]->parti_counts_), sizeof(int), 1, file);
    fread(&(comm_states_[c]->decay_kernel_word_counts_), sizeof(double), 1, file);
    fread(&(comm_states_[c]->decay_kernel_num_docs_), sizeof(double), 1, file);
    fread(&(comm_states_[c]->decay_kernel_zeta_), sizeof(double), 1, file);
    comm_states_[c]->parti_counts_by_r_ = new int [size_participants_];
    fread(comm_states_[c]->parti_counts_by_r_, sizeof(int), size_participants_, file);
    comm_states_[c]->tables_to_topics_.resize(comm_states_[c]->num_tables_, -1);
    comm_states_[c]->word_counts_by_b_.resize(comm_states_[c]->num_tables_, 0);

    fprintf(f, "%d %d %d %d ", comm_states_[c]->num_docs_, comm_states_[c]->num_tables_, comm_states_[c]->num_tables_init_, comm_states_[c]->word_counts_);

    if(comm_states_[c]->num_tables_init_ > 0)
    {
      comm_states_[c]->alive_b_ = new bool [comm_states_[c]->num_tables_init_];
      fread(comm_states_[c]->alive_b_, sizeof(bool), comm_states_[c]->num_tables_init_, file);
      comm_states_[c]->decay_kernel_zeta_by_r_ = new double [size_participants_];
      fread(comm_states_[c]->decay_kernel_zeta_by_r_, sizeof(double), size_participants_, file);
      comm_states_[c]->decay_kernel_word_counts_by_b_.resize(comm_states_[c]->num_tables_init_, 0.0);
      for(int b = 0; b < comm_states_[c]->num_tables_init_; b++)
      {
        if(!comm_states_[c]->alive_b_[b])  continue;
        fread(&(comm_states_[c]->decay_kernel_word_counts_by_b_[b]), sizeof(double), 1, file);
      }
    }
    for(int b = 0; b < comm_states_[c]->num_tables_; b++)
    {
      if(b < comm_states_[c]->num_tables_init_ && !comm_states_[c]->alive_b_[b])
      {
        comm_states_[c]->tables_to_topics_[b] = -1;
        comm_states_[c]->word_counts_by_b_[b] = 0;
        continue;
      }
      fread(&(comm_states_[c]->tables_to_topics_[b]), sizeof(int), 1, file);
      fread(&(comm_states_[c]->word_counts_by_b_[b]), sizeof(int), 1, file);
    }
  }

  fclose(file);
  fclose(f);
}


bool EpochState::StateCheckSum()
{
  bool status_OK = true;
  int sum = 0;
  for(int k = 0; k < num_topics_; k ++)
  {
    if(k < num_topics_init_ && !alive_z_[k])    continue;
    sum += word_counts_by_z_[k];
  }
  if(sum != total_tokens_)
  {
    printf("\ntotal words does not match\n");
    status_OK = false;
  }
  for(int k = 0; k < num_topics_; k ++)
  {
    if(k < num_topics_init_ && !alive_z_[k])    continue;
    sum = 0;
    for(int w = 0; w < size_vocab_; w ++)
    {
      sum += word_counts_by_zw_[k][w];
    }
    if(sum != word_counts_by_z_[k])
    {
      printf("\nin topic %d, total words does not match\n", k);
      status_OK = false;
    }
  }
  sum = 0;
  for(int k = 0; k < num_topics_; k ++)
  {
    if(k < num_topics_init_ && !alive_z_[k])    continue;
    sum += num_tables_by_z_[k];
  }
  if(sum != total_num_tables_)
  {
    printf("\ntotal num tables does not match\n");
    status_OK = false;
  }

  sum = 0;
  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])  continue;
    CommState* c_state = comm_states_[c];
    sum += c_state->num_docs_;
  }
  if(sum != num_docs_)
  {
    printf("\ntotal num docs does not match\n");
    status_OK = false;
  }
  for(int c = 0; c < num_comms_; c++)
  {
    sum = 0;
    if(c < num_comms_init_ && !alive_c_[c])  continue;
    CommState* c_state = comm_states_[c];
    for(int b = 0; b < c_state->num_tables_; b++)
    {
      if(b < c_state->num_tables_init_ && !c_state->alive_b_[b])  continue;
      sum += c_state->word_counts_by_b_[b];
    }
    if(sum != c_state->word_counts_)
    {
      printf("\nin comm %d, total words does not match\n", c);
      status_OK = false;
    }
  }


  double sum_kernel = 0.0;
  for(int k = 0; k < num_topics_init_; k ++)
  {
    if(k < num_topics_init_ && !alive_z_[k])    continue;
    sum_kernel += decay_kernel_num_tables_by_z_[k];
  }
  if(sum_kernel != decay_kernel_num_tables_)
  {
    printf("\ndecay_kernel_num_tables_ does not match\n");
    status_OK = false;
  }

  sum_kernel = 0;
  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])  continue;
    CommState* c_state = comm_states_[c];
    sum_kernel += c_state->decay_kernel_num_docs_;
  }
  if(sum_kernel != decay_kernel_num_docs_)
  {
    printf("\ndecay_kernel_num_docs_ does not match\n");
    status_OK = false;
  }

  return status_OK;
}

void EpochState::EpochDebug()
{
  assert(StateCheckSum());

  printf("********************************************************\n");
  for(int k = 0; k < num_topics_; k++)
    printf("word counts by z %d = %d\n", k, word_counts_by_z_[k]);
  for(int k = 0; k < num_topics_; k++)
    printf("num tables by z %d = %d\n", k, num_tables_by_z_[k]);
  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])     continue;
    printf("num docs by c %d = %d\n", c, comm_states_[c]->num_docs_);
  }
  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])     continue;
    printf("participants counts by c %d = %d\n", c, comm_states_[c]->parti_counts_);
  }

/*  int* word_counts_by_z = new int [num_topics_];
  for(int c = 0; c < num_comms_; c++)
  {
    memset(word_counts_by_z, 0, sizeof(int)*num_topics_);
    CommState* c_state = comm_states_[c];
    for(int b = 0; b < c_state->num_tables_; b++)
    {
      word_counts_by_z[c_state->tables_to_topics_[b]] += c_state->word_counts_by_b_[b];
    }
    printf("comm %d:\n", c);
    for(int k = 0; k < num_topics_; k++)
    {
      printf("    word counts by z %d = %d\n", k, word_counts_by_z[k]);
    }
  }
  delete [] word_counts_by_z;*/

  for(int c = 0; c < num_comms_; c++)
  {
    if(c < num_comms_init_ && !alive_c_[c])     continue;
    CommState* c_state = comm_states_[c];
    printf("comm %d:\n", c);
    for(int b = 0; b < c_state->num_tables_; b++)
    {
      if(b < c_state->num_tables_init_ && !c_state->alive_b_[b])        continue;
      printf("    word counts by b %d = %d  (z %d)\n", b, c_state->word_counts_by_b_[b], c_state->tables_to_topics_[b]);
    }
  }
}


DtcState::DtcState(int size_vocab, int total_tokens, int num_epoches)
{
  size_vocab_ = size_vocab;
  total_tokens_ = total_tokens;
  num_epoches_ = num_epoches;
  epoch_states_ = new EpochState* [num_epoches];
  memset(epoch_states_, NULL, sizeof(EpochState*)*num_epoches);
}

DtcState::~DtcState()
{
  for(int t = 0; t < num_epoches_; t++)
    delete epoch_states_[t];
  delete [] epoch_states_;
  epoch_states_ = NULL;
}

/***
 * @brief: The data in each epoch is exchangeable, but they are not exchangeable between epoches.
 * So, only re-shuffle the data in the same epoch.
 */
void DtcState::Reshuffle()
{
  for(int t = 0; t < num_epoches_; t++)
    epoch_states_[t]->ReshuffleEpoch();
}

double DtcState::ComputeKernel(double lambda, int delta, int v1, double v2, int v3)
{
  return exp(-1/lambda) * (v1 + v2) - exp(-(delta+1)/lambda) * v3;
}


void DtcState::InitNextEpochInfer(int epoch, double lambda, double lambda_con, int delta)
{

  int k, c, b, w, r;
  EpochState* e = epoch_states_[epoch];

  if(epoch > 0)
  {
    e->num_comms_init_ = epoch_states_[epoch-1]->num_comms_;
    e->num_topics_init_ = epoch_states_[epoch-1]->num_topics_;
    e->num_comms_ = e->num_comms_init_;
    e->num_topics_ = e->num_topics_init_;
  }

  //allocate necessary memory and initialize them first.
  if(e->num_comms_init_ > 0)
    e->alive_c_ = new bool[e->num_comms_init_];
  if(e->num_topics_init_ > 0)
    e->alive_z_ = new bool[e->num_topics_init_];

  bool z_init = e->num_topics_ < INIT_Z_SIZE;
  bool c_init = e->num_comms_ < INIT_C_SIZE;
  e->num_tables_by_z_.resize(z_init?INIT_Z_SIZE:(e->num_topics_init_ + 1), 0);
  e->word_counts_by_z_.resize(z_init?INIT_Z_SIZE:(e->num_topics_init_ + 1), 0);
  e->word_counts_by_zw_.resize(z_init?INIT_Z_SIZE:(e->num_topics_init_ + 1), NULL);

  e->decay_kernel_num_tables_by_z_.resize(e->num_topics_init_, 0.0);
  e->decay_kernel_tau_by_z_.resize(e->num_topics_init_, 0.0);
  e->decay_kernel_tau_by_zw_.resize(e->num_topics_init_, NULL);

  e->comm_states_.resize(c_init?INIT_C_SIZE:(e->num_comms_init_ + 1), NULL);

  if(epoch == 0)
  {
    for(k = 0; k < (int)e->word_counts_by_zw_.size(); k++)
    {
      e->word_counts_by_zw_[k] = new int[size_vocab_];
      memset(e->word_counts_by_zw_[k], 0, sizeof(int)*size_vocab_);
    }
    for(c = 0; c < (int)e->comm_states_.size(); c++)
      e->comm_states_[c] = new CommState(INIT_B_SIZE, e->size_participants_);
  }
  else
  {
    //compute alive_z
    memcpy(e->alive_z_, epoch_states_[epoch-1]->alive_z_, sizeof(bool)*(epoch_states_[epoch-1]->num_topics_init_));
    memset(e->alive_z_ + epoch_states_[epoch-1]->num_topics_init_, true, sizeof(bool)*(e->num_topics_init_ - epoch_states_[epoch-1]->num_topics_init_));
//    for(k = 0; k < epoch_states_[epoch-1]->num_topics_init_; k++)
//    {
//      if(e->alive_z_[k] && epoch > delta && epoch_states_[epoch-delta-1]->num_topics_ > k)
//        if(ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k],
//            epoch_states_[epoch-1]->decay_kernel_tau_by_z_[k], epoch_states_[epoch-delta-1]->word_counts_by_z_[k]) < EPSILON)
//          e->alive_z_[k] = false;
//    }
    //initialize word_counts_by_zw_
    for(k = 0; k < (int)e->word_counts_by_zw_.size(); k++)
      if(k >= e->num_topics_init_ || e->alive_z_[k])
      {
        e->word_counts_by_zw_[k] = new int[size_vocab_];
        memset(e->word_counts_by_zw_[k], 0, sizeof(int)*size_vocab_);
      }
    //compute decay_kernel related to z.
    double total_kernel = 0.0;
    for(k = 0; k < e->num_topics_init_; k++)
    {
      if(e->alive_z_[k])
      {
//        if(epoch > delta && epoch_states_[epoch-delta-1]->num_topics_ > k)
//        {
//          e->decay_kernel_tau_by_z_[k] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k],
//              epoch_states_[epoch-1]->decay_kernel_tau_by_z_[k], epoch_states_[epoch-delta-1]->word_counts_by_z_[k]);
//          e->decay_kernel_tau_by_zw_[k] = new double [size_vocab_];
//          for(w = 0; w < size_vocab_; w++)
//            e->decay_kernel_tau_by_zw_[k][w] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_zw_[k][w],
//                epoch_states_[epoch-1]->decay_kernel_tau_by_zw_[k][w], epoch_states_[epoch-delta-1]->word_counts_by_zw_[k][w]);
//          e->decay_kernel_num_tables_by_z_[k] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->num_tables_by_z_[k],
//              epoch_states_[epoch-1]->decay_kernel_num_tables_by_z_[k], epoch_states_[epoch-delta-1]->num_tables_by_z_[k]);
//        }
        if(epoch_states_[epoch-1]->num_topics_init_ > k)
        {
          e->decay_kernel_tau_by_z_[k] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k],
              epoch_states_[epoch-1]->decay_kernel_tau_by_z_[k]);
          e->decay_kernel_tau_by_zw_[k] = new double [size_vocab_];
          for(w = 0; w < size_vocab_; w++)
            e->decay_kernel_tau_by_zw_[k][w] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_zw_[k][w],
                epoch_states_[epoch-1]->decay_kernel_tau_by_zw_[k][w]);
          e->decay_kernel_num_tables_by_z_[k] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->num_tables_by_z_[k],
              epoch_states_[epoch-1]->decay_kernel_num_tables_by_z_[k]);
        }
        else
        {
          e->decay_kernel_tau_by_z_[k] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k]);
          e->decay_kernel_tau_by_zw_[k] = new double [size_vocab_];
          for(w = 0; w < size_vocab_; w++)
            e->decay_kernel_tau_by_zw_[k][w] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_zw_[k][w]);
          e->decay_kernel_num_tables_by_z_[k] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->num_tables_by_z_[k]);
        }
        total_kernel += e->decay_kernel_num_tables_by_z_[k];
      }
    }
    e->decay_kernel_num_tables_ = total_kernel;

    //compute alive_c, then alive_b in each alive c
    memcpy(e->alive_c_, epoch_states_[epoch-1]->alive_c_, sizeof(bool)*(epoch_states_[epoch-1]->num_comms_init_));
    memset(e->alive_c_ + epoch_states_[epoch-1]->num_comms_init_, true, sizeof(bool)*(e->num_comms_init_ - epoch_states_[epoch-1]->num_comms_init_));
//    for(c = 0; c < epoch_states_[epoch-1]->num_comms_init_; c++)
//    {
//      if(e->alive_c_[c] && epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c)
//        if(ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_, epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_num_docs_,
//            epoch_states_[epoch-delta-1]->comm_states_[c]->num_docs_) < EPSILON)
//          e->alive_c_[c] = false;
//    }
    //compute kernel related to community.
    total_kernel = 0.0;
    for(c = 0; c < (int)e->comm_states_.size(); c++)
    {
      if(c < e->num_comms_init_ && e->alive_c_[c])
      {
        e->comm_states_[c] = new CommState(epoch_states_[epoch-1]->comm_states_[c], e->size_participants_);
//        if(epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c)
//        {
//          e->comm_states_[c]->decay_kernel_num_docs_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_,
//              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_num_docs_, epoch_states_[epoch-delta-1]->comm_states_[c]->num_docs_);
//          e->comm_states_[c]->decay_kernel_word_counts_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_,
//              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_, epoch_states_[epoch-delta-1]->comm_states_[c]->word_counts_);
//          e->comm_states_[c]->decay_kernel_zeta_ = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_,
//              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_, epoch_states_[epoch-delta-1]->comm_states_[c]->parti_counts_);
//          for(r = 0; r < e->size_participants_; r++)
//            e->comm_states_[c]->decay_kernel_zeta_by_r_[r] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_by_r_[r],
//                epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_by_r_[r], epoch_states_[epoch-delta-1]->comm_states_[c]->parti_counts_by_r_[r]);
//        }
        if(epoch_states_[epoch-1]->num_comms_init_ > c)
        {
          e->comm_states_[c]->decay_kernel_num_docs_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_num_docs_);
          e->comm_states_[c]->decay_kernel_word_counts_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_);
          e->comm_states_[c]->decay_kernel_zeta_ = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_);
          for(r = 0; r < e->size_participants_; r++)
            e->comm_states_[c]->decay_kernel_zeta_by_r_[r] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_by_r_[r],
                epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_by_r_[r]);
        }
        else    //a community burned in epoch-1
        {
          e->comm_states_[c]->decay_kernel_num_docs_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_);
          e->comm_states_[c]->decay_kernel_word_counts_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_);
          e->comm_states_[c]->decay_kernel_zeta_ = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_);
          for(r = 0; r < e->size_participants_; r++)
            e->comm_states_[c]->decay_kernel_zeta_by_r_[r] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_by_r_[r]);
        }
        total_kernel += e->comm_states_[c]->decay_kernel_num_docs_;

        //compute alive_b
        memcpy(e->comm_states_[c]->alive_b_, epoch_states_[epoch-1]->comm_states_[c]->alive_b_, sizeof(bool)*(epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_));
        memset(e->comm_states_[c]->alive_b_ + epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_, true, sizeof(bool)*(e->comm_states_[c]->num_tables_init_ - epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_));
//        for(b = 0; b < epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_; b++)
//        {
//          if(e->comm_states_[c]->alive_b_[b] && epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c && epoch_states_[epoch-delta-1]->comm_states_[c]->num_tables_ > b)
//            if(ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b],
//                epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_by_b_[b], epoch_states_[epoch-delta-1]->comm_states_[c]->word_counts_by_b_[b]) < EPSILON)
//              e->comm_states_[c]->alive_b_[b] = false;
//        }
        //compute kernel related to table
        for(b = 0; b < e->comm_states_[c]->num_tables_init_; b++)
        {
          if(e->comm_states_[c]->alive_b_[b])
          {
//            if(epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c && epoch_states_[epoch-delta-1]->comm_states_[c]->num_tables_ > b)
//              e->comm_states_[c]->decay_kernel_word_counts_by_b_[b] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b],
//                  epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_by_b_[b], epoch_states_[epoch-delta-1]->comm_states_[c]->word_counts_by_b_[b]);
            if(b < epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_)
              e->comm_states_[c]->decay_kernel_word_counts_by_b_[b] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b],
                  epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_by_b_[b]);
            else
              e->comm_states_[c]->decay_kernel_word_counts_by_b_[b] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b]);
          }
        }
      }
      if(c >= e->num_comms_init_)
        e->comm_states_[c] = new CommState(INIT_B_SIZE, e->size_participants_);
    }
    e->decay_kernel_num_docs_ = total_kernel;
  }

}

/***
 * @brief: Build next epoch's states for sampling.
 * will not allocate memory for dead items.
 */
void DtcState::InitNextEpoch(int epoch, double lambda, double lambda_con, int delta)
{
  int k, c, b, w, r;
  EpochState* e = epoch_states_[epoch];

  if(epoch > 0)
  {
    e->num_comms_init_ = epoch_states_[epoch-1]->num_comms_;
    e->num_topics_init_ = epoch_states_[epoch-1]->num_topics_;
    e->num_comms_ = e->num_comms_init_;
    e->num_topics_ = e->num_topics_init_;
  }

  //allocate necessary memory and initialize them first.
  if(e->num_comms_init_ > 0)
    e->alive_c_ = new bool[e->num_comms_init_];
  if(e->num_topics_init_ > 0)
    e->alive_z_ = new bool[e->num_topics_init_];

  bool z_init = e->num_topics_ < INIT_Z_SIZE;
  bool c_init = e->num_comms_ < INIT_C_SIZE;
  e->num_tables_by_z_.resize(z_init?INIT_Z_SIZE:(e->num_topics_init_ + 1), 0);
  e->word_counts_by_z_.resize(z_init?INIT_Z_SIZE:(e->num_topics_init_ + 1), 0);
  e->word_counts_by_zw_.resize(z_init?INIT_Z_SIZE:(e->num_topics_init_ + 1), NULL);

  e->decay_kernel_num_tables_by_z_.resize(e->num_topics_init_, 0.0);
  e->decay_kernel_tau_by_z_.resize(e->num_topics_init_, 0.0);
  e->decay_kernel_tau_by_zw_.resize(e->num_topics_init_, NULL);

  e->comm_states_.resize(c_init?INIT_C_SIZE:(e->num_comms_init_ + 1), NULL);

  if(epoch == 0)
  {
    for(k = 0; k < (int)e->word_counts_by_zw_.size(); k++)
    {
      e->word_counts_by_zw_[k] = new int[size_vocab_];
      memset(e->word_counts_by_zw_[k], 0, sizeof(int)*size_vocab_);
    }
    for(c = 0; c < (int)e->comm_states_.size(); c++)
      e->comm_states_[c] = new CommState(INIT_B_SIZE, e->size_participants_);
  }
  else
  {
    //compute alive_z
    memcpy(e->alive_z_, epoch_states_[epoch-1]->alive_z_, sizeof(bool)*(epoch_states_[epoch-1]->num_topics_init_));
    memset(e->alive_z_ + epoch_states_[epoch-1]->num_topics_init_, true, sizeof(bool)*(e->num_topics_init_ - epoch_states_[epoch-1]->num_topics_init_));
    for(k = 0; k < epoch_states_[epoch-1]->num_topics_init_; k++)
    {
      if(e->alive_z_[k] && epoch > delta && epoch_states_[epoch-delta-1]->num_topics_ > k)
        if(ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k],
            epoch_states_[epoch-1]->decay_kernel_tau_by_z_[k], epoch_states_[epoch-delta-1]->word_counts_by_z_[k]) < EPSILON)
          e->alive_z_[k] = false;
    }
    //initialize word_counts_by_zw_
    for(k = 0; k < (int)e->word_counts_by_zw_.size(); k++)
      if(k >= e->num_topics_init_ || e->alive_z_[k])
      {
        e->word_counts_by_zw_[k] = new int[size_vocab_];
        memset(e->word_counts_by_zw_[k], 0, sizeof(int)*size_vocab_);
      }
    //compute decay_kernel related to z.
    double total_kernel = 0.0;
    for(k = 0; k < e->num_topics_init_; k++)
    {
      if(e->alive_z_[k])
      {
        if(epoch > delta && epoch_states_[epoch-delta-1]->num_topics_ > k)
        {
          e->decay_kernel_tau_by_z_[k] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k],
              epoch_states_[epoch-1]->decay_kernel_tau_by_z_[k], epoch_states_[epoch-delta-1]->word_counts_by_z_[k]);
          e->decay_kernel_tau_by_zw_[k] = new double [size_vocab_];
          for(w = 0; w < size_vocab_; w++)
            e->decay_kernel_tau_by_zw_[k][w] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_zw_[k][w],
                epoch_states_[epoch-1]->decay_kernel_tau_by_zw_[k][w], epoch_states_[epoch-delta-1]->word_counts_by_zw_[k][w]);
          e->decay_kernel_num_tables_by_z_[k] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->num_tables_by_z_[k],
              epoch_states_[epoch-1]->decay_kernel_num_tables_by_z_[k], epoch_states_[epoch-delta-1]->num_tables_by_z_[k]);
        }
        else if(epoch_states_[epoch-1]->num_topics_init_ > k)
        {
          e->decay_kernel_tau_by_z_[k] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k],
              epoch_states_[epoch-1]->decay_kernel_tau_by_z_[k]);
          e->decay_kernel_tau_by_zw_[k] = new double [size_vocab_];
          for(w = 0; w < size_vocab_; w++)
            e->decay_kernel_tau_by_zw_[k][w] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_zw_[k][w],
                epoch_states_[epoch-1]->decay_kernel_tau_by_zw_[k][w]);
          e->decay_kernel_num_tables_by_z_[k] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->num_tables_by_z_[k],
              epoch_states_[epoch-1]->decay_kernel_num_tables_by_z_[k]);
        }
        else
        {
          e->decay_kernel_tau_by_z_[k] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_z_[k]);
          e->decay_kernel_tau_by_zw_[k] = new double [size_vocab_];
          for(w = 0; w < size_vocab_; w++)
            e->decay_kernel_tau_by_zw_[k][w] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->word_counts_by_zw_[k][w]);
          e->decay_kernel_num_tables_by_z_[k] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->num_tables_by_z_[k]);
        }
        total_kernel += e->decay_kernel_num_tables_by_z_[k];
      }
    }
    e->decay_kernel_num_tables_ = total_kernel;

    //compute alive_c, then alive_b in each alive c
    memcpy(e->alive_c_, epoch_states_[epoch-1]->alive_c_, sizeof(bool)*(epoch_states_[epoch-1]->num_comms_init_));
    memset(e->alive_c_ + epoch_states_[epoch-1]->num_comms_init_, true, sizeof(bool)*(e->num_comms_init_ - epoch_states_[epoch-1]->num_comms_init_));
    for(c = 0; c < epoch_states_[epoch-1]->num_comms_init_; c++)
    {
      if(e->alive_c_[c] && epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c)
        if(ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_, epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_num_docs_,
            epoch_states_[epoch-delta-1]->comm_states_[c]->num_docs_) < EPSILON)
          e->alive_c_[c] = false;
    }
    //compute kernel related to community.
    total_kernel = 0.0;
    for(c = 0; c < (int)e->comm_states_.size(); c++)
    {
      if(c < e->num_comms_init_ && e->alive_c_[c])
      {
        e->comm_states_[c] = new CommState(epoch_states_[epoch-1]->comm_states_[c], e->size_participants_);
        if(epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c)
        {
          e->comm_states_[c]->decay_kernel_num_docs_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_num_docs_, epoch_states_[epoch-delta-1]->comm_states_[c]->num_docs_);
          e->comm_states_[c]->decay_kernel_word_counts_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_, epoch_states_[epoch-delta-1]->comm_states_[c]->word_counts_);
          e->comm_states_[c]->decay_kernel_zeta_ = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_, epoch_states_[epoch-delta-1]->comm_states_[c]->parti_counts_);
          for(r = 0; r < e->size_participants_; r++)
            e->comm_states_[c]->decay_kernel_zeta_by_r_[r] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_by_r_[r],
                epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_by_r_[r], epoch_states_[epoch-delta-1]->comm_states_[c]->parti_counts_by_r_[r]);
        }
        else if(epoch_states_[epoch-1]->num_comms_init_ > c)
        {
          e->comm_states_[c]->decay_kernel_num_docs_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_num_docs_);
          e->comm_states_[c]->decay_kernel_word_counts_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_);
          e->comm_states_[c]->decay_kernel_zeta_ = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_,
              epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_);
          for(r = 0; r < e->size_participants_; r++)
            e->comm_states_[c]->decay_kernel_zeta_by_r_[r] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_by_r_[r],
                epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_zeta_by_r_[r]);
        }
        else    //a community burned in epoch-1
        {
          e->comm_states_[c]->decay_kernel_num_docs_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->num_docs_);
          e->comm_states_[c]->decay_kernel_word_counts_ = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_);
          e->comm_states_[c]->decay_kernel_zeta_ = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_);
          for(r = 0; r < e->size_participants_; r++)
            e->comm_states_[c]->decay_kernel_zeta_by_r_[r] = ComputeKernel(lambda_con, delta, epoch_states_[epoch-1]->comm_states_[c]->parti_counts_by_r_[r]);
        }
        total_kernel += e->comm_states_[c]->decay_kernel_num_docs_;

        //compute alive_b
        memcpy(e->comm_states_[c]->alive_b_, epoch_states_[epoch-1]->comm_states_[c]->alive_b_, sizeof(bool)*(epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_));
        memset(e->comm_states_[c]->alive_b_ + epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_, true, sizeof(bool)*(e->comm_states_[c]->num_tables_init_ - epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_));
        for(b = 0; b < epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_; b++)
        {
          if(e->comm_states_[c]->alive_b_[b] && epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c && epoch_states_[epoch-delta-1]->comm_states_[c]->num_tables_ > b)
            if(ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b],
                epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_by_b_[b], epoch_states_[epoch-delta-1]->comm_states_[c]->word_counts_by_b_[b]) < EPSILON)
              e->comm_states_[c]->alive_b_[b] = false;
        }
        //compute kernel related to table
        for(b = 0; b < e->comm_states_[c]->num_tables_init_; b++)
        {
          if(e->comm_states_[c]->alive_b_[b])
          {
            if(epoch > delta && epoch_states_[epoch-delta-1]->num_comms_ > c && epoch_states_[epoch-delta-1]->comm_states_[c]->num_tables_ > b)
              e->comm_states_[c]->decay_kernel_word_counts_by_b_[b] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b],
                  epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_by_b_[b], epoch_states_[epoch-delta-1]->comm_states_[c]->word_counts_by_b_[b]);
            else if(b < epoch_states_[epoch-1]->comm_states_[c]->num_tables_init_)
              e->comm_states_[c]->decay_kernel_word_counts_by_b_[b] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b],
                  epoch_states_[epoch-1]->comm_states_[c]->decay_kernel_word_counts_by_b_[b]);
            else
              e->comm_states_[c]->decay_kernel_word_counts_by_b_[b] = ComputeKernel(lambda, delta, epoch_states_[epoch-1]->comm_states_[c]->word_counts_by_b_[b]);
          }
        }
      }
      if(c >= e->num_comms_init_)
        e->comm_states_[c] = new CommState(INIT_B_SIZE, e->size_participants_);
    }
    e->decay_kernel_num_docs_ = total_kernel;
  }
}

/*
void DtcState::NextGibbsSweep(bool permute, DtcHyperPara* hyper)
{
  if(permute)
    Reshuffle();

  int t, j, i, c, b;
  vector<double> q_k, q_b, f, f_c;
  EpochState* epoch_state;
  CommState* c_state;
  DocState* doc_state;

  for(t = 0; t < num_epoches_; t++)
  {
    if(t > 0)
      InitNextEpoch(t);

    epoch_state = epoch_states_[t];

    //Sample a table assignment for each word in current epoch.
    for(j = 0; j < epoch_state->num_docs_; j++)
    {
      doc_state = epoch_state->doc_states_[j];
      for(i = 0; i < doc_state->len_doc_; i++)
        epoch_state->SampleWordAssignment(doc_state, i, q_k, q_b, f, hyper, this);
    }

    //Sample a topic assignment for each table in each community in current epoch.
    epoch_state->SampleTables(q_k, f, hyper, this);

    //Sample a community indicator for each document in current epoch.
    epoch_state->SampleCommunities(q_b, f, f_c, q_k, hyper, this->size_vocab_);

    epoch_state->CompactEpochStates();
  }
}*/
