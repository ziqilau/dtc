/*
 * dtc.cpp
 *
 *  Created on: 2013-3-5
 *      Author: ziqilau
 */

#include "state.h"
#include "dtc.h"
#include "utils.h"

#define PERMUTE true
#define PERMUTE_LAG 10

Dtc::Dtc()
{
  dtc_hyper_para_ = NULL;
  dtc_state_ = NULL;
}

Dtc::~Dtc()
{
  dtc_hyper_para_ = NULL;
  delete dtc_state_;
  dtc_state_ = NULL;
}

/***
 * Setup DtcState, initialize all things but comm_states_.
 * Also build states related to corpus information.
 */
void Dtc::SetupStateFromCorpus(DtcHyperPara* hyper_para, Corpus* c)
{
  dtc_hyper_para_ = hyper_para;
  dtc_state_ = new DtcState(c->size_vocab_, c->total_tokens_, c->num_epoches_);
  int t, j;
  for(t = 0; t < c->num_epoches_; t++)
    dtc_state_->epoch_states_[t] = new EpochState(c, t);
  int* i_by_e = new int[c->num_epoches_];
  memset(i_by_e, 0, sizeof(int)*c->num_epoches_);
  for(j = 0; j < c->num_docs_; j++)
  {
    int epo = c->docs_[j]->epoch_;
    EpochState* e_state = dtc_state_->epoch_states_[epo];
    e_state->doc_states_[i_by_e[epo]] = new DocState(c->docs_[j]);
    i_by_e[epo]++;
  }
  delete [] i_by_e;
}

void Dtc::BatchEstimate()
{
  //TODO: init states.
  //TODO: compute likelihood.

  bool permute = false;

  time_t start, current;
  double diff;
  time(&start);

  for(int iter = 0; iter < dtc_hyper_para_->max_iter_; iter++)
  {
    if(PERMUTE && (iter % PERMUTE_LAG == 0) && (iter > 0))  permute = true;
    else permute = false;

    //dtc_state_->NextGibbsSweep(permute, dtc_hyper_para_);
    //TODO: compute likelihood, trace the best likelihood.
    time(&current);
    diff = difftime(current, start);
    //TODO: print, and save model.
  }
}

void Dtc::OnlineEstimate()
{
  time_t start, current;
  double diff;
  time(&start);

  bool permute;

  for(int epoch = 0; epoch < dtc_state_->num_epoches_; epoch++)
  {
    dtc_state_->InitNextEpoch(epoch, dtc_hyper_para_->lambda_, dtc_hyper_para_->delta_);

    EpochState* e_state = dtc_state_->epoch_states_[epoch];
    e_state->InitEpochGibbsState(dtc_hyper_para_);

    double best_likelihood = e_state->JointLikelihood(dtc_hyper_para_);

    for(int iter = 0; iter < dtc_hyper_para_->max_iter_; iter++)
    {
      if(PERMUTE && (iter % PERMUTE_LAG == 0) && (iter > 0))  permute = true;
      else permute = false;

      e_state->NextEpochGibbsSweep(permute, dtc_hyper_para_);
      //TODO: compute likelihood, trace the best likelihood, and save the states for use of consecutive epoches.
      double likelihood = e_state->JointLikelihood(dtc_hyper_para_);
      if(best_likelihood < likelihood)
      {
        best_likelihood = likelihood;
      }
      time(&current);
      diff = difftime(current, start);

      printf("#Epoch = %2d, #Iter = %4d, #Communities = %2d, #Topics = %2d, #Tables = %3d, #Likelihood = %.5f\n",
          epoch, iter, e_state->num_comms_, e_state->num_topics_, e_state->total_num_tables_, likelihood);
    }
  }
}
