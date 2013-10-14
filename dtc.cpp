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

void Dtc::OnlineEstimate(const char* directory, int epoch_start)
{
  time_t start, current;
  double diff;
  time(&start);

  bool permute;

  char filename_state[500];
  sprintf(filename_state, "%s/state.log", directory);
  FILE* f_state = fopen(filename_state, "w");

  for(int epoch = epoch_start; epoch < dtc_state_->num_epoches_; epoch++)
  {
    EpochState* e_state = dtc_state_->epoch_states_[epoch];

    for(int e = 0; e < epoch; e++)
    {
      delete dtc_state_->epoch_states_[e];
      dtc_state_->epoch_states_[e] = new EpochState(e, e_state->size_vocab_, e_state->size_participants_);
      dtc_state_->epoch_states_[e]->LoadModel(directory, e);
    }

    dtc_state_->InitNextEpoch(epoch, dtc_hyper_para_->lambda_, dtc_hyper_para_->lambda_con_, dtc_hyper_para_->delta_);
    printf("Epoch %d inited:\nnum_docs = %d\ntotal_tokens = %d\ninit_comms = %d\ninit_topics = %d\n",
        epoch, dtc_state_->epoch_states_[epoch]->num_docs_, dtc_state_->epoch_states_[epoch]->total_tokens_,
        dtc_state_->epoch_states_[epoch]->num_comms_init_, dtc_state_->epoch_states_[epoch]->num_topics_init_);
    fprintf(f_state, "Epoch %d inited:\nnum_docs = %d\ntotal_tokens = %d\ninit_comms = %d\ninit_topics = %d\n",
            epoch, dtc_state_->epoch_states_[epoch]->num_docs_, dtc_state_->epoch_states_[epoch]->total_tokens_,
            dtc_state_->epoch_states_[epoch]->num_comms_init_, dtc_state_->epoch_states_[epoch]->num_topics_init_);


    e_state->InitEpochGibbsState(dtc_hyper_para_);
    printf("num_comms = %d\nnum_topics = %d\n\n", e_state->num_comms_, e_state->num_topics_);
    fprintf(f_state, "num_comms = %d\nnum_topics = %d\n\n", e_state->num_comms_, e_state->num_topics_);


    for(int i = 0; i < dtc_state_->epoch_states_[epoch]->num_comms_init_; i++)
      if(!dtc_state_->epoch_states_[epoch]->alive_c_[i])
      {
        printf("#comm = %d is dead\n", i);
        fprintf(f_state, "#comm = %d is dead\n", i);
      }
    for(int i = 0; i < dtc_state_->epoch_states_[epoch]->num_topics_init_; i++)
      if(!dtc_state_->epoch_states_[epoch]->alive_z_[i])
      {
        printf("#topic = %d is dead\n", i);
        fprintf(f_state, "#topic = %d is dead\n", i);
      }


    double best_likelihood = -1e50;

    for(int iter = 1; iter <= dtc_hyper_para_->max_iter_; iter++)
    {
      if(PERMUTE && (iter % PERMUTE_LAG == 0) && (iter > 0))  permute = true;
      else permute = false;

      double accpt_ratio = e_state->NextEpochGibbsSweep(permute, dtc_hyper_para_);

      double likelihood = e_state->JointLikelihood(dtc_hyper_para_);

      if(best_likelihood < likelihood)
      {
        best_likelihood = likelihood;
        e_state->SaveState(directory);
        e_state->SaveModel(directory);
      }

      if(iter >= dtc_hyper_para_->burnin_ && iter % dtc_hyper_para_->save_lag_ == 0)
        e_state->SaveState(directory, iter);

      double perp = e_state->GetPerplexity(dtc_hyper_para_);

      time(&current);
      diff = difftime(current, start);

      printf("epoch = %02d, iter = %04d, #comms = %03d, #topics = %03d, #tables = %04d, lhood = %.5f, perp = %.5f, ar = %.5f\n",
          epoch, iter, e_state->num_comms_, e_state->num_topics_, e_state->total_num_tables_, likelihood, perp, accpt_ratio);
      fprintf(f_state, "epoch = %02d, iter = %04d, #comms = %03d, #topics = %03d, #tables = %04d, lhood = %.5f, perp = %.5f, ar = %.5f\n",
                epoch, iter, e_state->num_comms_, e_state->num_topics_, e_state->total_num_tables_, likelihood, perp, accpt_ratio);
      if(iter % 10 == 0)
        fflush(f_state);
    }
  }
  fclose(f_state);
}


void Dtc::Inference(const char* directory, int epoch_start)
{
  time_t start, current;
  double diff;
  time(&start);

  bool permute;

  char filename_state[500];
  sprintf(filename_state, "%s/state.log", directory);
  FILE* f_state = fopen(filename_state, "w");

  for(int epoch = epoch_start; epoch < dtc_state_->num_epoches_; epoch++)
  {
    EpochState* e_state = dtc_state_->epoch_states_[epoch];

    for(int e = 0; e < epoch; e++)
    {
      delete dtc_state_->epoch_states_[e];
      dtc_state_->epoch_states_[e] = new EpochState(e, e_state->size_vocab_, e_state->size_participants_);
      dtc_state_->epoch_states_[e]->LoadModel(directory, e);
    }

    dtc_state_->InitNextEpochInfer(epoch, dtc_hyper_para_->lambda_, dtc_hyper_para_->lambda_con_, dtc_hyper_para_->delta_);
    printf("Epoch %d inited:\nnum_docs = %d\ntotal_tokens = %d\ninit_comms = %d\ninit_topics = %d\n",
        epoch, dtc_state_->epoch_states_[epoch]->num_docs_, dtc_state_->epoch_states_[epoch]->total_tokens_,
        dtc_state_->epoch_states_[epoch]->num_comms_init_, dtc_state_->epoch_states_[epoch]->num_topics_init_);
    fprintf(f_state, "Epoch %d inited:\nnum_docs = %d\ntotal_tokens = %d\ninit_comms = %d\ninit_topics = %d\n",
        epoch, dtc_state_->epoch_states_[epoch]->num_docs_, dtc_state_->epoch_states_[epoch]->total_tokens_,
        dtc_state_->epoch_states_[epoch]->num_comms_init_, dtc_state_->epoch_states_[epoch]->num_topics_init_);


    e_state->InitEpochGibbsState(dtc_hyper_para_);
    printf("num_comms = %d\nnum_topics = %d\n\n", e_state->num_comms_, e_state->num_topics_);
    fprintf(f_state, "num_comms = %d\nnum_topics = %d\n\n", e_state->num_comms_, e_state->num_topics_);


    for(int i = 0; i < dtc_state_->epoch_states_[epoch]->num_comms_init_; i++)
      if(!dtc_state_->epoch_states_[epoch]->alive_c_[i])
      {
        printf("#comm = %d is dead\n", i);
        fprintf(f_state, "#comm = %d is dead\n", i);
      }
    for(int i = 0; i < dtc_state_->epoch_states_[epoch]->num_topics_init_; i++)
      if(!dtc_state_->epoch_states_[epoch]->alive_z_[i])
      {
        printf("#topic = %d is dead\n", i);
        fprintf(f_state, "#topic = %d is dead\n", i);
      }


    double best_likelihood = -1e50;

    for(int iter = 1; iter <= dtc_hyper_para_->max_iter_; iter++)
    {
      if(PERMUTE && (iter % PERMUTE_LAG == 0) && (iter > 0))  permute = true;
      else permute = false;

      double accpt_ratio = e_state->NextEpochGibbsSweep(permute, dtc_hyper_para_);

      double likelihood = e_state->JointLikelihood(dtc_hyper_para_);

      if(best_likelihood < likelihood)
      {
        best_likelihood = likelihood;
//        e_state->SaveState(directory);
//        e_state->SaveModel(directory);
      }

//      if(iter >= dtc_hyper_para_->burnin_ && iter % dtc_hyper_para_->save_lag_ == 0)
//        e_state->SaveState(directory, iter);

      double perp = e_state->GetPerplexity(dtc_hyper_para_);

      time(&current);
      diff = difftime(current, start);

      printf("epoch = %02d, iter = %04d, #comms = %03d, #topics = %03d, #tables = %04d, lhood = %.5f, perp = %.5f, ar = %.5f\n",
          epoch, iter, e_state->num_comms_, e_state->num_topics_, e_state->total_num_tables_, likelihood, perp, accpt_ratio);
      fprintf(f_state, "epoch = %02d, iter = %04d, #comms = %03d, #topics = %03d, #tables = %04d, lhood = %.5f, perp = %.5f, ar = %.5f\n",
          epoch, iter, e_state->num_comms_, e_state->num_topics_, e_state->total_num_tables_, likelihood, perp, accpt_ratio);
      if(iter % 10 == 0)
        fflush(f_state);
    }
  }
  fclose(f_state);
}

