/*
 * main.cpp
 *
 *  Created on: 2013-3-4
 *      Author: ziqilau
 */
#include "state.h"
#include "dtc.h"
#include "corpus.h"
#include "utils.h"

gsl_rng * g_random_number;

int main(int argc, char** argv)
{
  //Initialize the random seed using current time.
  time_t t;
  time(&t);
  long seed = static_cast<long>(t);
  g_random_number = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(g_random_number, seed);

  //Build the Corpus.
  char* data_path = argv[1];
  Corpus* c = new Corpus();
  c->ReadData(data_path);

  //Initialize the DTC's hyperparameter.
  DtcHyperPara* hyper_para = new DtcHyperPara();

  Dtc* dtc = new Dtc();
  dtc->SetupStateFromCorpus(hyper_para, c);
  dtc->OnlineEstimate();

  delete dtc;
  delete hyper_para;
  delete c;

  gsl_rng_free(g_random_number);

  return 0;
}
