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

gsl_rng* g_random_number;

int main(int argc, char** argv)
{
  double alpha = 1.0, beta = 5.0, gamma = 2.0, tau = 0.01, zeta = 0.01, lambda = 1.5, lambda_con = 4.0;
  int max_iter = 2500, save_lag = 200, delta = 4, burnin = 1800, epoch_start = 0, epoch_num = 28, train = 1;
  bool sample_hyperparameter = false, metropolis = true;
  char* directory = NULL;
  char* data_path = NULL;

  for(int i = 1; i < argc; i++)
  {
    if(!strcmp(argv[i], "--data"))             data_path = argv[++i];
    else if(!strcmp(argv[i], "--max_iter"))    max_iter = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--save_lag"))    save_lag = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--directory"))   directory = argv[++i];
    else if(!strcmp(argv[i], "--alpha"))       alpha = atof(argv[++i]);
    else if(!strcmp(argv[i], "--beta"))        beta = atof(argv[++i]);
    else if(!strcmp(argv[i], "--gamma"))       gamma = atof(argv[++i]);
    else if(!strcmp(argv[i], "--tau"))         tau = atof(argv[++i]);
    else if(!strcmp(argv[i], "--zeta"))        zeta = atof(argv[++i]);
    else if(!strcmp(argv[i], "--lambda"))      lambda = atof(argv[++i]);
    else if(!strcmp(argv[i], "--lambda_con"))  lambda_con = atof(argv[++i]);
    else if(!strcmp(argv[i], "--delta"))       delta = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--burnin"))      burnin = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--epoch_start")) epoch_start = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--epoch_num"))   epoch_num = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--train"))       train = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--metropolis"))
    {
      ++i;
      if(!strcmp(argv[i], "yes") || !strcmp(argv[i], "YES"))
        metropolis = true;
    }
    else if(!strcmp(argv[i], "--sample_hyper"))
    {
      ++i;
      if(!strcmp(argv[i], "yes") || !strcmp(argv[i], "YES"))
        sample_hyperparameter = true;
    }
    else
    {
      printf("%s, unknown parameters, exit\n", argv[i]);
      exit(0);
    }
  }
  if (directory == NULL || data_path == NULL)
  {
    printf("Note that directory and data are not optional!\n");
    exit(0);
  }
  if (!dir_exists(directory))
    mkdir(directory, S_IRUSR | S_IWUSR | S_IXUSR);



  //Initialize the random seed using current time.
  time_t t;
  time(&t);
  long seed = static_cast<long>(t);
  g_random_number = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(g_random_number, seed);

  //Build the Corpus.
  Corpus* c = new Corpus();
  c->ReadData(data_path);

  //Initialize the DTC's hyperparameter.
  DtcHyperPara* hyper_para = new DtcHyperPara();
  hyper_para->SetPara(alpha, beta, gamma, tau, zeta, lambda, lambda_con,
                      max_iter, save_lag, delta, burnin,
                      sample_hyperparameter, metropolis);

  Dtc* dtc = new Dtc();

  if(train)
  {
    dtc->SetupStateFromCorpus(hyper_para, c);
    dtc->OnlineEstimate(directory, epoch_start);
  }
  else	//inference is still need to be polished
  {
    int model_vocab = 12263, model_participants = 18306;
    c->size_vocab_ = model_vocab;
    c->size_participants_ = model_participants;
    dtc->SetupStateFromCorpus(hyper_para, c);
    dtc->Inference(directory, epoch_start);
  }

  delete dtc;
  delete hyper_para;
  delete c;

  gsl_rng_free(g_random_number);

  return 0;
}
