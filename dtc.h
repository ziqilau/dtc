/*
 * dtc.h
 *
 *  Created on: 2013-3-4
 *      Author: ziqilau
 */

#ifndef DTC_H_
#define DTC_H_

#include "corpus.h"

class Dtc
{
public:
  Dtc();
  ~Dtc();
  void SetupStateFromCorpus(DtcHyperPara* hyper_para, Corpus* c);
  void BatchEstimate();
  void OnlineEstimate();

public:
  DtcHyperPara* dtc_hyper_para_;
  DtcState* dtc_state_;
};

#endif /* DTC_H_ */
