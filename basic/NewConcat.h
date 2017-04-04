#ifndef NEWCONCAT
#define NEWCONCAT

#include "tensor.h"

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//@author Jie, 2nd Dec, 2016
//
// Extend the concat functions in LibN3L/concat.h, add one-one element concat(same with copy2right), unconcat(similar with copy2left, when bclear ==true, else accumulated), copy2right, copy2left
// for uncat functions, bclear by false denotes that the losses are accumulated.

//only applicable on Shape2(1,x), notice that we add the value to the target

template<typename xpu>
inline void concat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w) {
  if (w1.size(0) != 1 ||  w.size(0) != 1) {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  if (col1 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
  }
  return;
}


template<typename xpu>
inline void unconcat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w, bool bclear = false) {
  if (w1.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  if (col1 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    w1 = 0.0;
  }
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void unconcat(vector<Tensor<xpu, 2, dtype> > w1, vector<Tensor<xpu, 2, dtype> > w, bool bclear = false) {
  if (w1.size() != w.size()) {
    std::cerr << "unconcat vector size mismatch" <<w1.size()<< "<->" <<w.size()<<std::endl;
    return;
  }
  for(int idx = 0; idx < w.size();++idx) {
    unconcat(w1[idx],w[idx],bclear);
  }
  return;
}



// copy w1's value to w, as w = w1, to right
template<typename xpu>
inline void copy2right(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w) {
  if (w1.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "copy2right error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  if (col1 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
  }
  return;
}


// copy w1's value to w, as w = w1, to left
template<typename xpu>
inline void copy2left(Tensor<xpu, 2, dtype> w, Tensor<xpu, 2, dtype> w1) {
  if (w1.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "copy2left error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  if (col != col1) {
    std::cerr << "col check error!" <<col<< " VS " <<col1 << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
  }
  return;
}

// copy w1's value to w, as w = w1, to left
template<typename xpu>
inline void copy2left(vector<Tensor<xpu, 2, dtype> > w, vector<Tensor<xpu, 2, dtype> >w1) {
  if (w.size() != w1.size()) {
    std::cerr << "copy2left error, size not match!" << std::endl;
    return;
  }
  for(int idx = 0; idx < w.size(); ++idx) {
    copy2left(w[idx], w1[idx]);
  }
  return;
}

#endif
