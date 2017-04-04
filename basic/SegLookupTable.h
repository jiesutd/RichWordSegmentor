/*
 * SegLookupTable.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SegLookupTable_H_
#define SRC_SegLookupTable_H_
#include "tensor.h"
#include "Utiltensor.h"
#include "MyLib.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// Weight updating process implemented without theory support,
// but recently find an EMNLP 2015 paper "An Empirical Analysis of Optimization for Max-Margin NLP"
// In all my papers that use adagrad for sparse features, I use it for parameter updating.

template<typename xpu>
class SegLookupTable {

public:

  hash_set<int> _indexers;

  Tensor<xpu, 2, dtype> _E;
  Tensor<xpu, 2, dtype> _gradE;
  Tensor<xpu, 2, dtype> _eg2E;

  Tensor<xpu, 2, dtype> _ftE;

  bool _bFineTune;
  int _nDim;
  int _nVSize;

  int _max_update;
  NRVec<int> _last_update;

  NRVec<int> _freq;


public:

  SegLookupTable() {
    _indexers.clear();
  }


  inline void initial(const NRMat<dtype>& wordEmb) {
    _nVSize = wordEmb.nrows();
    _nDim = wordEmb.ncols();

    _E = NewTensor<xpu>(Shape2(_nVSize, _nDim), d_zero);
    _gradE = NewTensor<xpu>(Shape2(_nVSize, _nDim), d_zero);
    _eg2E = NewTensor<xpu>(Shape2(_nVSize, _nDim), d_zero);
    _ftE = NewTensor<xpu>(Shape2(_nVSize, _nDim), d_one);
    assign(_E, wordEmb);
    for (int idx = 0; idx < _nVSize; idx++) {
      norm2one(_E, idx);
    }

    _bFineTune = true;

    _max_update = 0;
    _last_update.resize(_nVSize);
    _last_update = 0;

    _freq.resize(_nVSize);
    _freq = -1;
  }

  inline void setEmbFineTune(bool bFineTune) {
    _bFineTune = bFineTune;
  }

  inline void setFrequency(hash_map<int, int> wordfreq) {
		static hash_map<int, int>::iterator action_iter;
		for (action_iter = wordfreq.begin(); action_iter != wordfreq.end(); action_iter++) {
			_freq[action_iter->first] = action_iter->second;
		}
  }

  inline int getFrequency(int id){
  	return _freq[id];
  }

  inline void release() {
    FreeSpace(&_E);
    FreeSpace(&_gradE);
    FreeSpace(&_eg2E);
    FreeSpace(&_ftE);
    _indexers.clear();
  }

  virtual ~SegLookupTable() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = 0;
    static hash_set<int>::iterator it;
    for (int idx = 0; idx < _nDim; idx++) {
      for (it = _indexers.begin(); it != _indexers.end(); ++it) {
        result += _gradE[*it][idx] * _gradE[*it][idx];
      }
    }


    return result;
  }

  inline void scaleGrad(dtype scale) {
    static hash_set<int>::iterator it;
    for (int idx = 0; idx < _nDim; idx++) {
      for (it = _indexers.begin(); it != _indexers.end(); ++it) {
        _gradE[*it][idx] = _gradE[*it][idx] * scale;
      }
    }
  }

  inline bool bEmbFineTune()
  {
    return _bFineTune;
  }

public:
  /* (1) unk is -1 when training
   * (2) if unk >= 0, then must be in test phase, if last_update equals zero,
   *     denoting that never be trained, thus we regards it as unknown.
   */
  void GetEmb(int id, Tensor<xpu, 2, dtype> y, int unk = -1) {
    updateSparseWeight(id);
    assert(y.size(0) == 1);
    y = 0.0;
    if(unk < 0 || _last_update[id] > 0){
    	y[0] += _E[id];
    }
    else{
    	y[0] += _E[unk];
    }
  }

  // loss is stopped at this layer, since the input is one-hold alike
  void EmbLoss(int id, Tensor<xpu, 2, dtype> ly) {
    if(!_bFineTune) return;
    //_gradE
    assert(ly.size(0) == 1);
    _gradE[id] += ly[0];
    _indexers.insert(id);

  }


  void randomprint(int num) {
    static int _nVSize, _nDim;
    _nVSize = _E.size(0);
    _nDim = _E.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % _nVSize;
      int idy = rand() % _nDim;

      std::cout << "_E[" << idx << "," << idy << "]=" << _E[idx][idy] << " ";

      count++;
    }

    std::cout << std::endl;
  }

  void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {

    if(!_bFineTune) return;
    static hash_set<int>::iterator it;
    _max_update++;

    Tensor<xpu, 1, dtype> sqrt_eg2E = NewTensor<xpu>(Shape1(_E.size(1)), d_zero);

    for (it = _indexers.begin(); it != _indexers.end(); ++it) {
      int index = *it;
      _eg2E[index] = _eg2E[index] + _gradE[index] * _gradE[index];
      sqrt_eg2E = F<nl_sqrt>(_eg2E[index] + adaEps);
      _E[index] = (_E[index] * sqrt_eg2E - _gradE[index] * adaAlpha) / (adaAlpha * regularizationWeight + sqrt_eg2E);
      _ftE[index] = sqrt_eg2E / (adaAlpha * regularizationWeight + sqrt_eg2E);
    }

    FreeSpace(&sqrt_eg2E);

    clearGrad();
  }

  void clearGrad() {
    static hash_set<int>::iterator it;

    for (it = _indexers.begin(); it != _indexers.end(); ++it) {
      int index = *it;
      _gradE[index] = 0.0;
    }

    _indexers.clear();

  }

  void updateSparseWeight(int wordId) {
    if(!_bFineTune) return;
    if (_last_update[wordId] < _max_update) {
      int times = _max_update - _last_update[wordId];
      _E[wordId] = _E[wordId] * F<nl_exp>(times * F<nl_log>(_ftE[wordId]));
      _last_update[wordId] = _max_update;
    }
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _E);
    SaveBinary(outf, _gradE);
    SaveBinary(outf, _eg2E);
    SaveBinary(outf, _ftE);

    WriteBinary(outf, _bFineTune);
    WriteBinary(outf, _nDim);
    WriteBinary(outf, _nVSize);
    WriteBinary(outf, _max_update);
    WriteVector(outf, _last_update);
    WriteVector(outf, _freq);
  }
  void loadModel(LStream &inf) {
    LoadBinary(inf, &_E, false);
    LoadBinary(inf, &_gradE, false);
    LoadBinary(inf, &_eg2E, false);
    LoadBinary(inf, &_ftE, false);  

    ReadBinary(inf, _bFineTune);
    ReadBinary(inf, _nDim);
    ReadBinary(inf, _nVSize);
    ReadBinary(inf, _max_update);
    ReadVector(inf, _last_update);
    ReadVector(inf, _freq);
  }

};

#endif /* SRC_SegLookupTable_H_ */
