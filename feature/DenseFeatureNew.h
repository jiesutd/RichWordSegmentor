/*
 * DenseFeature.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef FEATURE_DENSEFEATURE_H_
#define FEATURE_DENSEFEATURE_H_

#include "N3L.h"

template<typename xpu>
class DenseFeatureNew {
public:
	//all state inter dependent features
	vector<vector<int> >  _wordIds, _keyCharIds, _lengthIds;
	vector<Tensor<xpu, 3, dtype> > _keyCharPrime, _keyCharPrimeLoss;
	vector<Tensor<xpu, 3, dtype> > _lengthPrime, _lengthPrimeLoss;
	vector<Tensor<xpu, 2, dtype> > _keyCharRep, _keyCharRepLoss;
	vector<Tensor<xpu, 2, dtype> > _lengthRep, _lengthRepLoss;
	vector<Tensor<xpu, 2, dtype> > _stackRep, _stackRepLoss, _stackRepMask;
	vector<Tensor<xpu, 2, dtype> > _stackHidden, _stackHiddenLoss;

	vector<Tensor<xpu, 3, dtype> > _wordPrime, _wordPrimeLoss;
	vector<Tensor<xpu, 3, dtype> > _allwordPrime;

	vector<Tensor<xpu, 2, dtype> > _wordRep, _wordRepLoss;
	vector<Tensor<xpu, 2, dtype> > _allwordRep, _allwordRepLoss;
	vector<Tensor<xpu, 2, dtype> > _wordUnitRep, _wordUnitRepLoss, _wordUnitRepMask;
	vector<Tensor<xpu, 2, dtype> > _wordHidden, _wordHiddenLoss;

	vector<vector<Tensor<xpu, 2, dtype> > > _wordRNNHiddenBuf;
	vector<Tensor<xpu, 2, dtype> > _wordRNNHidden, _wordRNNHiddenLoss;  //lstm


	int _steps;
	int _wordnum;
	int _buffer;

public:
	DenseFeatureNew() {
		_steps = 0;
		_wordnum = 0;
		_buffer = 0;
	}

	~DenseFeatureNew() {
		clear();
	}

public:
	inline void init(int wordnum, int steps,  int charDim, int keyCharDim, int lengthDim, int stackDim, int stackHiddenSize, int keyCharNum, int lengthNum, 
						int wordDim, int allwordDim, int wordNgram, int wordUnitDim, int wordHiddenDim, int wordRNNDim, int buffer = 0) {

		clear();
		_steps = steps;
		_wordnum = wordnum;
		_buffer = buffer;


		if (wordnum > 0) {
			_wordIds.resize(wordnum);
			_wordPrime.resize(wordnum);
			_allwordPrime.resize(wordnum);
			_wordRep.resize(wordnum);
			_allwordRep.resize(wordnum);

			_wordUnitRep.resize(wordnum);
			_wordHidden.resize(wordnum);
			if(_buffer > 0){
				_wordRNNHiddenBuf.resize(_buffer);
				for(int idk = 0; idk < _buffer; idk++){
					_wordRNNHiddenBuf[idk].resize(wordnum);
				}
			}
			_wordRNNHidden.resize(wordnum);

			_wordPrimeLoss.resize(wordnum);
			_wordRepLoss.resize(wordnum);
			_allwordRepLoss.resize(wordnum);
			_wordUnitRepLoss.resize(wordnum);
			_wordUnitRepMask.resize(wordnum);
			_wordHiddenLoss.resize(wordnum);
			_wordRNNHiddenLoss.resize(wordnum);

			for (int idx = 0; idx < wordnum; idx++) {
				_wordIds[idx].resize(wordNgram);
				_wordPrime[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, wordDim), d_zero);
				_allwordPrime[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, allwordDim), d_zero);
				_wordRep[idx] = NewTensor<xpu>(Shape2(1, wordNgram * wordDim), d_zero);
				_allwordRep[idx] = NewTensor<xpu>(Shape2(1, wordNgram * allwordDim), d_zero);
				_wordUnitRep[idx] = NewTensor<xpu>(Shape2(1, wordUnitDim), d_zero);
				_wordHidden[idx] = NewTensor<xpu>(Shape2(1, wordHiddenDim), d_zero);
				for(int idk = 0; idk < _buffer; idk++){
					_wordRNNHiddenBuf[idk][idx]  = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);
				}
				_wordRNNHidden[idx] = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);

				_wordPrimeLoss[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, wordDim), d_zero);
				_wordRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordNgram * wordDim), d_zero);
				_allwordRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordNgram * allwordDim), d_zero);
				_wordUnitRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordUnitDim), d_zero);
				_wordUnitRepMask[idx] = NewTensor<xpu>(Shape2(1, wordUnitDim), d_zero);
				_wordHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, wordHiddenDim), d_zero);
				_wordRNNHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);
			}
		}



		if (_steps > 0) {
			_keyCharIds.resize(_steps);
			_lengthIds.resize(_steps);

			_keyCharPrime.resize(_steps);
			_keyCharRep.resize(_steps);
			_lengthPrime.resize(_steps);
			_lengthRep.resize(_steps);
			_stackRep.resize(_steps);
			
			_keyCharPrimeLoss.resize(_steps);
			_keyCharRepLoss.resize(_steps);
			_lengthPrimeLoss.resize(_steps);
			_lengthRepLoss.resize(_steps);
			_stackRepLoss.resize(_steps);
			_stackRepMask.resize(_steps);
			_stackHidden.resize(_steps);
			_stackHiddenLoss.resize(_steps);

			for (int idx = 0; idx < _steps; idx++) {
				_keyCharIds[idx].resize(keyCharNum);
				_lengthIds[idx].resize(lengthNum);
				
				_keyCharPrime[idx] = NewTensor<xpu>(Shape3(keyCharNum, 1, keyCharDim), d_zero);
				_keyCharRep[idx] = NewTensor<xpu>(Shape2(1, keyCharNum * keyCharDim), d_zero);
				_lengthPrime[idx] = NewTensor<xpu>(Shape3(lengthNum, 1, lengthDim), d_zero);
				_lengthRep[idx] = NewTensor<xpu>(Shape2(1, lengthNum * lengthDim), d_zero);
				_stackRep[idx] = NewTensor<xpu>(Shape2(1, stackDim), d_zero);
				_stackHidden[idx] = NewTensor<xpu>(Shape2(1, stackHiddenSize), d_zero);

				_keyCharPrimeLoss[idx] = NewTensor<xpu>(Shape3(keyCharNum, 1, keyCharDim), d_zero);
				_keyCharRepLoss[idx] = NewTensor<xpu>(Shape2(1, keyCharNum * keyCharDim), d_zero);
				_lengthPrimeLoss[idx] = NewTensor<xpu>(Shape3(lengthNum, 1, lengthDim), d_zero);
				_lengthRepLoss[idx] = NewTensor<xpu>(Shape2(1, lengthNum * lengthDim), d_zero);
				_stackRepLoss[idx] = NewTensor<xpu>(Shape2(1, stackDim), d_zero);
				_stackRepMask[idx] = NewTensor<xpu>(Shape2(1, stackDim), d_zero);
				_stackHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, stackHiddenSize), d_zero);

			}
		}

	}

	inline void clear() {
		for (int idx = 0; idx < _wordnum; idx++) {
			_wordIds[idx].clear();
			FreeSpace(&(_wordPrime[idx]));
			FreeSpace(&(_allwordPrime[idx]));
			FreeSpace(&(_wordRep[idx]));
			FreeSpace(&(_allwordRep[idx]));
			FreeSpace(&(_wordUnitRep[idx]));
			FreeSpace(&(_wordHidden[idx]));
			for(int idk = 0; idk < _buffer; idk++){
				FreeSpace(&(_wordRNNHiddenBuf[idk][idx]));
			}
			FreeSpace(&(_wordRNNHidden[idx]));

			FreeSpace(&(_wordPrimeLoss[idx]));
			FreeSpace(&(_wordRepLoss[idx]));
			FreeSpace(&(_allwordRepLoss[idx]));
			FreeSpace(&(_wordUnitRepLoss[idx]));
			FreeSpace(&(_wordUnitRepMask[idx]));
			FreeSpace(&(_wordHiddenLoss[idx]));
			FreeSpace(&(_wordRNNHiddenLoss[idx]));
		}
		_wordIds.clear();
		_wordPrime.clear();
		_allwordPrime.clear();
		_wordRep.clear();
		_allwordRep.clear();
		_wordUnitRep.clear();
		_wordHidden.clear();
		for(int idk = 0; idk < _buffer; idk++){
			_wordRNNHiddenBuf[idk].clear();
		}
		_wordRNNHiddenBuf.clear();
		_wordRNNHidden.clear();

		_wordPrimeLoss.clear();
		_wordRepLoss.clear();
		_allwordRepLoss.clear();
		_wordUnitRepLoss.clear();
		_wordUnitRepMask.clear();
		_wordHiddenLoss.clear();
		_wordRNNHiddenLoss.clear();


		for (int idx = 0; idx < _steps; idx++) {
			_keyCharIds[idx].clear();
			_lengthIds[idx].clear();
			FreeSpace(&(_keyCharPrime[idx]));
			FreeSpace(&(_keyCharRep[idx]));
			FreeSpace(&(_lengthPrime[idx]));
			FreeSpace(&(_lengthRep[idx]));
			FreeSpace(&(_stackRep[idx]));
			FreeSpace(&(_stackHidden[idx]));
	
			FreeSpace(&(_keyCharPrimeLoss[idx]));
			FreeSpace(&(_keyCharRepLoss[idx]));
			FreeSpace(&(_lengthPrimeLoss[idx]));
			FreeSpace(&(_lengthRepLoss[idx]));
			FreeSpace(&(_stackRepLoss[idx]));
			FreeSpace(&(_stackRepMask[idx]));
			FreeSpace(&(_stackHiddenLoss[idx]));

		}
		_keyCharIds.clear();
		_lengthIds.clear();
		_keyCharPrime.clear();
		_keyCharRep.clear();
		_lengthPrime.clear();
		_lengthRep.clear();
		_stackRep.clear();
		_stackHidden.clear();

		_keyCharPrimeLoss.clear();
		_keyCharRepLoss.clear();
		_lengthPrimeLoss.clear();
		_lengthRepLoss.clear();
		_stackRepLoss.clear();
		_stackRepMask.clear();
		_stackHiddenLoss.clear();

		_wordnum = 0;
		_steps = 0;
		_buffer = 0;

	}

};

#endif /* FEATURE_DENSEFEATURE_H_ */
