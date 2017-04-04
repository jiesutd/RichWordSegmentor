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
class DenseFeature {
public:
	//all state inter dependent features
	vector<vector<int> > _wordIds, _keyCharIds, _lengthIds, _actionIds;
	vector<Tensor<xpu, 3, dtype> > _wordPrime, _wordPrimeLoss;
	vector<Tensor<xpu, 3, dtype> > _allwordPrime;
	vector<Tensor<xpu, 3, dtype> > _keyCharPrime, _keyCharPrimeLoss;
	vector<Tensor<xpu, 3, dtype> > _lengthPrime, _lengthPrimeLoss;
	vector<Tensor<xpu, 2, dtype> > _wordRep, _wordRepLoss;
	vector<Tensor<xpu, 2, dtype> > _allwordRep, _allwordRepLoss;
	vector<Tensor<xpu, 2, dtype> > _keyCharRep, _keyCharRepLoss;
	vector<Tensor<xpu, 2, dtype> > _lengthRep, _lengthRepLoss;
	
	vector<Tensor<xpu, 2, dtype> > _wordUnitRep, _wordUnitRepLoss, _wordUnitRepMask;
	vector<Tensor<xpu, 2, dtype> > _wordHidden, _wordHiddenLoss;

	vector<vector<Tensor<xpu, 2, dtype> > > _wordRNNHiddenBuf;
	vector<Tensor<xpu, 2, dtype> > _wordRNNHidden, _wordRNNHiddenLoss;  //lstm


	vector<Tensor<xpu, 3, dtype> > _actionPrime, _actionPrimeLoss;
	vector<Tensor<xpu, 2, dtype> > _actionRep, _actionRepLoss, _actionRepMask;
	vector<Tensor<xpu, 2, dtype> > _actionHidden, _actionHiddenLoss;

	vector<vector<Tensor<xpu, 2, dtype> > > _actionRNNHiddenBuf;
	vector<Tensor<xpu, 2, dtype> > _actionRNNHidden, _actionRNNHiddenLoss;  //lstm

	int _steps;
	int _wordnum;
	int _buffer;

public:
	DenseFeature() {
		_steps = 0;
		_wordnum = 0;
		_buffer = 0;
	}

	~DenseFeature() {
		clear();
	}

public:
	inline void init(int wordnum, int steps, int wordDim, int allwordDim, int charDim, int lengthDim, int wordNgram, int wordUnitDim, int wordHiddenDim, int wordRNNDim,
			int actionDim, int actionNgram, int actionHiddenDim, int actionRNNDim, int keyCharNum, int lengthNum, int buffer = 0) {

		clear();
		_steps = steps;
		_wordnum = wordnum;
		_buffer = buffer;

		if (wordnum > 0) {
			_wordIds.resize(wordnum);
			_keyCharIds.resize(wordnum);
			_lengthIds.resize(wordnum);

			_wordPrime.resize(wordnum);
			_allwordPrime.resize(wordnum);
			_wordRep.resize(wordnum);
			_allwordRep.resize(wordnum);
			_keyCharPrime.resize(wordnum);
			_keyCharRep.resize(wordnum);
			_lengthPrime.resize(wordnum);
			_lengthRep.resize(wordnum);
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
			_keyCharPrimeLoss.resize(wordnum);
			_keyCharRepLoss.resize(wordnum);
			_lengthPrimeLoss.resize(wordnum);
			_lengthRepLoss.resize(wordnum);
			_wordUnitRepLoss.resize(wordnum);
			_wordUnitRepMask.resize(wordnum);
			_wordHiddenLoss.resize(wordnum);
			_wordRNNHiddenLoss.resize(wordnum);

			for (int idx = 0; idx < wordnum; idx++) {
				_wordIds[idx].resize(wordNgram);
				_keyCharIds[idx].resize(2 * wordNgram + 1);
				_lengthIds[idx].resize(wordNgram);
				_wordPrime[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, wordDim), d_zero);
				_allwordPrime[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, allwordDim), d_zero);
				_wordRep[idx] = NewTensor<xpu>(Shape2(1, wordNgram * wordDim), d_zero);
				_allwordRep[idx] = NewTensor<xpu>(Shape2(1, wordNgram * allwordDim), d_zero);
				_keyCharPrime[idx] = NewTensor<xpu>(Shape3(2 * wordNgram + 1, 1, charDim), d_zero);
				_keyCharRep[idx] = NewTensor<xpu>(Shape2(1, (2 * wordNgram + 1) * charDim), d_zero);
				_lengthPrime[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, lengthDim), d_zero);
				_lengthRep[idx] = NewTensor<xpu>(Shape2(1, wordNgram * lengthDim), d_zero);
				_wordUnitRep[idx] = NewTensor<xpu>(Shape2(1, wordUnitDim), d_zero);
				_wordHidden[idx] = NewTensor<xpu>(Shape2(1, wordHiddenDim), d_zero);
				for(int idk = 0; idk < _buffer; idk++){
					_wordRNNHiddenBuf[idk][idx]  = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);
				}
				_wordRNNHidden[idx] = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);

				_wordPrimeLoss[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, wordDim), d_zero);
				_wordRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordNgram * wordDim), d_zero);
				_allwordRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordNgram * allwordDim), d_zero);
				_keyCharPrimeLoss[idx] = NewTensor<xpu>(Shape3(2 * wordNgram + 1, 1, charDim), d_zero);
				_keyCharRepLoss[idx] = NewTensor<xpu>(Shape2(1, (2 * wordNgram + 1) * charDim), d_zero);
				_lengthPrimeLoss[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, lengthDim), d_zero);
				_lengthRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordNgram * lengthDim), d_zero);
				_wordUnitRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordUnitDim), d_zero);
				_wordUnitRepMask[idx] = NewTensor<xpu>(Shape2(1, wordUnitDim), d_zero);
				_wordHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, wordHiddenDim), d_zero);
				_wordRNNHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);
			}
		}

		if (steps > 0) {
			_actionIds.resize(steps);
			_actionPrime.resize(steps);
			_actionRep.resize(steps);
			_actionHidden.resize(steps);
			if(_buffer > 0){
				_actionRNNHiddenBuf.resize(_buffer);
				for(int idk = 0; idk < _buffer; idk++){
					_actionRNNHiddenBuf[idk].resize(steps);
				}
			}
			_actionRNNHidden.resize(steps);

			_actionPrimeLoss.resize(steps);
			_actionRepLoss.resize(steps);
			_actionRepMask.resize(steps);
			_actionHiddenLoss.resize(steps);
			_actionRNNHiddenLoss.resize(steps);
			for (int idx = 0; idx < steps; idx++) {
				_actionIds[idx].resize(actionNgram);
				_actionPrime[idx] = NewTensor<xpu>(Shape3(actionNgram, 1, actionDim), d_zero);
				_actionRep[idx] = NewTensor<xpu>(Shape2(1, actionNgram * actionDim), d_zero);
				_actionHidden[idx] = NewTensor<xpu>(Shape2(1, actionHiddenDim), d_zero);
				for(int idk = 0; idk < _buffer; idk++){
					_actionRNNHiddenBuf[idk][idx]  = NewTensor<xpu>(Shape2(1, actionRNNDim), d_zero);
				}
				_actionRNNHidden[idx] = NewTensor<xpu>(Shape2(1, actionRNNDim), d_zero);

				_actionPrimeLoss[idx] = NewTensor<xpu>(Shape3(actionNgram, 1, actionDim), d_zero);
				_actionRepLoss[idx] = NewTensor<xpu>(Shape2(1, actionNgram * actionDim), d_zero);
				_actionRepMask[idx] = NewTensor<xpu>(Shape2(1, actionNgram * actionDim), d_zero);
				_actionHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, actionHiddenDim), d_zero);
				_actionRNNHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, actionRNNDim), d_zero);
			}
		}

	}

	inline void clear() {
		for (int idx = 0; idx < _wordnum; idx++) {
			_wordIds[idx].clear();
			_keyCharIds[idx].clear();
			_lengthIds[idx].clear();
			FreeSpace(&(_wordPrime[idx]));
			FreeSpace(&(_allwordPrime[idx]));
			FreeSpace(&(_wordRep[idx]));
			FreeSpace(&(_allwordRep[idx]));
			FreeSpace(&(_keyCharPrime[idx]));
			FreeSpace(&(_keyCharRep[idx]));
			FreeSpace(&(_lengthPrime[idx]));
			FreeSpace(&(_lengthRep[idx]));
			FreeSpace(&(_wordUnitRep[idx]));
			FreeSpace(&(_wordHidden[idx]));
			for(int idk = 0; idk < _buffer; idk++){
				FreeSpace(&(_wordRNNHiddenBuf[idk][idx]));
			}
			FreeSpace(&(_wordRNNHidden[idx]));

			FreeSpace(&(_wordPrimeLoss[idx]));
			FreeSpace(&(_wordRepLoss[idx]));
			FreeSpace(&(_allwordRepLoss[idx]));
			FreeSpace(&(_keyCharPrimeLoss[idx]));
			FreeSpace(&(_keyCharRepLoss[idx]));
			FreeSpace(&(_lengthPrimeLoss[idx]));
			FreeSpace(&(_lengthRepLoss[idx]));
			FreeSpace(&(_wordUnitRepLoss[idx]));
			FreeSpace(&(_wordUnitRepMask[idx]));
			FreeSpace(&(_wordHiddenLoss[idx]));
			FreeSpace(&(_wordRNNHiddenLoss[idx]));
		}
		_wordIds.clear();
		_keyCharIds.clear();
		_lengthIds.clear();
		_wordPrime.clear();
		_allwordPrime.clear();
		_wordRep.clear();
		_allwordRep.clear();
		_keyCharPrime.clear();
		_keyCharRep.clear();
		_lengthPrime.clear();
		_lengthRep.clear();
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
		_keyCharPrimeLoss.clear();
		_keyCharRepLoss.clear();
		_lengthPrimeLoss.clear();
		_lengthRepLoss.clear();
		_wordUnitRepLoss.clear();
		_wordUnitRepMask.clear();
		_wordHiddenLoss.clear();
		_wordRNNHiddenLoss.clear();

		for (int idx = 0; idx < _steps; idx++) {
			_actionIds[idx].clear();
			FreeSpace(&(_actionPrime[idx]));
			FreeSpace(&(_actionRep[idx]));
			FreeSpace(&(_actionHidden[idx]));
			for(int idk = 0; idk < _buffer; idk++){
				FreeSpace(&(_actionRNNHiddenBuf[idk][idx]));
			}
			FreeSpace(&(_actionRNNHidden[idx]));

			FreeSpace(&(_actionPrimeLoss[idx]));
			FreeSpace(&(_actionRepLoss[idx]));
			FreeSpace(&(_actionRepMask[idx]));
			FreeSpace(&(_actionHiddenLoss[idx]));
			FreeSpace(&(_actionRNNHiddenLoss[idx]));
		}
		_actionIds.clear();
		_actionPrime.clear();
		_actionRep.clear();
		_actionHidden.clear();
		for(int idk = 0; idk < _buffer; idk++){
			_actionRNNHiddenBuf[idk].clear();
		}
		_actionRNNHiddenBuf.clear();
		_actionRNNHidden.clear();

		_actionPrimeLoss.clear();
		_actionRepLoss.clear();
		_actionRepMask.clear();
		_actionHiddenLoss.clear();
		_actionRNNHiddenLoss.clear();

		_wordnum = 0;
		_steps = 0;
		_buffer = 0;
	}

};

#endif /* FEATURE_DENSEFEATURE_H_ */
