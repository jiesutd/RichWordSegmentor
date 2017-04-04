/*
 * DenseFeatureForward.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef FEATURE_DENSEFEATUREFORWARD_H_
#define FEATURE_DENSEFEATUREFORWARD_H_

#include "N3L.h"
template<typename xpu>
class DenseFeatureForward {
public:
	//state inter dependent features
	//word
	Tensor<xpu, 3, dtype> _wordPrime;
	Tensor<xpu, 3, dtype> _allwordPrime;
	Tensor<xpu, 3, dtype> _keyCharPrime;
	Tensor<xpu, 3, dtype> _lengthPrime;
	Tensor<xpu, 2, dtype> _wordRep, _allwordRep, _keyCharRep, _lengthRep;
	Tensor<xpu, 2, dtype> _wordUnitRep, _wordUnitRepMask;
	Tensor<xpu, 2, dtype> _wordHidden;

	Tensor<xpu, 2, dtype> _stackRep, _stackRepMask;
	Tensor<xpu, 2, dtype> _stackHidden;

	vector<Tensor<xpu, 2, dtype> > _wordRNNHiddenBuf;
	Tensor<xpu, 2, dtype> _wordRNNHidden;  //lstm




	bool _bAllocated;
	bool _bTrain;
	int _buffer;

	int _wordDim, _allwordDim, _lengthDim, _charDim,_keyCharDim, _wordNgram, _wordHiddenDim, _wordRNNDim, _wordUnitDim;
	int _keyCharNum, _lengthNum;
	int _stackRepDim, _stackHiddenSize;


public:
	DenseFeatureForward() {
		_bAllocated = false;
		_bTrain = false;
		_buffer = 0;

		_wordDim = 0;
		_allwordDim = 0;
		_charDim = 0;
		_lengthDim = 0;
		_wordNgram = 0;
		_wordUnitDim = 0;
		_wordHiddenDim = 0;
		_wordRNNDim = 0;

		_stackRepDim = 0;
		_stackHiddenSize = 0;

		_keyCharNum = 0;
		_lengthNum = 0;

	}

	~DenseFeatureForward() {
		clear();
	}

public:
	inline void init(int wordDim, int allwordDim, int charDim, int keyCharDim, int lengthDim, int wordNgram, int wordUnitDim, int wordHiddenDim, int wordRNNDim,
					 int stackRepDim, int stackHiddenSize, int keyCharNum, int lengthNum, int buffer = 0, bool bTrain = false) {
		clear();
		_buffer = buffer;
		_keyCharDim = keyCharDim;
		_wordDim = wordDim;
		_allwordDim = allwordDim;
		_charDim = charDim;
		_lengthDim = lengthDim;
		_wordNgram = wordNgram;
		_wordUnitDim = wordUnitDim;
		// _wordUnitDim = wordNgram * wordDim + wordNgram * allwordDim + (2 * wordNgram + 1) * charDim + wordNgram * lengthDim;
		_wordHiddenDim = wordHiddenDim;
		_wordRNNDim = wordRNNDim;
		_keyCharNum = keyCharNum;
		_lengthNum = lengthNum;

		_stackRepDim = stackRepDim;
		_stackHiddenSize = stackHiddenSize;

		_wordPrime = NewTensor<xpu>(Shape3(_wordNgram, 1, _wordDim), d_zero);
		_allwordPrime = NewTensor<xpu>(Shape3(_wordNgram, 1, _allwordDim), d_zero);
		_wordRep = NewTensor<xpu>(Shape2(1, _wordNgram * _wordDim), d_zero);
		_allwordRep = NewTensor<xpu>(Shape2(1, _wordNgram * _allwordDim), d_zero);
		_keyCharPrime = NewTensor<xpu>(Shape3(_keyCharNum, 1, _keyCharDim), d_zero);
		_keyCharRep = NewTensor<xpu>(Shape2(1, _keyCharNum * _keyCharDim), d_zero);
		_lengthPrime = NewTensor<xpu>(Shape3(_lengthNum, 1, _lengthDim), d_zero);
		_lengthRep = NewTensor<xpu>(Shape2(1, _lengthNum*_lengthDim), d_zero);
		_wordUnitRep = NewTensor<xpu>(Shape2(1, _wordUnitDim), d_zero);
		_wordHidden = NewTensor<xpu>(Shape2(1, _wordHiddenDim), d_zero);

		_stackRep = NewTensor<xpu>(Shape2(1, _stackRepDim), d_zero);
		_stackHidden = NewTensor<xpu>(Shape2(1, _stackHiddenSize), d_zero);

		if (_buffer > 0) {
			_wordRNNHiddenBuf.resize(_buffer);
			for (int idk = 0; idk < _buffer; idk++) {
				_wordRNNHiddenBuf[idk] = NewTensor<xpu>(Shape2(1, _wordRNNDim), d_zero);
			}
		}
		_wordRNNHidden = NewTensor<xpu>(Shape2(1, _wordRNNDim), d_zero);

		if (bTrain) {
			_bTrain = bTrain;
			_wordUnitRepMask = NewTensor<xpu>(Shape2(1, _wordUnitDim), d_zero);
			_stackRepMask = NewTensor<xpu>(Shape2(1, _stackRepDim), d_zero);

		}

		_bAllocated = true;
	}

	inline void clear() {
		if (_bAllocated) {
			_wordDim = 0;
			_allwordDim = 0;
			_charDim = 0;
			_lengthDim = 0;
			_wordNgram = 0;
			_wordUnitDim = 0;
			_wordHiddenDim = 0;
			_wordRNNDim = 0;
			_keyCharDim = 0;
			_keyCharNum = 0;
			_lengthNum = 0;

			_stackRepDim = 0;
			_stackHiddenSize = 0;
			FreeSpace(&_wordPrime);
			FreeSpace(&_allwordPrime);
			FreeSpace(&_wordRep);
			FreeSpace(&_allwordRep);
			FreeSpace(&_keyCharPrime);
			FreeSpace(&_keyCharRep);
			FreeSpace(&_lengthPrime);
			FreeSpace(&_lengthRep);
			FreeSpace(&_wordUnitRep);
			FreeSpace(&_wordHidden);

			FreeSpace(&_stackRep);
			FreeSpace(&_stackHidden);
			if (_buffer > 0) {
				for (int idk = 0; idk < _buffer; idk++) {
					FreeSpace(&(_wordRNNHiddenBuf[idk]));
				}
				_wordRNNHiddenBuf.clear();
			}
			FreeSpace(&_wordRNNHidden);
			
			if (_bTrain) {
				FreeSpace(&_wordUnitRepMask);
				FreeSpace(&_stackRepMask);
			}
			_bAllocated = false;
			_bTrain = false;
			_buffer = 0;
		}
	}

	inline void copy(const DenseFeatureForward<xpu>& other) {
		if (other._bAllocated) {
			if (_bAllocated) {
				if (_wordDim != other._wordDim || _allwordDim != other._allwordDim ||  _charDim != other._charDim || _keyCharDim != other._keyCharDim || _lengthDim != other._lengthDim
						|| _wordNgram != other._wordNgram || _wordUnitDim != other._wordUnitDim ||_wordHiddenDim != other._wordHiddenDim || _wordRNNDim != other._wordRNNDim
						|| _stackRepDim != other._stackRepDim || _stackHiddenSize != other._stackHiddenSize ) {
					std::cout << "please check, error allocatation somewhere" << std::endl;
					return;
				}
			} else {
				init(other._wordDim, other._allwordDim, other._charDim, other._keyCharDim, other._lengthDim, other._wordNgram, other._wordUnitDim,other._wordHiddenDim, other._wordRNNDim, 
					 other._stackRepDim, other._stackHiddenSize, other._keyCharNum, other._lengthNum, other._buffer, other._bTrain);
			}
			Copy(_wordPrime, other._wordPrime);
			Copy(_allwordPrime, other._allwordPrime);
			Copy(_wordRep, other._wordRep);
			Copy(_allwordRep, other._allwordRep);
			Copy(_keyCharPrime, other._keyCharPrime);
			Copy(_keyCharRep, other._keyCharRep);
			Copy(_lengthPrime, other._lengthPrime);
			Copy(_lengthRep, other._lengthRep);
			Copy(_wordUnitRep, other._wordUnitRep);
			Copy(_wordHidden, other._wordHidden);

			Copy(_stackRep, other._stackRep);
			Copy(_stackHidden, other._stackHidden);


			for(int idk = 0; idk < _buffer; idk++){
				Copy(_wordRNNHiddenBuf[idk], other._wordRNNHiddenBuf[idk]);
			}
			Copy(_wordRNNHidden, other._wordRNNHidden);


			if (other._bTrain) {
				Copy(_wordUnitRepMask, other._wordUnitRepMask);
				Copy(_stackRepMask, other._stackRepMask);
			}
		} else {
			clear();
		}

		_bAllocated = other._bAllocated;
		_bTrain = other._bTrain;
		_buffer = other._buffer;
	}


	/*
	 inline DenseFeatureForward<xpu>& operator=(const DenseFeatureForward<xpu> &rhs) {
	 // Check for self-assignment!
	 if (this == &rhs)      // Same object?
	 return *this;        // Yes, so skip assignment, and just return *this.
	 copy(rhs);
	 return *this;
	 }
	 */
};

#endif /* FEATURE_DENSEFEATUREFORWARD_H_ */
