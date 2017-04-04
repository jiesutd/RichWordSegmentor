/*
 * DenseFeatureForward.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef HIDDENFORWARD_H_
#define HIDDENFORWARD_H_

#include "N3L.h"
template<typename xpu>
class HiddenForward {
public:
	//state inter independent features
	Tensor<xpu, 2, dtype> _finalInHidden, _finalInHiddenLoss;  //sep in
	Tensor<xpu, 2, dtype> _finalOutHidden, _finalOutHiddenLoss;  //sep out

	bool _bAllocated;
	bool _bTrain;

	int _finalInHiddenDim, _finalOutHiddenDim;

public:
	HiddenForward() {
		_bAllocated = false;
		_bTrain = false;

		_finalInHiddenDim = 0;
		_finalOutHiddenDim = 0;
	}

	~HiddenForward() {
		clear();
	}

public:
	inline void init(int finalInHiddenDim, int finalOutHiddenDim, bool bTrain = false) {
		clear();
		_finalInHiddenDim = finalInHiddenDim;
		_finalOutHiddenDim = finalOutHiddenDim;

		_finalInHidden = NewTensor<xpu>(Shape2(1, _finalInHiddenDim), d_zero);
		_finalOutHidden = NewTensor<xpu>(Shape2(1, _finalOutHiddenDim), d_zero);

		if (bTrain) {
			_bTrain = bTrain;
			_finalInHiddenLoss = NewTensor<xpu>(Shape2(1, _finalInHiddenDim), d_zero);
			_finalOutHiddenLoss = NewTensor<xpu>(Shape2(1, _finalOutHiddenDim), d_zero);
		}

		_bAllocated = true;
	}


	inline void clear() {
		if (_bAllocated) {
			_finalInHiddenDim = 0;
			_finalOutHiddenDim = 0;
			FreeSpace(&_finalInHidden);
			FreeSpace(&_finalOutHidden);

			if (_bTrain) {
				FreeSpace(&_finalInHiddenLoss);
				FreeSpace(&_finalOutHiddenLoss);
			}
			_bAllocated = false;
			_bTrain = false;
		}
	}


	inline void copy(const HiddenForward<xpu>& other) {
		if (other._bAllocated) {
			if (_bAllocated) {
				if (_finalInHiddenDim != other._finalInHiddenDim || _finalOutHiddenDim != other._finalOutHiddenDim) {
					std::cout << "please check, error allocatation somewhere" << std::endl;
					return;
				}
			} else {
				init(other._finalInHiddenDim, other._finalOutHiddenDim, other._bTrain);
			}
			if (other._bTrain) {
				Copy(_finalInHiddenLoss, other._finalInHiddenLoss);
				Copy(_finalOutHiddenLoss, other._finalOutHiddenLoss);
			}
		} else {
			clear();
		}

		_bAllocated = other._bAllocated;
		_bTrain = other._bTrain;
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
