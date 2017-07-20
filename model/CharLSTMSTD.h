/*
 * LSTMBeamSearcher.h
 *  CharLSTMSTD, use both allword and word, while word embeding size is 0 in default
 *  Created on: Jan 25, 2017
 *      Author: Jie Yang
 */

#ifndef SRC_TLSTMBeamSearcher_H_
#define SRC_TLSTMBeamSearcher_H_

#if defined __GNUC__ || defined __APPLE__
#include <ext/hash_set>
#else
#include <hash_set>
#endif

#include <iostream>
#include <assert.h>
#include "../basic/io.h"
#include "Feature.h"
#include "DenseFeatureExtraction.h"
#include "DenseFeatureNew.h"
#include "DenseFeatureChar.h"
#include "N3L.h"
#include "NeuralState.h"
#include "Action.h"
#include "SegLookupTable.h"
#include "NewConcat.h"


#define LSTM_ALG LSTM_STD

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//re-implementation of Yue and Clark ACL (2007)
template<typename xpu>
class BeamSearcher {
public:
	BeamSearcher() {
		_dropOut = 0.5;
		_delta = 0.2;
		_oovRatio = 0.2;
		_oovFreq = 3;
		_buffer = 6;
	}
	~BeamSearcher() {
	}

public:

	UniLayer1O<xpu> _nnlayer_sep_output;
	UniLayer1O<xpu> _nnlayer_app_output;

	UniLayer<xpu> _nnlayer_final_hidden;

	LSTM_ALG<xpu> _char_left_rnn;
	LSTM_ALG<xpu> _char_right_rnn;

	UniLayer<xpu> _nnlayer_char_hidden;
	UniLayer<xpu> _nnlayer_char_hidden2;

	
	UniLayer<xpu> _nnlayer_stack_hidden;
	UniLayer<xpu> _nnlayer_word_hidden;

	SegLookupTable<xpu> _words;
	LookupTable<xpu> _allwords;

	LookupTable<xpu> _chars;
	LookupTable<xpu> _bichars;
	LookupTable<xpu> _stackchars;
	LookupTable<xpu> _lengths;

	int _wordSize = 0;
	int _allwordSize = 0;
	int _lengthSize = 0;
	int _wordDim = 0;
	int _allwordDim = 0;
	int _lengthDim = 0;
	int _wordNgram = 0;
	int _wordRepresentDim = 0;
	int _charSize = 0;
	int _biCharSize = 0;
	int _charDim = 0;
	int _biCharDim = 0;

	int _actionNgram = 0;

	int _charcontext = 0;
	int _charwindow = 0;
	int _charRepresentDim = 0;

	int _wordRNNHiddenSize = 0;
	int _charRNNHiddenSize = 0;

	int _wordHiddenSize = 0;
	int _charHiddenSize = 0;
	int _charDummySize = 0;


	int _final_hiddenInSize = 0;
	int _final_hiddenOutSize = 0;


	int _keyCharNum = 0;
	int _lengthNum = 0;
	int _keyCharDim = 0;

	int _stackRepDim = 0;
	int _stackHiddenSize = 0;

	DenseFeatureExtraction<xpu> fe;

	int _linearfeatSize = 0;

	Metric _eval;

	dtype _dropOut;

	dtype _delta;

	dtype _oovRatio;

	int _oovFreq = 0;

	int _buffer;

	enum {
		BEAM_SIZE = 8, MAX_SENTENCE_SIZE = 512
	};

public:

	inline void addToFeatureAlphabet(hash_map<string, int> feat_stat, int featCutOff = 0) {
		fe.addToFeatureAlphabet(feat_stat, featCutOff);
	}

	inline void addToWordAlphabet(hash_map<string, int> word_stat, int wordCutOff = 0) {
		fe.addToWordAlphabet(word_stat, wordCutOff);
	}

	inline void addToAllWordAlphabet(hash_map<string, int> allword_stat, int allwordCutOff = 0) {
		fe.addToAllWordAlphabet(allword_stat, allwordCutOff);
	}
	
	inline void addToCharAlphabet(hash_map<string, int> char_stat, int charCutOff = 0) {
		fe.addToCharAlphabet(char_stat, charCutOff);
	}

	inline void addToBiCharAlphabet(hash_map<string, int> bichar_stat, int bicharCutOff = 0) {
		fe.addToBiCharAlphabet(bichar_stat, bicharCutOff);
	}

	inline void addToActionAlphabet(hash_map<string, int> action_stat) {
		fe.addToActionAlphabet(action_stat);
	}

	inline void setAlphaIncreasing(bool bAlphaIncreasing) {
		fe.setAlphaIncreasing(bAlphaIncreasing);
	}

	inline void initAlphabet() {
		fe.initAlphabet();
	}

	inline void loadAlphabet() {
		fe.loadAlphabet();
	}

	inline void extractFeature(const CStateItem<xpu>* curState, const CAction& nextAC, Feature& feat) {
		fe.extractFeature(curState, nextAC, feat);
	}

public:

	inline void init(const NRMat<dtype>& wordEmb, const NRMat<dtype>& allwordEmb, const NRMat<dtype>& lengthEmb, int wordNgram, int wordHiddenSize, int wordRNNHiddenSize,
			const NRMat<dtype>& charEmb, const NRMat<dtype>& bicharEmb, int charcontext, int charHiddenSize, int charRNNHiddenSize, int stackHiddenSize,
			int hidden_out_size,  dtype delta) {
		
		_buffer = 6;
		_wordNgram = 2;
		_actionNgram = 1;
		_delta = delta;
		_linearfeatSize = 3 * fe._featAlphabet.size();
		//_splayer_output.initial(_linearfeatSize, 10);

		_charSize = fe._charAlphabet.size();
		if (_charSize != charEmb.nrows())
			std::cout << "char number does not match for initialization of char emb table" << std::endl;
		_biCharSize = fe._bicharAlphabet.size();
		if (_biCharSize != bicharEmb.nrows())
			std::cout << "bichar number does not match for initialization of bichar emb table" << std::endl;

		_charDim = charEmb.ncols();
		_biCharDim = bicharEmb.ncols();
		
		_charcontext = charcontext;
		_charwindow = 2 * charcontext + 1;
		_charRepresentDim = (_charDim + _biCharDim) * _charwindow;

		_allwordSize = fe._allwordAlphabet.size();
		if (_allwordSize != allwordEmb.nrows())
			std::cout << "allword number does not match for initialization of allword emb table" << std::endl;				
		_wordDim = wordEmb.ncols();
		_allwordDim = allwordEmb.ncols();

		_lengthSize = lengthEmb.nrows();
		_lengthDim = lengthEmb.ncols();

		_keyCharNum = 2;
		_lengthNum = 1;
		_keyCharDim = _charDim;

		_stackRepDim = _keyCharNum*_keyCharDim + _lengthNum*_lengthDim;
		_stackHiddenSize = stackHiddenSize;
		// _stackHiddenSize = 0;


		_wordRepresentDim = _wordNgram * _wordDim +  _wordNgram * _allwordDim;
		_wordHiddenSize = wordHiddenSize;
		// _wordHiddenSize = 0;		

		_charRNNHiddenSize = charRNNHiddenSize;	
		_charHiddenSize = charHiddenSize;
		_charDummySize = _charRNNHiddenSize;

		_final_hiddenOutSize = hidden_out_size;
		_final_hiddenInSize = _wordHiddenSize + 2*_charRNNHiddenSize+ _stackHiddenSize;


		_nnlayer_sep_output.initial(_final_hiddenOutSize, 10);
		_nnlayer_app_output.initial(_final_hiddenOutSize, 20);

		_words.initial(wordEmb);
		_words.setEmbFineTune(true);
		_allwords.initial(allwordEmb);
		_allwords.setEmbFineTune(false);

		_chars.initial(charEmb);
		// _chars.setEmbFineTune(false);
		_bichars.initial(bicharEmb);
		// _bichars.setEmbFineTune(false);
		_stackchars.initial(charEmb);
		_stackchars.setEmbFineTune(true);
		_lengths.initial(lengthEmb);
		_lengths.setEmbFineTune(true);

		_nnlayer_final_hidden.initial(_final_hiddenOutSize, _final_hiddenInSize, true, 70, 0);
		_nnlayer_word_hidden.initial(_wordHiddenSize, _wordRepresentDim, true, 90, 0);
		_nnlayer_char_hidden.initial(_charHiddenSize, _charRepresentDim, true, 100, 0);
		_nnlayer_char_hidden2.initial(_charHiddenSize, _charHiddenSize, true, 100, 0);

		_char_left_rnn.initial(_charRNNHiddenSize, _charHiddenSize, true, 30);
		_char_right_rnn.initial(_charRNNHiddenSize, _charHiddenSize, false, 30);

		_nnlayer_stack_hidden.initial(_stackHiddenSize, _stackRepDim, true, 110, 0);
		showParameters();
	}

	inline void release() {
		//_splayer_output.release();

		_nnlayer_sep_output.release();
		_nnlayer_app_output.release();

		_words.release();
		_allwords.release();

		_chars.release();
		_bichars.release();

		_stackchars.release();
		_lengths.release();

		_nnlayer_final_hidden.release();
		_nnlayer_char_hidden.release();
		_nnlayer_char_hidden2.release();

		_char_left_rnn.release();
		_char_right_rnn.release();

		_nnlayer_word_hidden.release();
		_nnlayer_stack_hidden.release();

	}

	dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
		fe.setFeatureFormat(false);
		//setAlphaIncreasing(true);
		fe.setFeatAlphaIncreasing(true);
		_eval.reset();
		dtype cost = 0.0;
		for (int idx = 0; idx < sentences.size(); idx++) {
			//print sentence content 
			// cout << " Train sentence: " << endl;
			// for(int idy = 0; idy < sentences[idx].size(); ++idy) {
			// 	cout << sentences[idx][idy] << " ";
			// }
			// cout << endl;

			cost += trainOneExample(sentences[idx], goldACs[idx], sentences.size());
		}

		return cost;
	}

	// scores do not accumulate together...., big bug, refine it tomorrow or at thursday.
	dtype trainOneExample(const std::vector<std::string>& sentence, const vector<CAction>& goldAC, int num) {
		if (sentence.size() >= MAX_SENTENCE_SIZE)
			return 0.0;
		static CStateItem<xpu> lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
		static CStateItem<xpu> *lattice_index[MAX_SENTENCE_SIZE + 1];

		int length = sentence.size();
		dtype cost = 0.0;
		dtype score = 0.0;

		const static CStateItem<xpu> *pGenerator;
		const static CStateItem<xpu> *pBestGen;
		static CStateItem<xpu> *correctState;

		bool bCorrect;  // used in learning for early update
		int index, tmp_i, tmp_j;
		CAction correct_action;
		bool correct_action_scored;
		std::vector<CAction> actions; // actions to apply for a candidate
		static NRHeap<CScoredStateAction<xpu>, CScoredStateAction_Compare<xpu> > beam(BEAM_SIZE);
		static CScoredStateAction<xpu> scored_action; // used rank actions
		static CScoredStateAction<xpu> scored_correct_action;
		static DenseFeatureNew<xpu> pBestGenFeat, pGoldFeat;
		static DenseFeatureChar<xpu> charFeat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence);

		index = 0;

		/*
		 Add Character hidden  here
		 */
		charFeat.init(length, _charDim, _biCharDim, _charcontext, _charHiddenSize,_charRNNHiddenSize,_charDummySize, _buffer, true);
		int unknownCharID = fe._charAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = fe._charAlphabet[sentence[idx]];
			if (charFeat._charIds[idx] < 0)
				charFeat._charIds[idx] = unknownCharID;
		}

		int unknownBiCharID = fe._bicharAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._bicharIds[idx] = idx < length - 1 ? fe._bicharAlphabet[sentence[idx] + sentence[idx + 1]] : fe._bicharAlphabet[sentence[idx] + fe.nullkey];
			if (charFeat._bicharIds[idx] < 0)
				charFeat._bicharIds[idx] = unknownBiCharID;
		}

		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);
			dropoutcol(charFeat._charpreMask[idx], _dropOut);
			charFeat._charpre[idx] = charFeat._charpre[idx] * charFeat._charpreMask[idx];
		}
		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);

		_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
		_nnlayer_char_hidden2.ComputeForwardScore(charFeat._charHidden, charFeat._charHidden2);
		
		_char_left_rnn.ComputeForwardScore(charFeat._charHidden2, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
				charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden);
		_char_right_rnn.ComputeForwardScore(charFeat._charHidden2, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
				charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden);

		correctState = lattice_index[0];

		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;
			correct_action = goldAC[index - 1];
			bCorrect = false;
			correct_action_scored = false;
			//std::cout << "check beam start" << std::endl;
			for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
				//std::cout << "new" << std::endl;
				//std::cout << pGenerator->str() << std::endl;
				pGenerator->getCandidateActions(actions);
				for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
					// scored_action.clear();
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.nn_extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram);
					// fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram,_actionNgram);
					// // show key chars and length
					// cout << "Show feat: " << endl;
					// cout << "Key chars: ";
					// for (int idx = 0; idx < scored_action.feat._nKeyChars.size(); ++idx) {
					// 	cout << fe._charAlphabet.from_id(scored_action.feat._nKeyChars[idx]) << ",";
					// }
					// cout << endl;
					// cout << "Stack length: ";
					// for (int idx = 0; idx < scored_action.feat._nWordLengths.size(); ++idx) {
					// 	cout << scored_action.feat._nWordLengths[idx] << ",";
					// }
					// cout << endl;
					// cout << "Word feature: ";
					// for (int idx = 0; idx < scored_action.feat._nAllWordFeat.size(); ++idx) {
					// 	cout << fe._allwordAlphabet.from_id(scored_action.feat._nAllWordFeat[idx]) << ",";
					// }
					// cout << endl;
					// //end key chars and length 

					//_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
					scored_action.score = 0;
					scored_action.score += pGenerator->_score;

					scored_action.nnfeat.init(_wordDim, _allwordDim, _charDim,_keyCharDim, _lengthDim, _wordNgram, _wordRepresentDim,_wordHiddenSize, _wordRNNHiddenSize,
							_stackRepDim, _stackHiddenSize, _keyCharNum,_lengthNum, _buffer, true);

					scored_action.hidden.init(_final_hiddenInSize, _final_hiddenOutSize, true);

					// select the last two char, last and first char of stack word
					for (int tmp_k = 0; tmp_k < _keyCharNum; tmp_k++) {
						_stackchars.GetEmb(scored_action.feat._nKeyChars[tmp_k], scored_action.nnfeat._keyCharPrime[tmp_k]);
					}
					concat(scored_action.nnfeat._keyCharPrime, scored_action.nnfeat._keyCharRep);
					assert(scored_action.feat._nWordLengths.size()==1);
					for (int tmp_k = 0; tmp_k < _lengthNum; tmp_k++) {
						_lengths.GetEmb(scored_action.feat._nWordLengths[tmp_k], scored_action.nnfeat._lengthPrime[tmp_k]);
					}
					concat(scored_action.nnfeat._lengthPrime, scored_action.nnfeat._lengthRep);
					// use name wordUnitRep represent key chars and length embedding, temperally
					concat(scored_action.nnfeat._keyCharRep, scored_action.nnfeat._lengthRep, scored_action.nnfeat._stackRep);
					dropoutcol(scored_action.nnfeat._stackRepMask, _dropOut);
					scored_action.nnfeat._stackRep = scored_action.nnfeat._stackRep * scored_action.nnfeat._stackRepMask;
					_nnlayer_stack_hidden.ComputeForwardScore(scored_action.nnfeat._stackRep, scored_action.nnfeat._stackHidden);

					// add word features
					// cout << " extract words: ";
					for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
						int unknownID = fe._wordAlphabet[fe.unknownkey];
						int curFreq = _words.getFrequency(scored_action.feat._nWordFeat[tmp_k]);
						if (curFreq >= 0 && curFreq <= _oovFreq){
							//if (1.0 * rand() / RAND_MAX < _oovRatio){
								scored_action.feat._nWordFeat[tmp_k] = unknownID;
							//}
						}
						// cout << fe._allwordAlphabet.from_id(scored_action.feat._nAllWordFeat[tmp_k]) << " ";
						_words.GetEmb(scored_action.feat._nWordFeat[tmp_k], scored_action.nnfeat._wordPrime[tmp_k]);
						_allwords.GetEmb(scored_action.feat._nAllWordFeat[tmp_k], scored_action.nnfeat._allwordPrime[tmp_k]);
					}
					// cout << endl;
					concat(scored_action.nnfeat._wordPrime, scored_action.nnfeat._wordRep);
					concat(scored_action.nnfeat._allwordPrime, scored_action.nnfeat._allwordRep);
					concat(scored_action.nnfeat._wordRep, scored_action.nnfeat._allwordRep, scored_action.nnfeat._wordUnitRep);
					dropoutcol(scored_action.nnfeat._wordUnitRepMask, _dropOut);
					scored_action.nnfeat._wordUnitRep = scored_action.nnfeat._wordUnitRep * scored_action.nnfeat._wordUnitRepMask;
					_nnlayer_word_hidden.ComputeForwardScore(scored_action.nnfeat._wordUnitRep, scored_action.nnfeat._wordHidden);

					
					if (pGenerator->_nextPosition < length) {
						concat(scored_action.nnfeat._wordHidden, scored_action.nnfeat._stackHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition],charFeat._charRightRNNHidden[pGenerator->_nextPosition], scored_action.hidden._finalInHidden);
					} else {
						concat(scored_action.nnfeat._wordHidden, scored_action.nnfeat._stackHidden, charFeat._charDummy, charFeat._charDummy, scored_action.hidden._finalInHidden);
					}
					_nnlayer_final_hidden.ComputeForwardScore(scored_action.hidden._finalInHidden, scored_action.hidden._finalOutHidden);
					
					if (scored_action.action._code == CAction::SEP || scored_action.action._code == CAction::FIN) {
						_nnlayer_sep_output.ComputeForwardScore(scored_action.hidden._finalOutHidden, score);
					} else {
						_nnlayer_app_output.ComputeForwardScore(scored_action.hidden._finalOutHidden, score);
					}
					
					//std::cout << "score = " << score << std::endl;
				
					scored_action.score += score;
					//std::cout << "add start, action = " << actions[tmp_j] << ", cur ac score = " << scored_action.score << ", orgin score = " << pGenerator->_score << std::endl;;

					if (actions[tmp_j] != correct_action) {
						scored_action.score += _delta;
					}
					beam.add_elem(scored_action);

					//std::cout << "new scored_action : " << scored_action.score << ", action = " << scored_action.action << ", state = " << scored_action.item->str() << std::endl;
					//for (int tmp_k = 0; tmp_k < beam.elemsize(); ++tmp_k) {
					//  std::cout << tmp_k << ": " << beam[tmp_k].score << ", action = " << beam[tmp_k].action << ", state = " << beam[tmp_k].item->str() << std::endl;
					//}

					if (pGenerator == correctState && actions[tmp_j] == correct_action) {
						scored_correct_action = scored_action;
						correct_action_scored = true;
						//std::cout << "add gold finish" << std::endl;
					} else {
						//std::cout << "add finish" << std::endl;
					}
				}
			}
			// std::cout << "check beam finish" << std::endl;
			if (beam.elemsize() == 0) {
				std::cout << "error" << std::endl;
				for (int idx = 0; idx < sentence.size(); idx++) {
					std::cout << sentence[idx] << std::endl;
				}
				std::cout << "" << std::endl;
				return -1;
			}

			// std::cout << "check beam start" << std::endl;
			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action);
				lattice_index[index + 1]->_score = beam[tmp_j].score;
				lattice_index[index + 1]->_curFeat.copy(beam[tmp_j].feat);
				lattice_index[index + 1]->_nnfeat.copy(beam[tmp_j].nnfeat);
				lattice_index[index + 1]->_hidden.copy(beam[tmp_j].hidden);

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}
				if (pGenerator == correctState && beam[tmp_j].action == correct_action) {
					correctState = lattice_index[index + 1];
					bCorrect = true;
				}

				++lattice_index[index + 1];
			}

			if (pBestGen->IsTerminated())
				break; // while
			// update items if correct item jump out of the agenda
			if (!bCorrect) {
				// note that if bCorrect == true then the correct state has
				// already been updated, and the new value is one of the new states
				// among the newly produced from lattice[index+1].
				correctState->move(lattice_index[index + 1], correct_action);
				correctState = lattice_index[index + 1];
				lattice_index[index + 1]->_score = scored_correct_action.score;
				lattice_index[index + 1]->_curFeat.copy(scored_correct_action.feat);
				lattice_index[index + 1]->_nnfeat.copy(scored_correct_action.nnfeat);
				lattice_index[index + 1]->_hidden.copy(scored_correct_action.hidden);

				++lattice_index[index + 1];
				assert(correct_action_scored); // scored_correct_act valid
				//TRACE(index << " updated");
				//std::cout << index << " updated" << std::endl;
				//std::cout << "best score: " << pBestGen->_score << " , gold score: " << correctState->_score << std::endl;
				pBestGenFeat.init(index, index, _charDim, _keyCharDim, _lengthDim, _stackRepDim,_stackHiddenSize, _keyCharNum, _lengthNum, _wordDim, _allwordDim,
								 _wordNgram, _wordRepresentDim, _wordHiddenSize, _wordRNNHiddenSize,_buffer);
				pGoldFeat.init(index, index, _charDim, _keyCharDim, _lengthDim, _stackRepDim,_stackHiddenSize, _keyCharNum, _lengthNum,_wordDim, _allwordDim,
								 _wordNgram, _wordRepresentDim ,_wordHiddenSize, _wordRNNHiddenSize,_buffer);
				backPropagationStates(pBestGen, correctState, 1.0/num, -1.0/num, charFeat._charLeftRNNHidden_Loss, charFeat._charRightRNNHidden_Loss, charFeat._charDummy_Loss,pBestGenFeat, pGoldFeat);
				_char_left_rnn.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
						charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden,
						charFeat._charLeftRNNHidden_Loss, charFeat._charHidden2_Loss);
				_char_right_rnn.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
						charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden,
						charFeat._charRightRNNHidden_Loss, charFeat._charHidden2_Loss);
				_nnlayer_char_hidden2.ComputeBackwardLoss(charFeat._charHidden, charFeat._charHidden2, charFeat._charHidden2_Loss, charFeat._charHidden_Loss);
				_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput, charFeat._charHidden, charFeat._charHidden_Loss, charFeat._charInput_Loss);
				windowlized_backward(charFeat._charpre_Loss, charFeat._charInput_Loss, _charcontext);
				charFeat._charpre_Loss = charFeat._charpre_Loss * charFeat._charpreMask;
				for(int idx = 0; idx < length; idx++){
					unconcat(charFeat._charprime_Loss[idx], charFeat._bicharprime_Loss[idx], charFeat._charpre_Loss[idx]);
					_chars.EmbLoss(charFeat._charIds[idx], charFeat._charprime_Loss[idx]);
					_bichars.EmbLoss(charFeat._bicharIds[idx], charFeat._bicharprime_Loss[idx]);
				}

				pBestGenFeat.clear();
				pGoldFeat.clear();
				charFeat.clear();

				_eval.correct_label_count += index;
				_eval.overall_label_count += length + 1;
				return cost;
			}

		}
		
		// make sure that the correct item is stack top finally
		if (pBestGen != correctState) {
			if (!bCorrect) {
				correctState->move(lattice_index[index + 1], correct_action);
				correctState = lattice_index[index + 1];
				lattice_index[index + 1]->_score = scored_correct_action.score;
				lattice_index[index + 1]->_curFeat.copy(scored_correct_action.feat);
				lattice_index[index + 1]->_nnfeat.copy(scored_correct_action.nnfeat);
				lattice_index[index + 1]->_hidden.copy(scored_correct_action.hidden);
				assert(correct_action_scored); // scored_correct_act valid
			}

			//std::cout << "best:" << pBestGen->str() << std::endl;
			//std::cout << "gold:" << correctState->str() << std::endl;
			//std::cout << index << " updated" << std::endl;
			//std::cout << "best score: " << pBestGen->_score << " , gold score: " << correctState->_score << std::endl;
			pBestGenFeat.init(index,index, _charDim, _keyCharDim, _lengthDim, _stackRepDim,_stackHiddenSize, _keyCharNum, _lengthNum,_wordDim, _allwordDim,
								 _wordNgram, _wordRepresentDim ,_wordHiddenSize, _wordRNNHiddenSize,_buffer);
			pGoldFeat.init(index, index, _charDim, _keyCharDim, _lengthDim, _stackRepDim,_stackHiddenSize, _keyCharNum, _lengthNum,_wordDim, _allwordDim,
								 _wordNgram, _wordRepresentDim ,_wordHiddenSize, _wordRNNHiddenSize,_buffer);

			backPropagationStates(pBestGen, correctState, 1.0/num, -1.0/num, charFeat._charLeftRNNHidden_Loss,charFeat._charRightRNNHidden_Loss, charFeat._charDummy_Loss, pBestGenFeat, pGoldFeat);
			_char_left_rnn.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
					charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden,
					charFeat._charLeftRNNHidden_Loss, charFeat._charHidden2_Loss);
			_char_right_rnn.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
					charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden,
					charFeat._charRightRNNHidden_Loss, charFeat._charHidden2_Loss);
			_nnlayer_char_hidden2.ComputeBackwardLoss(charFeat._charHidden, charFeat._charHidden2, charFeat._charHidden2_Loss, charFeat._charHidden_Loss);
			_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput, charFeat._charHidden, charFeat._charHidden_Loss, charFeat._charInput_Loss);
			windowlized_backward(charFeat._charpre_Loss, charFeat._charInput_Loss, _charcontext);
			charFeat._charpre_Loss = charFeat._charpre_Loss * charFeat._charpreMask;

			for(int idx = 0; idx < length; idx++){
				unconcat(charFeat._charprime_Loss[idx], charFeat._bicharprime_Loss[idx], charFeat._charpre_Loss[idx]);
				_chars.EmbLoss(charFeat._charIds[idx], charFeat._charprime_Loss[idx]);
				_bichars.EmbLoss(charFeat._bicharIds[idx], charFeat._bicharprime_Loss[idx]);
			}

			pBestGenFeat.clear();
			pGoldFeat.clear();
			charFeat.clear();
			_eval.correct_label_count += length;
			_eval.overall_label_count += length + 1;
		} else {
			_eval.correct_label_count += length + 1;
			_eval.overall_label_count += length + 1;
		}
		return cost;
	}

	void backPropagationStates(const CStateItem<xpu> *pPredState, const CStateItem<xpu> *pGoldState, dtype predLoss, dtype goldLoss,
			Tensor<xpu, 3, dtype> charLeftRNNHidden_Loss,Tensor<xpu, 3, dtype> charRightRNNHidden_Loss, Tensor<xpu, 2, dtype> charDummy_Loss, DenseFeatureNew<xpu>& predDenseFeature, DenseFeatureNew<xpu>& goldDenseFeature) {
		if (pPredState->_nextPosition != pGoldState->_nextPosition) {
		}

		static int position, word_position;
		if (pPredState->_nextPosition == 0) {
			// preFeature
			_nnlayer_stack_hidden.ComputeBackwardLoss(predDenseFeature._stackRep, predDenseFeature._stackHidden, predDenseFeature._stackHiddenLoss, predDenseFeature._stackRepLoss);
			for(int idx = 0; idx < predDenseFeature._stackRepLoss.size(); idx++){
				predDenseFeature._stackRepLoss[idx] = predDenseFeature._stackRepLoss[idx] * predDenseFeature._stackRepMask[idx];
				unconcat(predDenseFeature._keyCharRepLoss[idx], predDenseFeature._lengthRepLoss[idx], predDenseFeature._stackRepLoss[idx]);
				unconcat(predDenseFeature._keyCharPrimeLoss[idx],predDenseFeature._keyCharRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _keyCharNum; tmp_k++) {
					_stackchars.EmbLoss(predDenseFeature._keyCharIds[idx][tmp_k], predDenseFeature._keyCharPrimeLoss[idx][tmp_k]);
				}
				unconcat(predDenseFeature._lengthPrimeLoss[idx], predDenseFeature._lengthRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _lengthNum; tmp_k++) {
					_lengths.EmbLoss(predDenseFeature._lengthIds[idx][tmp_k], predDenseFeature._lengthPrimeLoss[idx][tmp_k]);
				}	
			}

			_nnlayer_word_hidden.ComputeBackwardLoss(predDenseFeature._wordUnitRep, predDenseFeature._wordHidden, predDenseFeature._wordHiddenLoss, predDenseFeature._wordUnitRepLoss);
			for(int idx = 0; idx < predDenseFeature._wordUnitRepLoss.size(); idx++){
				predDenseFeature._wordUnitRepLoss[idx] = predDenseFeature._wordUnitRepLoss[idx] * predDenseFeature._wordUnitRepMask[idx];
				unconcat(predDenseFeature._wordRepLoss[idx], predDenseFeature._allwordRepLoss[idx], predDenseFeature._wordUnitRepLoss[idx]);
				unconcat(predDenseFeature._wordPrimeLoss[idx], predDenseFeature._wordRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
					_words.EmbLoss(predDenseFeature._wordIds[idx][tmp_k], predDenseFeature._wordPrimeLoss[idx][tmp_k]);
				}
			}

			// goldFeature
			_nnlayer_stack_hidden.ComputeBackwardLoss(goldDenseFeature._stackRep, goldDenseFeature._stackHidden, goldDenseFeature._stackHiddenLoss, goldDenseFeature._stackRepLoss);
			for(int idx = 0; idx < goldDenseFeature._stackRepLoss.size(); idx++){
				goldDenseFeature._stackRepLoss[idx] = goldDenseFeature._stackRepLoss[idx] * goldDenseFeature._stackRepMask[idx];
				unconcat(goldDenseFeature._keyCharRepLoss[idx], goldDenseFeature._lengthRepLoss[idx], goldDenseFeature._stackRepLoss[idx]);

				unconcat(goldDenseFeature._keyCharPrimeLoss[idx],goldDenseFeature._keyCharRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _keyCharNum; tmp_k++) {
					_stackchars.EmbLoss(goldDenseFeature._keyCharIds[idx][tmp_k], goldDenseFeature._keyCharPrimeLoss[idx][tmp_k]);
				}
				unconcat(goldDenseFeature._lengthPrimeLoss[idx], goldDenseFeature._lengthRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _lengthNum; tmp_k++) {
					_lengths.EmbLoss(goldDenseFeature._lengthIds[idx][tmp_k], goldDenseFeature._lengthPrimeLoss[idx][tmp_k]);
				}	
			}

			_nnlayer_word_hidden.ComputeBackwardLoss(goldDenseFeature._wordUnitRep, goldDenseFeature._wordHidden, goldDenseFeature._wordHiddenLoss, goldDenseFeature._wordUnitRepLoss);
			for(int idx = 0; idx < goldDenseFeature._wordUnitRepLoss.size(); idx++){
				goldDenseFeature._wordUnitRepLoss[idx] = goldDenseFeature._wordUnitRepLoss[idx] * goldDenseFeature._wordUnitRepMask[idx];
				unconcat(goldDenseFeature._wordRepLoss[idx], goldDenseFeature._allwordRepLoss[idx], goldDenseFeature._wordUnitRepLoss[idx]);
				unconcat(goldDenseFeature._wordPrimeLoss[idx], goldDenseFeature._wordRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
					_words.EmbLoss(goldDenseFeature._wordIds[idx][tmp_k], goldDenseFeature._wordPrimeLoss[idx][tmp_k]);
				}
			}
			return;
		}

		if (pPredState != pGoldState) {
			//sparse
			//_splayer_output.ComputeBackwardLoss(pPredState->_curFeat._nSparseFeat, predLoss);
			//_splayer_output.ComputeBackwardLoss(pGoldState->_curFeat._nSparseFeat, goldLoss);

			int length = charLeftRNNHidden_Loss.size(0);

			// predState
			position = pPredState->_nextPosition - 1;
			word_position = pPredState->_wordnum - 1;
			if (pPredState->_lastAction._code == CAction::SEP || pPredState->_lastAction._code == CAction::FIN) {
				_nnlayer_sep_output.ComputeBackwardLoss(pPredState->_hidden._finalOutHidden, predLoss, pPredState->_hidden._finalOutHiddenLoss);
			} else {
				_nnlayer_app_output.ComputeBackwardLoss(pPredState->_hidden._finalOutHidden, predLoss, pPredState->_hidden._finalOutHiddenLoss);
			}
			_nnlayer_final_hidden.ComputeBackwardLoss(pPredState->_hidden._finalInHidden, pPredState->_hidden._finalOutHidden, pPredState->_hidden._finalOutHiddenLoss,
						pPredState->_hidden._finalInHiddenLoss);
			if (position < length) {
				unconcat(predDenseFeature._wordHiddenLoss[position], predDenseFeature._stackHiddenLoss[position], charLeftRNNHidden_Loss[position], charRightRNNHidden_Loss[position], pPredState->_hidden._finalInHiddenLoss);
			} else {
				unconcat(predDenseFeature._wordHiddenLoss[position],predDenseFeature._stackHiddenLoss[position], charDummy_Loss, charDummy_Loss, pPredState->_hidden._finalInHiddenLoss);
			}

			// goldState
			position = pGoldState->_nextPosition - 1;
			word_position = pGoldState->_wordnum - 1;

			if (pGoldState->_lastAction._code == CAction::SEP || pGoldState->_lastAction._code == CAction::FIN) {
				_nnlayer_sep_output.ComputeBackwardLoss(pGoldState->_hidden._finalOutHidden, goldLoss, pGoldState->_hidden._finalOutHiddenLoss);
			} else {
				_nnlayer_app_output.ComputeBackwardLoss(pGoldState->_hidden._finalOutHidden, goldLoss, pGoldState->_hidden._finalOutHiddenLoss);
			}
			_nnlayer_final_hidden.ComputeBackwardLoss(pGoldState->_hidden._finalInHidden, pGoldState->_hidden._finalOutHidden, pGoldState->_hidden._finalOutHiddenLoss,
					pGoldState->_hidden._finalInHiddenLoss);
			if (position < length) {
				unconcat(goldDenseFeature._wordHiddenLoss[position], goldDenseFeature._stackHiddenLoss[position], charLeftRNNHidden_Loss[position], charRightRNNHidden_Loss[position], pGoldState->_hidden._finalInHiddenLoss);
			} else {
				unconcat(goldDenseFeature._wordHiddenLoss[position], goldDenseFeature._stackHiddenLoss[position], charDummy_Loss, charDummy_Loss, pGoldState->_hidden._finalInHiddenLoss);
			}			
		}
		// predstate
		word_position = pPredState->_wordnum - 1;
		Copy(predDenseFeature._lengthPrime[position], pPredState->_nnfeat._lengthPrime);
		Copy(predDenseFeature._lengthRep[position], pPredState->_nnfeat._lengthRep);
		Copy(predDenseFeature._keyCharPrime[position], pPredState->_nnfeat._keyCharPrime);
		Copy(predDenseFeature._keyCharRep[position], pPredState->_nnfeat._keyCharRep);
		Copy(predDenseFeature._stackRep[position], pPredState->_nnfeat._stackRep);
		Copy(predDenseFeature._stackRepMask[position], pPredState->_nnfeat._stackRepMask);
		Copy(predDenseFeature._stackHidden[position], pPredState->_nnfeat._stackHidden);

		Copy(predDenseFeature._wordUnitRep[position], pPredState->_nnfeat._wordUnitRep);
		Copy(predDenseFeature._wordHidden[position], pPredState->_nnfeat._wordHidden);
		// for(int idk = 0; idk < _buffer; idk++){
		// 	Copy(predDenseFeature._wordRNNHiddenBuf[idk][position], pPredState->_nnfeat._wordRNNHiddenBuf[idk]);
		// }
		// Copy(predDenseFeature._wordRNNHidden[position], pPredState->_nnfeat._wordRNNHidden);
		Copy(predDenseFeature._wordUnitRepMask[position], pPredState->_nnfeat._wordUnitRepMask);

		for (int tmp_k = 0; tmp_k < _keyCharNum; tmp_k++) {
			predDenseFeature._keyCharIds[position][tmp_k] = pPredState->_curFeat._nKeyChars[tmp_k];
		}
		for (int tmp_k = 0; tmp_k < _lengthNum; tmp_k++) {
			predDenseFeature._lengthIds[position][tmp_k] = pPredState->_curFeat._nWordLengths[tmp_k];
		}
		// goldstate
		word_position = pGoldState->_wordnum - 1;
		Copy(goldDenseFeature._lengthPrime[position], pGoldState->_nnfeat._lengthPrime);
		Copy(goldDenseFeature._lengthRep[position], pGoldState->_nnfeat._lengthRep);
		Copy(goldDenseFeature._keyCharPrime[position], pGoldState->_nnfeat._keyCharPrime);
		Copy(goldDenseFeature._keyCharRep[position], pGoldState->_nnfeat._keyCharRep);
		Copy(goldDenseFeature._stackRep[position], pGoldState->_nnfeat._stackRep);
		Copy(goldDenseFeature._stackRepMask[position], pGoldState->_nnfeat._stackRepMask);
		Copy(goldDenseFeature._stackHidden[position], pGoldState->_nnfeat._stackHidden);

		Copy(goldDenseFeature._wordUnitRep[position], pGoldState->_nnfeat._wordUnitRep);
		Copy(goldDenseFeature._wordHidden[position], pGoldState->_nnfeat._wordHidden);
		// for(int idk = 0; idk < _buffer; idk++){
		// 	Copy(goldDenseFeature._wordRNNHiddenBuf[idk][position], pGoldState->_nnfeat._wordRNNHiddenBuf[idk]);
		// }
		// Copy(goldDenseFeature._wordRNNHidden[position], pGoldState->_nnfeat._wordRNNHidden);
		Copy(goldDenseFeature._wordUnitRepMask[position], pGoldState->_nnfeat._wordUnitRepMask);
		for (int tmp_k = 0; tmp_k < _keyCharNum; tmp_k++) {
			goldDenseFeature._keyCharIds[position][tmp_k] = pGoldState->_curFeat._nKeyChars[tmp_k];
		}
		for (int tmp_k = 0; tmp_k < _lengthNum; tmp_k++) {
			goldDenseFeature._lengthIds[position][tmp_k] = pGoldState->_curFeat._nWordLengths[tmp_k];
		}
		//currently we use a uniform loss
		backPropagationStates(pPredState->_prevState, pGoldState->_prevState, predLoss, goldLoss, charLeftRNNHidden_Loss, charRightRNNHidden_Loss, charDummy_Loss, predDenseFeature, goldDenseFeature);
	}


	bool decode(const std::vector<string>& sentence, std::vector<std::string>& words) {
		setAlphaIncreasing(false);
		if (sentence.size() >= MAX_SENTENCE_SIZE)
			return false;
		static CStateItem<xpu> lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
		static CStateItem<xpu> *lattice_index[MAX_SENTENCE_SIZE + 1];

		int length = sentence.size();
		dtype cost = 0.0;
		dtype score = 0.0;

		const static CStateItem<xpu> *pGenerator;
		const static CStateItem<xpu> *pBestGen;

		int index, tmp_i, tmp_j;
		std::vector<CAction> actions; // actions to apply for a candidate
		static NRHeap<CScoredStateAction<xpu>, CScoredStateAction_Compare<xpu> > beam(BEAM_SIZE);
		static CScoredStateAction<xpu> scored_action; // used rank actions
		static Feature feat;
		static DenseFeatureChar<xpu> charFeat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence);

		index = 0;
		// cout << "813"<<endl;
		/*
		 Add Character bi rnn  here
		 */
		charFeat.init(length, _charDim, _biCharDim, _charcontext, _charHiddenSize,_charRNNHiddenSize,_charDummySize, _buffer);
		int unknownCharID = fe._charAlphabet[fe.unknownkey];
		// cout << "820"<<endl;
		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = fe._charAlphabet[sentence[idx]];
			if (charFeat._charIds[idx] < 0)
				charFeat._charIds[idx] = unknownCharID;
		}
		// cout << "824"<<endl;
		int unknownBiCharID = fe._bicharAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._bicharIds[idx] = idx < length - 1 ? fe._bicharAlphabet[sentence[idx] + sentence[idx + 1]] : fe._bicharAlphabet[sentence[idx] + fe.nullkey];
			if (charFeat._bicharIds[idx] < 0)
				charFeat._bicharIds[idx] = unknownBiCharID;
		}

		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx],  charFeat._charpre[idx]);
		}
		// cout << "838"<<endl;
		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);

		_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
		_nnlayer_char_hidden2.ComputeForwardScore(charFeat._charHidden, charFeat._charHidden2);
		_char_left_rnn.ComputeForwardScore(charFeat._charHidden2, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
				charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden);
		_char_right_rnn.ComputeForwardScore(charFeat._charHidden2, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
				charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden);
		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;
			//std::cout << "check beam start" << std::endl;
			for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
				//std::cout << "new" << std::endl;
				//std::cout << pGenerator->str() << std::endl;
				pGenerator->getCandidateActions(actions);
				for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
					//scored_action.clear();
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.setFeatureFormat(false);
					fe.nn_extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram);
					// fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram,_actionNgram);
					//_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
					scored_action.score = 0;
					scored_action.score += pGenerator->_score;
					scored_action.nnfeat.init(_wordDim, _allwordDim, _charDim, _keyCharDim, _lengthDim, _wordNgram, _wordRepresentDim,_wordHiddenSize, _wordRNNHiddenSize,
							 _stackRepDim, _stackHiddenSize,_keyCharNum,_lengthNum, _buffer);
					scored_action.hidden.init(_final_hiddenInSize,_final_hiddenOutSize);

					// select the last two char, last and first char of stack word
					for (int tmp_k = 0; tmp_k < _keyCharNum; tmp_k++) {
						_stackchars.GetEmb(scored_action.feat._nKeyChars[tmp_k], scored_action.nnfeat._keyCharPrime[tmp_k]);
					}
					concat(scored_action.nnfeat._keyCharPrime, scored_action.nnfeat._keyCharRep);
					assert(scored_action.feat._nWordLengths.size()==1);
					for (int tmp_k = 0; tmp_k < _lengthNum; tmp_k++) {
							_lengths.GetEmb(scored_action.feat._nWordLengths[tmp_k], scored_action.nnfeat._lengthPrime[tmp_k]);
						}
					concat(scored_action.nnfeat._lengthPrime, scored_action.nnfeat._lengthRep);
					// use name wordUnitRep represent key chars and length embedding, temperally
					concat(scored_action.nnfeat._keyCharRep, scored_action.nnfeat._lengthRep, scored_action.nnfeat._stackRep);
					_nnlayer_stack_hidden.ComputeForwardScore(scored_action.nnfeat._stackRep, scored_action.nnfeat._stackHidden);

					// add word features
					for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
						int unknownID = fe._wordAlphabet[fe.unknownkey];
						int curFreq = _words.getFrequency(scored_action.feat._nWordFeat[tmp_k]);
						if (curFreq >= 0 && curFreq <= _oovFreq){
							//if (1.0 * rand() / RAND_MAX < _oovRatio){
								scored_action.feat._nWordFeat[tmp_k] = unknownID;
							//}
						}
						_words.GetEmb(scored_action.feat._nWordFeat[tmp_k], scored_action.nnfeat._wordPrime[tmp_k]);
						_allwords.GetEmb(scored_action.feat._nAllWordFeat[tmp_k], scored_action.nnfeat._allwordPrime[tmp_k]);
					}
					concat(scored_action.nnfeat._wordPrime, scored_action.nnfeat._wordRep);
					concat(scored_action.nnfeat._allwordPrime, scored_action.nnfeat._allwordRep);
					concat(scored_action.nnfeat._wordRep, scored_action.nnfeat._allwordRep, scored_action.nnfeat._wordUnitRep);
					_nnlayer_word_hidden.ComputeForwardScore(scored_action.nnfeat._wordUnitRep, scored_action.nnfeat._wordHidden);

					if (pGenerator->_nextPosition < length) {
						concat(scored_action.nnfeat._wordHidden, scored_action.nnfeat._stackHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition], charFeat._charRightRNNHidden[pGenerator->_nextPosition], scored_action.hidden._finalInHidden);
					} else {
						concat(scored_action.nnfeat._wordHidden, scored_action.nnfeat._stackHidden, charFeat._charDummy, charFeat._charDummy, scored_action.hidden._finalInHidden);
					}
					_nnlayer_final_hidden.ComputeForwardScore(scored_action.hidden._finalInHidden, scored_action.hidden._finalOutHidden);
					
					if (scored_action.action._code == CAction::SEP || scored_action.action._code == CAction::FIN) {
						
						_nnlayer_sep_output.ComputeForwardScore(scored_action.hidden._finalOutHidden, score);
					} else {
						_nnlayer_app_output.ComputeForwardScore(scored_action.hidden._finalOutHidden, score);
					}
					//std::cout << "score = " << score << std::endl;
					scored_action.score += score;
					beam.add_elem(scored_action);
				}

			}
			if (beam.elemsize() == 0) {
				std::cout << "error" << std::endl;
				for (int idx = 0; idx < sentence.size(); idx++) {
					std::cout << sentence[idx] << std::endl;
				}
				std::cout << "" << std::endl;
				return false;
			}

			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action);
				lattice_index[index + 1]->_score = beam[tmp_j].score;
				lattice_index[index + 1]->_nnfeat.copy(beam[tmp_j].nnfeat);
				lattice_index[index + 1]->_hidden.copy(beam[tmp_j].hidden);

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}

				++lattice_index[index + 1];
			}

			if (pBestGen->IsTerminated())
				break; // while
		}
		pBestGen->getSegResults(words);

		charFeat.clear();

		return true;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps, dtype clip = -1.0) {
		if(clip > 0.0) {
			dtype norm = 0.0;
			//norm += _splayer_output.squarenormAll();
			norm += _nnlayer_sep_output.squarenormAll();
			norm += _nnlayer_app_output.squarenormAll();
			norm += _chars.squarenormAll();
			norm += _bichars.squarenormAll();
			norm += _nnlayer_final_hidden.squarenormAll();
			norm += _nnlayer_char_hidden.squarenormAll();
			norm += _nnlayer_char_hidden2.squarenormAll();

			norm += _char_left_rnn.squarenormAll();
			norm += _char_right_rnn.squarenormAll();

			norm += _lengths.squarenormAll();
			norm += _stackchars.squarenormAll();
			norm += _words.squarenormAll();
			norm += _allwords.squarenormAll();

			norm += _nnlayer_word_hidden.squarenormAll();
			norm += _nnlayer_stack_hidden.squarenormAll();
			
			if(norm > clip * clip){
				dtype scale = clip/sqrt(norm);
				//_splayer_output.scaleGrad(scale);
				_nnlayer_sep_output.scaleGrad(scale);
				_nnlayer_app_output.scaleGrad(scale);
				
				_chars.scaleGrad(scale);
				_bichars.scaleGrad(scale);
				_words.scaleGrad(scale);
				_allwords.scaleGrad(scale);

				_lengths.scaleGrad(scale);
				_stackchars.scaleGrad(scale);
				
				_nnlayer_final_hidden.scaleGrad(scale);
				_nnlayer_char_hidden.scaleGrad(scale);
				_nnlayer_char_hidden2.scaleGrad(scale);
				_char_left_rnn.scaleGrad(scale);
				_char_right_rnn.scaleGrad(scale);
				_nnlayer_stack_hidden.scaleGrad(scale);
				_nnlayer_word_hidden.scaleGrad(scale);
				
			}
		}
		
		//_splayer_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_nnlayer_sep_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_app_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_bichars.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_allwords.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_stack_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_lengths.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_stackchars.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_nnlayer_final_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_char_left_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_char_right_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_word_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}

	void loadInitialLayer(const string& inputLayerFile) {
		std::cout << "Load char hidden layer paremeters from file: " << inputLayerFile << std::endl;
		LStream inf(inputLayerFile, "rb");
		_nnlayer_char_hidden.loadModel(inf);
		cout << "Char hidden layer 1 initialized, input X output: "<< _nnlayer_char_hidden._W.size(1) << " X " << _nnlayer_char_hidden._W.size(0) << endl;
		_nnlayer_char_hidden2.loadModel(inf);
		cout << "Char hidden layer 2 initialized, input X output: "<< _nnlayer_char_hidden2._W.size(1) << " X " << _nnlayer_char_hidden2._W.size(0) << endl;
	}

void loadInitialLayerNumber(const string& inputLayerFile) {
		std::cout << "Load number format char hidden layer from file: " << inputLayerFile  << std::endl;
		static ifstream inf;
		if (inf.is_open()) {
		  inf.close();
		  inf.clear();
		}
		inf.open(inputLayerFile.c_str());
		static string strLine;
		int w1_in, w1_out, b1, w2_in, w2_out, b2;
		static vector<string> vecInfo;
		static string flag;
		static int line_count; 
		//find the first line, find the w1;
		
		while (1) {
			if (!my_getline(inf, strLine)) {
				break;
			}
			if (strLine.empty()){
      			continue;
			}
			split_bychar(strLine, vecInfo, ' ');

			if (vecInfo[0] == "W1:") {
				flag = "W1:";
				w1_in = atoi(vecInfo[1].c_str());
				w1_out = atoi(vecInfo[2].c_str());
				line_count = 0;
				assert(w1_in == _nnlayer_char_hidden._W.size(1));
				assert(w1_out == _nnlayer_char_hidden._W.size(0));
				cout << "Load W1:"<< w1_in << "X" << w1_out << endl;
				continue;
			} else if (vecInfo[0] == "b1:") {
				flag = "b1:";
				b1 = atoi(vecInfo[1].c_str());
				line_count = 0;
				assert(b1 == _nnlayer_char_hidden._b.size(1));
				cout << "Load b1:"<< b1 << endl;
				continue;
			} else if (vecInfo[0] == "W2:") {
				flag = "W2:";
				w2_in = atoi(vecInfo[1].c_str());
				w2_out = atoi(vecInfo[2].c_str());
				line_count = 0;
				assert(w2_in == _nnlayer_char_hidden2._W.size(1));
				assert(w2_out == _nnlayer_char_hidden2._W.size(0));
				cout << "Load W2:"<< w2_in << "X" << w2_out << endl;
				continue;
			} else if (vecInfo[0] == "b2:") {
				flag = "b2:";
				b2 = atoi(vecInfo[1].c_str());
				line_count = 0;
				assert(b2 == _nnlayer_char_hidden2._b.size(1));
				cout << "Load b2:"<< b2 << endl;
				continue;
			} else {

			}

			if (flag == "W1:") {
				for (int idx = 0; idx < w1_out; ++idx) {
					_nnlayer_char_hidden._W[idx][line_count] = atof(vecInfo[idx].c_str());
				}
				line_count++;
			} else if (flag == "b1:") { 
				for (int idx = 0; idx < b1; ++idx) {
					_nnlayer_char_hidden._b[line_count][idx] = atof(vecInfo[idx].c_str());
				}
				line_count++;
			} else if (flag == "W2:") {
				for (int idx = 0; idx < w2_out; ++idx) {
					_nnlayer_char_hidden2._W[idx][line_count] = atof(vecInfo[idx].c_str());
				}
				line_count++;
			} else if (flag == "b2:") { 
				for (int idx = 0; idx < b2; ++idx) {
					_nnlayer_char_hidden2._b[line_count][idx] = atof(vecInfo[idx].c_str());
				}
				line_count++;
			}
		}
		cout << "Number layer loaded finished!" << endl;
		cout << "Sample last value: " << _nnlayer_char_hidden._W[w1_out-1][w1_in-1] << "," << _nnlayer_char_hidden._b[0][b1-1] << "," <<_nnlayer_char_hidden2._W[w2_out-1][w2_in-1] << ","<< _nnlayer_char_hidden2._b[0][b2-1]<< endl;
	}


	void showParameters() {
		cout << "Classifier pamameters: " << endl;

		cout << "BEAM_SIZE = "<< BEAM_SIZE<< endl;
		cout << "MAX_SENTENCE_SIZE = " << MAX_SENTENCE_SIZE << endl;

		cout << "_wordSize = "<< _wordSize<< endl;
		cout << "_allwordSize = " <<  _allwordSize<< endl;
		cout << "_lengthSize = " <<  _lengthSize << endl;
		cout << "_wordDim = " <<  _wordDim << endl;
		cout << "_allwordDim = " <<  _allwordDim << endl;
		cout << "_lengthDim = " <<  _lengthDim<< endl;
		cout << "_wordNgram = " <<  _wordNgram << endl;
		cout << "_wordRepresentDim = " <<  _wordRepresentDim << endl;
		cout << "_charSize = " <<  _charSize<< endl;
		cout << "_biCharSize = " <<  _biCharSize << endl;
		cout << "_charDim = " <<  _charDim << endl;
		cout << "_biCharDim = " <<  _biCharDim << endl;
		cout << "_charcontext = " <<  _charcontext << endl;
		cout << "_charwindow = " <<  _charwindow << endl;
		cout << "_charRepresentDim = " <<  _charRepresentDim << endl;
		cout << "_actionNgram = " <<  _actionNgram << endl;
		cout << "_wordRNNHiddenSize = " <<  _wordRNNHiddenSize << endl;
		cout << "_charRNNHiddenSize = " <<  _charRNNHiddenSize << endl;
		cout << "_wordHiddenSize = " <<  _wordHiddenSize << endl;
		cout << "_charHiddenSize = " <<  _charHiddenSize << endl;
		cout << "_charDummySize = " <<  _charDummySize << endl;
		cout << "_final_hiddenOutSize = " <<  _final_hiddenOutSize << endl;
		cout << "_final_hiddenInSize = " <<  _final_hiddenInSize << endl;
		cout << "_keyCharNum = " <<  _keyCharNum << endl;
		cout << "_lengthNum = " <<  _lengthNum << endl;
		cout << "_keyCharDim = " <<  _keyCharDim << endl;
		cout << "_stackRepDim = " <<  _stackRepDim << endl;
		cout << "_stackHiddenSize = " <<  _stackHiddenSize << endl;
		cout << "_linearfeatSize = " <<  _linearfeatSize << endl;
		cout << "_dropOut = " <<  _dropOut << endl;
		cout << "_delta = " <<  _delta << endl;
		cout << "_oovRatio = " <<  _oovRatio<< endl;
		cout << "_oovFreq = " <<  _oovFreq << endl;
		cout << "_buffer = " <<  _buffer<< endl; 
	}

public:

	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

	inline void setOOVRatio(dtype oovRatio) {
		_oovRatio = oovRatio;
	}

	inline void setOOVFreq(dtype oovFreq) {
		_oovFreq = oovFreq;
	}

	inline void setWordFreq(hash_map<string, int> word_stat){
		hash_map<int, int> wordFreq;
		hash_map<string, int>::iterator word_iter;
		for (word_iter = word_stat.begin(); word_iter != word_stat.end(); word_iter++) {
			wordFreq[fe._wordAlphabet.from_string(word_iter->first)] = word_iter->second;
		}
		_words.setFrequency(wordFreq);
	}

	void writeModel(LStream &outf) {
		_nnlayer_sep_output.writeModel(outf);
		_nnlayer_app_output.writeModel(outf);
		_nnlayer_final_hidden.writeModel(outf);
		_char_left_rnn.writeModel(outf);
		_char_right_rnn.writeModel(outf);
		_nnlayer_char_hidden.writeModel(outf);
		_nnlayer_char_hidden2.writeModel(outf);
		_nnlayer_stack_hidden.writeModel(outf);
		_nnlayer_word_hidden.writeModel(outf);
		_words.writeModel(outf);
		_allwords.writeModel(outf);
		_chars.writeModel(outf);
		_bichars.writeModel(outf);
		_stackchars.writeModel(outf);
		_lengths.writeModel(outf);

		WriteBinary(outf, _wordSize);
		WriteBinary(outf, _allwordSize);
		WriteBinary(outf, _lengthSize);
		WriteBinary(outf, _wordDim);
		WriteBinary(outf, _allwordDim);
		WriteBinary(outf, _lengthDim);
		WriteBinary(outf, _wordNgram);
		WriteBinary(outf, _wordRepresentDim);
		WriteBinary(outf, _charSize);
		WriteBinary(outf, _biCharSize);
		WriteBinary(outf, _charDim);
		WriteBinary(outf, _biCharDim);
		WriteBinary(outf, _actionNgram);
		WriteBinary(outf, _charcontext);
		WriteBinary(outf, _charwindow);
		WriteBinary(outf, _charRepresentDim);
		WriteBinary(outf, _wordRNNHiddenSize);

		WriteBinary(outf, _charRNNHiddenSize);
		WriteBinary(outf, _wordHiddenSize);
		WriteBinary(outf, _charHiddenSize);
		WriteBinary(outf, _charDummySize);
		WriteBinary(outf, _final_hiddenInSize);
		WriteBinary(outf, _final_hiddenOutSize);
		WriteBinary(outf, _keyCharNum);
		WriteBinary(outf, _lengthNum);
		WriteBinary(outf, _keyCharDim);
		WriteBinary(outf, _stackRepDim);
		WriteBinary(outf, _stackHiddenSize);

		WriteBinary(outf, _linearfeatSize);
		WriteBinary(outf, _dropOut);
		WriteBinary(outf, _delta);
		WriteBinary(outf, _oovRatio);
		WriteBinary(outf, _oovFreq);
		WriteBinary(outf, _buffer);

		fe.writeModel(outf);
		_eval.writeModel(outf);

	}

	void loadModel(LStream &inf) {
		_nnlayer_sep_output.loadModel(inf);
		_nnlayer_app_output.loadModel(inf);
		_nnlayer_final_hidden.loadModel(inf);
		_char_left_rnn.loadModel(inf);
		_char_right_rnn.loadModel(inf);
		_nnlayer_char_hidden.loadModel(inf);
		_nnlayer_char_hidden2.loadModel(inf);
		_nnlayer_stack_hidden.loadModel(inf);
		cout << "word hidden"<< endl;
		_nnlayer_word_hidden.loadModel(inf);
		_words.loadModel(inf);
		_allwords.loadModel(inf);
		_chars.loadModel(inf);
		_bichars.loadModel(inf);
		_stackchars.loadModel(inf);
		_lengths.loadModel(inf);
		ReadBinary(inf, _wordSize);
		ReadBinary(inf, _allwordSize);
		ReadBinary(inf, _lengthSize);
		ReadBinary(inf, _wordDim);
		ReadBinary(inf, _allwordDim);
		ReadBinary(inf, _lengthDim);
		ReadBinary(inf, _wordNgram);
		ReadBinary(inf, _wordRepresentDim);
		ReadBinary(inf, _charSize);
		ReadBinary(inf, _biCharSize);
		ReadBinary(inf, _charDim);
		ReadBinary(inf, _biCharDim);
		ReadBinary(inf, _actionNgram);
		ReadBinary(inf, _charcontext);
		ReadBinary(inf, _charwindow);
		ReadBinary(inf, _charRepresentDim);
		ReadBinary(inf, _wordRNNHiddenSize);

		ReadBinary(inf, _charRNNHiddenSize);
		ReadBinary(inf, _wordHiddenSize);
		ReadBinary(inf, _charHiddenSize);
		ReadBinary(inf, _charDummySize);
		ReadBinary(inf, _final_hiddenInSize);
		ReadBinary(inf, _final_hiddenOutSize);
		ReadBinary(inf, _keyCharNum);
		ReadBinary(inf, _lengthNum);
		ReadBinary(inf, _keyCharDim);
		ReadBinary(inf, _stackRepDim);
		ReadBinary(inf, _stackHiddenSize);

		ReadBinary(inf, _linearfeatSize);
		ReadBinary(inf, _dropOut);
		ReadBinary(inf, _delta);
		ReadBinary(inf, _oovRatio);
		ReadBinary(inf, _oovFreq);
		ReadBinary(inf, _buffer);

		fe.loadModel(inf);
		_eval.loadModel(inf);
	}

};

#endif /* SRC_TLSTMBeamSearcher_H_ */
