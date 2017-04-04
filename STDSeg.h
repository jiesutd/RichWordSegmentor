/*
 * Segmentor.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include "N3L.h"

#include "model/CharLSTMSTD_noword.h"
#include "Options.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Segmentor {
public:
	std::string nullkey;
	std::string unknownkey;
	std::string paddingtag;
	std::string seperateKey;

public:
	Segmentor();
	virtual ~Segmentor();

public:

#if USE_CUDA==1
	BeamSearcher<gpu> m_classifier;
#else
	BeamSearcher<cpu> m_classifier;
#endif

	Options m_options;

	Pipe m_pipe;

	hash_map<string, int> m_word_stat;

public:
	void readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb);

	void readWordClusters(const string& inFile);

	int createAlphabet(const vector<Instance>& vecInsts);

	int addTestWordAlpha(const vector<Instance>& vecInsts);
	
	int allWordAlphaEmb(const string& inFile, NRMat<dtype>& emb);

public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
		const string& wordEmbFile, const string& charEmbFile, const string& bicharEmbFile, const string& layerFile, const string& numberlayerFile);

	void predict(const Instance& input, vector<string>& output);
	void test(const string& testFile, const string& outputFile, const string& modelFile);

	// static training
	void getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions);

public:
	void readEmbeddings(Alphabet &alpha, const string& inFile, NRMat<dtype>& emb);

	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

public:


};

#endif /* SRC_PARSER_H_ */
