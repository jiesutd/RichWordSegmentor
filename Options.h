#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3L.h"

using namespace std;

class Options {
public:

	int wordCutOff;
	int featCutOff;
	int charCutOff;
	int bicharCutOff;
	dtype initRange;
	int maxIter;
	int batchSize;
	dtype adaEps;
	dtype adaAlpha;
	dtype regParameter;
	dtype dropProb;
	dtype delta;
	dtype clip;
	dtype oovRatio;

	int sepHiddenSize;
	int appHiddenSize;

	int wordEmbSize;
	int lengthEmbSize;
	int keycharEmbSize;
	int wordNgram;
	int wordHiddenSize;
	int wordRNNHiddenSize;
	bool wordEmbFineTune;

	int charEmbSize;
	int bicharEmbSize;
	int mapcharEmbSize;
	int charcontext;
	int charHiddenSize;
	int charRNNHiddenSize;
	bool charEmbFineTune;
	bool bicharEmbFineTune;
	bool mapcharEmbFineTune;

	int actionEmbSize;
	int actionNgram;
	int actionHiddenSize;
	int actionRNNHiddenSize;
	int stackHiddenSize;

	int verboseIter;
	bool saveIntermediate;
	bool train;
	int maxInstance;
	vector<string> testFiles;
	string outBest;
	int finalHiddenSize;

	Options() {
		wordCutOff = 4;
		featCutOff = 0;
		charCutOff = 0;
		bicharCutOff = 0;
		initRange = 0.01;
		maxIter = 1000;
		batchSize = 1;
		adaEps = 1e-6;
		adaAlpha = 0.01;
		regParameter = 1e-8;
		dropProb = 0.0;
		delta = 0.2;
		clip = -1.0;
		oovRatio = 0.2;

		sepHiddenSize = 100;
		appHiddenSize = 80;

		wordEmbSize = 0;
		lengthEmbSize = 20;
		keycharEmbSize = 50;
		wordNgram = 2;
		wordHiddenSize = 100;
		wordRNNHiddenSize = 100;
		wordEmbFineTune = true;

		charEmbSize = 50;
		bicharEmbSize = 50;
		mapcharEmbSize =256;
		charcontext = 2;
		charHiddenSize = 150;
		charRNNHiddenSize = 100;
		charEmbFineTune = false;
		bicharEmbFineTune = false;
		mapcharEmbFineTune = false;
		stackHiddenSize = 50;

		actionEmbSize = 20;
		actionNgram = 2;
		actionHiddenSize = 30;
		actionRNNHiddenSize = 20;

		verboseIter = 100;
		saveIntermediate = true;
		train = false;
		maxInstance = -1;
		testFiles.clear();
		outBest = "";
		finalHiddenSize = 200;

	}

	virtual ~Options() {

	}

	void setOptions(const vector<string> &vecOption) {
		int i = 0;
		for (; i < vecOption.size(); ++i) {
			pair<string, string> pr;
			string2pair(vecOption[i], pr, '=');
			if (pr.first == "wordCutOff")
				wordCutOff = atoi(pr.second.c_str());
			if (pr.first == "featCutOff")
				featCutOff = atoi(pr.second.c_str());
			if (pr.first == "charCutOff")
				charCutOff = atoi(pr.second.c_str());
			if (pr.first == "bicharCutOff")
				bicharCutOff = atoi(pr.second.c_str());
			if (pr.first == "initRange")
				initRange = atof(pr.second.c_str());
			if (pr.first == "maxIter")
				maxIter = atoi(pr.second.c_str());
			if (pr.first == "batchSize")
				batchSize = atoi(pr.second.c_str());
			if (pr.first == "adaEps")
				adaEps = atof(pr.second.c_str());
			if (pr.first == "adaAlpha")
				adaAlpha = atof(pr.second.c_str());
			if (pr.first == "regParameter")
				regParameter = atof(pr.second.c_str());
			if (pr.first == "dropProb")
				dropProb = atof(pr.second.c_str());
			if (pr.first == "delta")
				delta = atof(pr.second.c_str());
			if (pr.first == "clip")
				clip = atof(pr.second.c_str());
			if (pr.first == "oovRatio")
				oovRatio = atof(pr.second.c_str());

			if (pr.first == "sepHiddenSize")
				sepHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "appHiddenSize")
				appHiddenSize = atoi(pr.second.c_str());

			if (pr.first == "wordEmbSize")
				wordEmbSize = atoi(pr.second.c_str());
			if (pr.first == "lengthEmbSize")
				lengthEmbSize = atoi(pr.second.c_str());
			if (pr.first == "keycharEmbSize")
				keycharEmbSize = atoi(pr.second.c_str());
			if (pr.first == "wordNgram")
				wordNgram = atoi(pr.second.c_str());
			if (pr.first == "wordHiddenSize")
				wordHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "wordRNNHiddenSize")
				wordRNNHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "wordEmbFineTune")
				wordEmbFineTune = (pr.second == "true") ? true : false;

			if (pr.first == "charEmbSize")
				charEmbSize = atoi(pr.second.c_str());
			if (pr.first == "bicharEmbSize")
				bicharEmbSize = atoi(pr.second.c_str());
			if (pr.first == "mapcharEmbSize")
				mapcharEmbSize = atoi(pr.second.c_str());
			if (pr.first == "charcontext")
				charcontext = atoi(pr.second.c_str());
			if (pr.first == "charHiddenSize")
				charHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "charRNNHiddenSize")
				charRNNHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "charEmbFineTune")
				charEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "bicharEmbFineTune")
				bicharEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "mapcharEmbFineTune")
				mapcharEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "stackHiddenSize")
				stackHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "actionEmbSize")
				actionEmbSize = atoi(pr.second.c_str());
			if (pr.first == "actionNgram")
				actionNgram = atoi(pr.second.c_str());
			if (pr.first == "actionHiddenSize")
				actionHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "actionRNNHiddenSize")
				actionRNNHiddenSize = atoi(pr.second.c_str());

			if (pr.first == "verboseIter")
				verboseIter = atoi(pr.second.c_str());
			if (pr.first == "train")
				train = (pr.second == "true") ? true : false;
			if (pr.first == "saveIntermediate")
				saveIntermediate = (pr.second == "true") ? true : false;
			if (pr.first == "maxInstance")
				maxInstance = atoi(pr.second.c_str());
			if (pr.first == "testFile")
				testFiles.push_back(pr.second);
			if (pr.first == "outBest")
				outBest = pr.second;
			if (pr.first == "finalHiddenSize")
				finalHiddenSize = atoi(pr.second.c_str());
		}
	}

	void showOptions() {
		std::cout << "wordCutOff = " << wordCutOff << std::endl;
		std::cout << "featCutOff = " << featCutOff << std::endl;
		std::cout << "charCutOff = " << charCutOff << std::endl;
		std::cout << "bicharCutOff = " << bicharCutOff << std::endl;
		std::cout << "initRange = " << initRange << std::endl;
		std::cout << "maxIter = " << maxIter << std::endl;
		std::cout << "batchSize = " << batchSize << std::endl;
		std::cout << "adaEps = " << adaEps << std::endl;
		std::cout << "adaAlpha = " << adaAlpha << std::endl;
		std::cout << "regParameter = " << regParameter << std::endl;
		std::cout << "dropProb = " << dropProb << std::endl;
		std::cout << "delta = " << delta << std::endl;
		std::cout << "clip = " << clip << std::endl;
		std::cout << "oovRatio = " << oovRatio << std::endl;

		std::cout << "sepHiddenSize = " << sepHiddenSize << std::endl;
		std::cout << "appHiddenSize = " << appHiddenSize << std::endl;

		std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
		std::cout << "lengthEmbSize = " << lengthEmbSize << std::endl;
		std::cout << "keycharEmbSize = " << keycharEmbSize << std::endl;
		std::cout << "wordNgram = " << wordNgram << std::endl;
		std::cout << "wordHiddenSize = " << wordHiddenSize << std::endl;
		std::cout << "wordRNNHiddenSize = " << wordRNNHiddenSize << std::endl;
		std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;

		std::cout << "charEmbSize = " << charEmbSize << std::endl;
		std::cout << "bicharEmbSize = " << bicharEmbSize << std::endl;
		std::cout << "mapcharEmbSize = " << mapcharEmbSize << std::endl;
		std::cout << "charcontext = " << charcontext << std::endl;
		std::cout << "charHiddenSize = " << charHiddenSize << std::endl;
		std::cout << "charRNNHiddenSize = " << charRNNHiddenSize << std::endl;
		std::cout << "charEmbFineTune = " << charEmbFineTune << std::endl;
		std::cout << "bicharEmbFineTune = " << bicharEmbFineTune << std::endl;
		std::cout << "mapcharEmbFineTune = " << mapcharEmbFineTune << std::endl;
		std::cout << "stackHiddenSize = " << stackHiddenSize << std::endl;

		std::cout << "actionEmbSize = " << actionEmbSize << std::endl;
		std::cout << "actionNgram = " << actionNgram << std::endl;
		std::cout << "actionHiddenSize = " << actionHiddenSize << std::endl;
		std::cout << "actionRNNHiddenSize = " << actionRNNHiddenSize << std::endl;

		std::cout << "verboseIter = " << verboseIter << std::endl;
		std::cout << "saveItermediate = " << saveIntermediate << std::endl;
		std::cout << "train = " << train << std::endl;
		std::cout << "maxInstance = " << maxInstance << std::endl;
		std::cout << "finalHiddenSize = " << finalHiddenSize << std::endl;

		for (int idx = 0; idx < testFiles.size(); idx++) {
			std::cout << "testFile = " << testFiles[idx] << std::endl;
		}
		std::cout << "outBest = " << outBest << std::endl;
	}

	void load(const std::string& infile) {
		ifstream inf;
		inf.open(infile.c_str());
		vector<string> vecLine;
		while (1) {
			string strLine;
			if (!my_getline(inf, strLine)) {
				break;
			}
			if (strLine.empty())
				continue;
			vecLine.push_back(strLine);
		}
		inf.close();
		setOptions(vecLine);
	}

	void writeModel(LStream &outf) {

		WriteVector(outf, testFiles);
	    WriteString(outf, outBest);


	    WriteBinary(outf, wordCutOff);
	    WriteBinary(outf, featCutOff);
	    WriteBinary(outf, charCutOff);
	    WriteBinary(outf, bicharCutOff);
	    WriteBinary(outf, initRange);
	    WriteBinary(outf, maxIter);
	    WriteBinary(outf, batchSize);
	    WriteBinary(outf, adaEps);
	    WriteBinary(outf, adaAlpha);
	    WriteBinary(outf, regParameter);

	    WriteBinary(outf, dropProb);
	    WriteBinary(outf, delta);
	    WriteBinary(outf, clip);
	    WriteBinary(outf, oovRatio);
	    WriteBinary(outf, sepHiddenSize);
	    WriteBinary(outf, appHiddenSize);
	    WriteBinary(outf, wordEmbSize);
	    WriteBinary(outf, lengthEmbSize);
	    WriteBinary(outf, keycharEmbSize);
	    WriteBinary(outf, wordNgram);
	    WriteBinary(outf, wordHiddenSize);

	    WriteBinary(outf, wordRNNHiddenSize);
	    WriteBinary(outf, wordEmbFineTune);
	    WriteBinary(outf, charEmbSize);
	    WriteBinary(outf, bicharEmbSize);
	    WriteBinary(outf, mapcharEmbSize);
	    WriteBinary(outf, charcontext);
	    WriteBinary(outf, charHiddenSize);
	    WriteBinary(outf, charRNNHiddenSize);
	    WriteBinary(outf, charEmbFineTune);
	    WriteBinary(outf, bicharEmbFineTune);
	    WriteBinary(outf, mapcharEmbFineTune);
	    WriteBinary(outf, actionEmbSize);
	    WriteBinary(outf, actionNgram);

	    WriteBinary(outf, actionHiddenSize);
	    WriteBinary(outf, actionRNNHiddenSize);
	    WriteBinary(outf, stackHiddenSize);
	    WriteBinary(outf, verboseIter);
	    WriteBinary(outf, saveIntermediate);
	    WriteBinary(outf, train);
	    WriteBinary(outf, maxInstance);
	    WriteBinary(outf, finalHiddenSize);

	}

	void loadModel(LStream &inff) {

	}

};

#endif

