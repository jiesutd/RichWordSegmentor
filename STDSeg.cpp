/*
 * Segmentor.cpp
 *
 *  Created on: Oct 23, 2015
 *      Author: mszhang
 */

#include "STDSeg.h"

#include "Argument_helper.h"

Segmentor::Segmentor() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  paddingtag = "-padding-";
  seperateKey = "#";
}

Segmentor::~Segmentor() {
  // TODO Auto-generated destructor stub
	m_classifier.release();
}

// all linear features are extracted from positive examples
int Segmentor::createAlphabet(const vector<Instance>& vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance = vecInsts.size();

  hash_map<string, int> char_stat;
  hash_map<string, int> bichar_stat;
  hash_map<string, int> action_stat;
  hash_map<string, int> feat_stat;
  hash_map<string, int> word_stat;

  assert(numInstance > 0);

  static Metric eval;
#if USE_CUDA==1
  static CStateItem<gpu> state[m_classifier.MAX_SENTENCE_SIZE];
#else
  static CStateItem<cpu> state[m_classifier.MAX_SENTENCE_SIZE];
#endif
  static Feature feat;
  static vector<string> output;
  static CAction answer;
  static int actionNum;
  m_classifier.initAlphabet();
  eval.reset();
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance &instance = vecInsts[numInstance];
        
    for (int idx = 0; idx < instance.wordsize(); idx++) {
      m_word_stat[normalize_to_lowerwithdigit(instance.words[idx])]++;
    }
    
    for(int distance = 1; distance <= 2; distance++) {
      for (int idx = 0; idx < instance.charsize(); idx++) {
      	if(idx + distance >= instance.charsize()) break;
      	string curWord = instance.chars[idx];
      	for(int idz = 1; idz < distance; idz++){
      		curWord= curWord + instance.chars[idx+idz];
      	}
      	curWord = normalize_to_lowerwithdigit(curWord);
      	word_stat[curWord]++;
      }
    }


    for (int idx = 0; idx < instance.charsize(); idx++) {
      char_stat[instance.chars[idx]]++;
    }
    for (int idx = 0; idx < instance.charsize() - 1; idx++) {
      bichar_stat[instance.chars[idx] + instance.chars[idx + 1]]++;
    }
    bichar_stat[instance.chars[instance.charsize() - 1] + m_classifier.fe.nullkey]++;
    bichar_stat[m_classifier.fe.nullkey + instance.chars[0]]++;
    actionNum = 0;
    state[actionNum].initSentence(&instance.chars);
    state[actionNum].clear();

    while (!state[actionNum].IsTerminated()) {
      state[actionNum].getGoldAction(instance.words, answer);
      action_stat[answer.str()]++;

      m_classifier.extractFeature(state+actionNum, answer, feat);
      for (int idx = 0; idx < feat._strSparseFeat.size(); idx++) {
        feat_stat[feat._strSparseFeat[idx]]++;
      }
      state[actionNum].move(state+actionNum+1, answer);
      actionNum++;
    }

    if(actionNum-1 != instance.charsize()) {
      std::cout << "action number is not correct, please check" << std::endl;
    }
    state[actionNum].getSegResults(output);

    instance.evaluate(output, eval);

    if (!eval.bIdentical()) {
      std::cout << "error state conversion!" << std::endl;
      exit(0);
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }
	
	int discount = 2;
	hash_map<string, int>::iterator word_iter;
	for (word_iter = word_stat.begin(); word_iter != word_stat.end(); word_iter++) {
		if (word_iter->second > discount && m_word_stat.find(word_iter->first) == m_word_stat.end()) {
			m_word_stat[word_iter->first] = word_iter->second - discount;
		}
	}

  m_classifier.addToActionAlphabet(action_stat);
  m_classifier.addToWordAlphabet(m_word_stat);
  m_classifier.addToCharAlphabet(char_stat, m_options.charEmbFineTune ? m_options.charCutOff : 0);
  m_classifier.addToBiCharAlphabet(bichar_stat, m_options.bicharEmbFineTune ? m_options.bicharCutOff : 0);
  m_classifier.addToFeatureAlphabet(feat_stat, m_options.featCutOff);

  cout << numInstance << " " << endl;
  cout << "Action num: " << m_classifier.fe._actionAlphabet.size() << endl;
  cout << "Total word num: " << m_word_stat.size() << endl;
  cout << "Total char num: " << char_stat.size() << endl;
  cout << "Total bichar num: " << bichar_stat.size() << endl;
  cout << "Total feat num: " << feat_stat.size() << endl;

  cout << "Remain word num: " << m_classifier.fe._wordAlphabet.size() << endl;
  cout << "Remain char num: " << m_classifier.fe._charAlphabet.size() << endl;
  cout << "Remain bichar num: " << m_classifier.fe._bicharAlphabet.size() << endl;
  cout << "Remain feat num: " << m_classifier.fe._featAlphabet.size() << endl;

  //m_classifier.setFeatureCollectionState(false);

  return 0;
}

int Segmentor::addTestWordAlpha(const vector<Instance>& vecInsts) {
  cout << "Add test Alphabet..." << endl;

  hash_map<string, int> char_stat;
  hash_map<string, int> bichar_stat;
  int numInstance;

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance &instance = vecInsts[numInstance];

    for (int idx = 0; idx < instance.charsize(); idx++) {
      char_stat[instance.chars[idx]]++;
    }
    for (int idx = 0; idx < instance.charsize() - 1; idx++) {
      bichar_stat[instance.chars[idx] + instance.chars[idx + 1]]++;
    }
    bichar_stat[instance.chars[instance.charsize() - 1] + m_classifier.fe.nullkey]++;
    bichar_stat[m_classifier.fe.nullkey + instance.chars[0]]++;

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  m_classifier.addToCharAlphabet(char_stat, 0);
  m_classifier.addToBiCharAlphabet(bichar_stat, 0);

  cout << "Remain char num: " << m_classifier.fe._charAlphabet.size() << endl;
  cout << "Remain bichar num: " << m_classifier.fe._bicharAlphabet.size() << endl;

  return 0;
}

int Segmentor::allWordAlphaEmb(const string& inFile, NRMat<dtype>& emb) {
  cout << "All word  alphabet and emb creating..." << endl;

  hash_map<string, int> word_stat;
  
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;
  static vector<string> vecInfo;
  vector<string> allLines;

  int wordDim = 0;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty()){
      split_bychar(strLine, vecInfo, ' '); 
      if(wordDim == 0){
      	wordDim = vecInfo.size() - 1;
      	std::cout << "allword embedding dim is " << wordDim << std::endl;
      }
      curWord = normalize_to_lowerwithdigit(vecInfo[0]);
      word_stat[curWord]++;
      allLines.push_back(strLine);
    }
  }

  m_classifier.addToAllWordAlphabet(word_stat);
  cout << "Remain all word num: " << m_classifier.fe._allwordAlphabet.size() << endl;
  
  emb.resize(m_classifier.fe._allwordAlphabet.size(), wordDim);
  emb = 0.0;
  
  int unknownId = m_classifier.fe._allwordAlphabet.from_string(m_classifier.fe.unknownkey);
  dtype sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  for (int idx = 0; idx < wordDim; idx++) {
    sum[idx] = 0.0;
  }
  
  for(int idx = 0; idx < allLines.size(); idx++){
    split_bychar(allLines[idx], vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_classifier.fe._allwordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;

      for (int idx = 0; idx < wordDim; idx++) {
        dtype curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] += curValue;
        emb[wordId][idx] += curValue;
      }
    }
    else{
    	std::cout << "read all word embedding strange...." << std::endl;
    }	
  	
  }  
  
  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      emb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  return 0;
}


void Segmentor::getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions){
  vecActions.clear();

  static Metric eval;
#if USE_CUDA==1
  static CStateItem<gpu> state[m_classifier.MAX_SENTENCE_SIZE];
#else
  static CStateItem<cpu> state[m_classifier.MAX_SENTENCE_SIZE];
#endif
  static vector<string> output;
  static CAction answer;
  eval.reset();
  static int numInstance, actionNum;
  vecActions.resize(vecInsts.size());
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance &instance = vecInsts[numInstance];

    actionNum = 0;
    state[actionNum].initSentence(&instance.chars);
    state[actionNum].clear();

    while (!state[actionNum].IsTerminated()) {
      state[actionNum].getGoldAction(instance.words, answer);
      vecActions[numInstance].push_back(answer);
      state[actionNum].move(state+actionNum+1, answer);
      actionNum++;
    }

    if(actionNum-1 != instance.charsize()) {
      std::cout << "action number is not correct, please check" << std::endl;
    }
    state[actionNum].getSegResults(output);

    instance.evaluate(output, eval);

    if (!eval.bIdentical()) {
      std::cout << "error state conversion!" << std::endl;
      exit(0);
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }
}


void Segmentor::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
    const string& wordEmbFile, const string& charEmbFile, const string& bicharEmbFile, const string& layerFile, const string& numberlayerFile) {
  clock_t train_start_time = clock();
  if (optionFile != "")
    m_options.load(optionFile);

  m_options.showOptions();
  vector<Instance> trainInsts, devInsts, testInsts;
  m_pipe.readInstances(trainFile, trainInsts, m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);
  if (devFile != "")
    m_pipe.readInstances(devFile, devInsts, m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);
  if (testFile != "")
    m_pipe.readInstances(testFile, testInsts, m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);

  vector<vector<Instance> > otherInsts(m_options.testFiles.size());
  for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
    m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);
  }

  createAlphabet(trainInsts);

  addTestWordAlpha(devInsts);
  addTestWordAlpha(testInsts);
    
  NRMat<dtype> wordEmb, allwordEmb;
  if (wordEmbFile != "") {    
    allWordAlphaEmb(wordEmbFile, allwordEmb);
  } else {
    std::cout << "must not be here, allword must be pretrained." << std::endl;
  }
  wordEmb.resize(m_classifier.fe._wordAlphabet.size(), m_options.wordEmbSize);
  wordEmb.randu(1000);

  cout << "word emb dim is " << wordEmb.ncols() << std::endl;

  NRMat<dtype> charEmb;
  if (charEmbFile != "") {
    readEmbeddings(m_classifier.fe._charAlphabet, charEmbFile, charEmb);
  } else {
    charEmb.resize(m_classifier.fe._charAlphabet.size(), m_options.charEmbSize);
    charEmb.randu(2000);
  }

  cout << "char emb dim is " << charEmb.ncols() << std::endl;

  NRMat<dtype> bicharEmb;
  if (bicharEmbFile != "") {
    readEmbeddings(m_classifier.fe._bicharAlphabet, bicharEmbFile, bicharEmb);
  } else {
    bicharEmb.resize(m_classifier.fe._bicharAlphabet.size(), m_options.bicharEmbSize);
    bicharEmb.randu(2000);
  }

  cout << "bichar emb dim is " << bicharEmb.ncols() << std::endl;

  NRMat<dtype> lengthEmb;
  lengthEmb.resize(6, m_options.lengthEmbSize);
  lengthEmb.randu(3000);

  cout << "length emb dim is " << lengthEmb.ncols() << std::endl;

  m_classifier.setDropValue(m_options.dropProb);
  m_classifier.init(wordEmb, allwordEmb, lengthEmb, m_options.wordNgram, m_options.wordHiddenSize, m_options.wordRNNHiddenSize,
      charEmb, bicharEmb, m_options.charcontext, m_options.charHiddenSize, m_options.charRNNHiddenSize,
      m_options.stackHiddenSize, m_options.finalHiddenSize, m_options.delta);

  m_classifier.setOOVFreq(m_options.wordCutOff);
  m_classifier.setOOVRatio(m_options.oovRatio);
  m_classifier.setWordFreq(m_word_stat);


  if (layerFile != "") {
    m_classifier.loadInitialLayer(layerFile);
  }
  
  if (numberlayerFile != "") {
    m_classifier.loadInitialLayerNumber(numberlayerFile);
  }

  vector<vector<CAction> > trainInstGoldactions;
  getGoldActions(trainInsts, trainInstGoldactions);
  double bestFmeasure = 0;

  int inputSize = trainInsts.size();

  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval, metric_dev, metric_test;

  // int maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);
  int maxIter = m_options.maxIter;
  int oneIterMaxRound = (inputSize + m_options.batchSize -1) / m_options.batchSize;
  std::cout << "maxIter = " << maxIter << std::endl;
  int devNum = devInsts.size(), testNum = testInsts.size();

  static vector<vector<string> > decodeInstResults;
  static vector<string> curDecodeInst;
  static bool bCurIterBetter;
  static vector<vector<string> > subInstances;
  static vector<vector<CAction> > subInstGoldActions;
  std::cout << "Train init finished. Total time taken is: " << double(clock() - train_start_time) / CLOCKS_PER_SEC << "s"<< std::endl;
  for (int iter = 0; iter < maxIter; ++iter) {
    clock_t train_iter_start_time = clock();
    std::cout << "##### Iteration " << iter << std::endl;
    srand(iter);
    random_shuffle(indexes.begin(), indexes.end());
    std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;    
    bool bEvaluate = false;
    if(m_options.batchSize == 1){
      eval.reset();
      bEvaluate = true;
      clock_t batch_start_time = clock();
      for (int idy = 0; idy < inputSize; idy++) {
        subInstances.clear();
        subInstGoldActions.clear();
        subInstances.push_back(trainInsts[indexes[idy]].chars);
        subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
          
        double cost = m_classifier.train(subInstances, subInstGoldActions);
  
        eval.overall_label_count += m_classifier._eval.overall_label_count;
        eval.correct_label_count += m_classifier._eval.correct_label_count;
  
        if ((idy + 1) % (m_options.verboseIter*10) == 0) {
          std::cout << "current: " << idy + 1<< ", Time = "<<double(clock() - batch_start_time) / CLOCKS_PER_SEC  << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()  << std::endl;
          batch_start_time = clock();
        }
        m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps, m_options.clip);
      }
      std::cout << "current: " << iter + 1  << ", Correct(%) = " << eval.getAccuracy() << std::endl;
    }
    else{
      if(iter == 0)eval.reset();
      clock_t batch_start_time = clock();
      subInstances.clear();
      subInstGoldActions.clear();
      for (int idy = 0; idy < m_options.batchSize; idy++) {
        subInstances.push_back(trainInsts[indexes[idy]].chars);
        subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);       
      }
      double cost = m_classifier.train(subInstances, subInstGoldActions);

      eval.overall_label_count += m_classifier._eval.overall_label_count;
      eval.correct_label_count += m_classifier._eval.correct_label_count;
      
      if ((iter + 1) % (m_options.verboseIter) == 0) {
        std::cout << "current: " << iter + 1 << ", Time = "<<double(clock() - batch_start_time) / CLOCKS_PER_SEC << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()  << std::endl;
        batch_start_time = clock();
        eval.reset();
        bEvaluate = true;
      }
      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps, m_options.clip);
    }
    clock_t train_iter_end_time = clock();
    std::cout << "Iter "<< iter << " finished. Total time taken is: " << double(train_iter_end_time- train_iter_start_time) / CLOCKS_PER_SEC<< "s" << std::endl;
    if (bEvaluate && devNum > 0) {
      bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_dev.reset();
      for (int idx = 0; idx < devInsts.size(); idx++) {
        predict(devInsts[idx], curDecodeInst);
        devInsts[idx].evaluate(curDecodeInst, metric_dev);
        if (!m_options.outBest.empty()) {
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "dev:" << std::endl;
      metric_dev.print();

      if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestFmeasure) {
        m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }
    clock_t decode_dev_end_time = clock();
    std::cout << "Decode dev finished. Total time taken is: " << double(decode_dev_end_time-train_iter_end_time) / CLOCKS_PER_SEC<< "s" << std::endl;
      if (testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idx = 0; idx < testInsts.size(); idx++) {
          predict(testInsts[idx], curDecodeInst);
          testInsts[idx].evaluate(curDecodeInst, metric_test);
          if (bCurIterBetter && !m_options.outBest.empty()) {
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
        }
      }
      clock_t decode_test_end_time = clock();
      std::cout << "Decode test finished. Total time taken is: " << double(decode_test_end_time- decode_dev_end_time) / CLOCKS_PER_SEC<< "s" << std::endl;
      for (int idx = 0; idx < otherInsts.size(); idx++) {
        std::cout << "processing " << m_options.testFiles[idx] << std::endl;
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
          predict(otherInsts[idx][idy], curDecodeInst);
          otherInsts[idx][idy].evaluate(curDecodeInst, metric_test);
          if (bCurIterBetter && !m_options.outBest.empty()) {
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
        }
      }

      if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestFmeasure) {
        std::cout << "Exceeds best previous DIS of " << bestFmeasure << ". Saving model file.." << std::endl;
        bestFmeasure = metric_dev.getAccuracy();
        writeModelFile(modelFile);
      }
    }
  }
}



void Segmentor::predict(const Instance& input, vector<string>& output) {
  m_classifier.decode(input.chars, output);
}


void Segmentor::test(const string& testFile, const string& outputFile, const string& modelFile) {
  loadModelFile(modelFile);
  vector<Instance> testInsts;
  m_pipe.readInstances(testFile, testInsts, m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);
  cout << "instance num:"<< testInsts.size()<<endl;

  vector<vector<string> > testInstResults(testInsts.size());
  Metric metric_test;
  metric_test.reset();
  std::ofstream os(outputFile.c_str());
  clock_t last_decode_clock = clock();
  for (int idx = 0; idx < testInsts.size(); idx++) {
    if (idx % 1000 == 0) {
      cout << "decoding instance:"<< idx << ", Time cost:"<< double(clock() - last_decode_clock) / CLOCKS_PER_SEC<< endl;
      last_decode_clock = clock();
    }
    vector<string> result_labels;
    predict(testInsts[idx], testInstResults[idx]);
    testInsts[idx].evaluate(testInstResults[idx], metric_test);
    for(int idy = 0; idy < testInstResults[idx].size(); idy++){
      os << testInstResults[idx][idy] << " ";
    }
    os << std::endl;
  }
  std::cout << "test:" << std::endl;
  metric_test.print();
  os.close();

  // for (int idx = 0; idx < testInsts.size(); idx++) {
  //   for(int idy = 0; idy < testInstResults[idx].size(); idy++){
  //     os << testInstResults[idx][idy] << " ";
  //   }
  //   os << std::endl;
  // }
  
}


void Segmentor::readEmbeddings(Alphabet &alpha, const string& inFile, NRMat<dtype>& emb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = alpha.from_string(m_classifier.fe.unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;

  std::cout << "embedding dim is " << wordDim << std::endl;

  emb.resize(alpha.size(), wordDim);
  emb = 0.0;
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = alpha.from_string(curWord);
  hash_set<int> indexers;
  dtype sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      dtype curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      emb[wordId][idx] = curValue;
    }

  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = alpha.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        dtype curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] += curValue;
        emb[wordId][idx] += curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      emb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < alpha.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        emb[id][idx] = emb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << alpha.size() << ", embedding oov ratio is " << oovWords * 1.0 / alpha.size()
      << std::endl;

}


void Segmentor::loadModelFile(const string& inputModelFile) {
  std::cout << "Start load model from file: " << inputModelFile << std::endl;
  LStream inf(inputModelFile, "rb");
  ReadString(inf, nullkey);
  ReadString(inf, unknownkey);
  ReadString(inf, paddingtag);
  ReadString(inf, seperateKey);
  m_classifier.loadModel(inf);
  m_options.loadModel(inf);
  std::cout << "Model has been loaded from file: " << inputModelFile << std::endl;
  m_options.showOptions();
}

void Segmentor::writeModelFile(const string& outputModelFile) {
  std::cout << "Start write model to file: " << outputModelFile << std::endl;
  LStream outf(outputModelFile, "w+");
  WriteString(outf, nullkey);
  WriteString(outf, unknownkey);
  WriteString(outf, paddingtag);
  WriteString(outf, seperateKey);
  m_classifier.writeModel(outf);
  m_options.writeModel(outf);
  std::cout << "Model has been written in file: " << outputModelFile << std::endl;
}

int main(int argc, char* argv[]) {
  std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
  std::string wordEmbFile = "", charEmbFile = "", bicharEmbFile = "",mapcharEmbFile = "", optionFile = "";
  std::string outputFile = "";
  std::string layerFile = "";
  std::string numberlayerFile = "";
  bool bTrain = false;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
  ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
  ah.new_named_string("test", "testCorpus", "named_string",
      "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
  ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
  ah.new_named_string("bichar", "bicharEmbFile", "named_string", "pretrained bichar embedding file to train a model, optional when training", bicharEmbFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
  ah.new_named_string("layer", "layerFile", "named_string", "layer file pretrained for other task", layerFile);
  ah.new_named_string("numlayer", "numberlayerFile", "named_string", "number format layer file pretrained for other task", numberlayerFile);

  ah.process(argc, argv);

  Segmentor segmentor;
  if (bTrain) {
    segmentor.train(trainFile, devFile, testFile, modelFile, optionFile, wordEmbFile, charEmbFile, bicharEmbFile, layerFile, numberlayerFile);
  } else {
    segmentor.test(testFile, outputFile, modelFile);
  }

  //test(argv);
  //ah.write_values(std::cout);

}
