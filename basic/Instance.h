#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "N3L.h"
#include "Metric.h"

using namespace std;

class Instance {
public:
	Instance() {
	}
	~Instance() {
	}

	int wordsize() const {
		return words.size();
	}

  int charsize() const {
    return chars.size();
  }

	void clear() {
		words.clear();
		chars.clear();
	}

	void allocate(int length, int charLength) {
		clear();
		words.resize(length);
		chars.resize(charLength);
	}

	void copyValuesFrom(const Instance& anInstance) {
		allocate(anInstance.wordsize(), anInstance.charsize());
		for (int i = 0; i < anInstance.wordsize(); i++) {
			words[i] = anInstance.words[i];
		}
    for (int i = 0; i < anInstance.charsize(); i++) {
      chars[i] = anInstance.chars[i];
    }
	}


	void evaluate(const vector<string>& resulted_segs, Metric& eval) const {
	  hash_set<string> golds;
	  getSegIndexes(words, golds);

	  hash_set<string> preds;
	  getSegIndexes(resulted_segs, preds);

    hash_set<string>::iterator iter;
    eval.overall_label_count += golds.size();
    eval.predicated_label_count += preds.size();
    for (iter = preds.begin(); iter != preds.end(); iter++) {
      if (golds.find(*iter) != golds.end()) {
        eval.correct_label_count++;
      }
    }

	}

	void getSegIndexes(const vector<string>& segs, hash_set<string>& segIndexes) const{
	  segIndexes.clear();
	  int idx = 0, idy = 0;
	  string curWord = "";
	  int beginId = 0;
	  while(idx < chars.size() && idy < segs.size()){
	    curWord = curWord + chars[idx];
	    if(curWord.length() == segs[idy].length()){
        stringstream ss;
        ss << "[" << beginId << "," << idx << "]";
        segIndexes.insert(ss.str());
        idy++;
        beginId = idx+1;
        curWord = "";
	    }
	    idx++;
	  }

	  if(idx != chars.size() || idy != segs.size()){
	    std::cout << "error segs, please check" << std::endl;
	  }
	}

public:
	vector<string> words;
	vector<string> chars;
};

#endif

