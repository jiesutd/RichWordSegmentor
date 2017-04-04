#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3L.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader: public Reader {
public:
  InstanceReader() {
  }
  ~InstanceReader() {
  }

  Instance *getNext() {
    m_instance.clear();
    string strLine;
    while (1) {
      if (!my_getline(m_inf, strLine)) {
        break;
      }
      if (!strLine.empty())
        break;
    }

    vector<string> wordInfo;
    split_bychar(strLine, wordInfo, ' ');

    string sentence = "";
    for (int i = 0; i < wordInfo.size(); ++i) {
      sentence = sentence + wordInfo[i];
    }

    vector<string> charInfo;
    getCharactersFromUTF8String(sentence, charInfo);

    m_instance.allocate(wordInfo.size(), charInfo.size());
    for (int i = 0; i < wordInfo.size(); ++i) {
      m_instance.words[i] = wordInfo[i];
    }
    for (int i = 0; i < charInfo.size(); ++i) {
      m_instance.chars[i] = charInfo[i];
    }

    return &m_instance;
  }
};

#endif

