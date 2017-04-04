/*
 * State.h
 *
 *  Created on: Oct 1, 2015
 *      Author: mszhang
 */

#ifndef SEG_STATE_H_
#define SEG_STATE_H_

#include "Feature.h"
#include "Action.h"

class CStateItem {
public:
  std::string _strlastWord;
  int _lastWordStart;
  int _lastWordEnd;
  const CStateItem *_prevStackState;
  const CStateItem *_prevState;
  int _nextPosition;

  const std::vector<std::string> *_pCharacters;
  int _characterSize;

  CAction _lastAction;
  Feature _curFeat;
  dtype _score;
  int _wordnum;

public:
  CStateItem() {
    _strlastWord = "";
    _lastWordStart = -1;
    _lastWordEnd = -1;
    _prevStackState = 0;
    _prevState = 0;
    _nextPosition = 0;
    _pCharacters = 0;
    _characterSize = 0;
    _lastAction.clear();
    _curFeat.clear();
    _score = 0.0;
    _wordnum = 0;
  }

  CStateItem(const std::vector<std::string>* pCharacters) {
    _strlastWord = "";
    _lastWordStart = -1;
    _lastWordEnd = -1;
    _prevStackState = 0;
    _prevState = 0;
    _nextPosition = 0;
    _pCharacters = pCharacters;
    _characterSize = pCharacters->size();
    _lastAction.clear();
    _curFeat.clear();
    _score = 0.0;
    _wordnum = 0;
  }

  virtual ~CStateItem(){
	  clear();
  }

  void initSentence(const std::vector<std::string>* pCharacters) {
    _pCharacters = pCharacters;
    _characterSize = pCharacters->size();
  }

  void clear() {
    _strlastWord = "";
    _lastWordStart = -1;
    _lastWordEnd = -1;
    _prevStackState = 0;
    _prevState = 0;
    _nextPosition = 0;
    _lastAction.clear();
    _curFeat.clear();
    _score = 0.0;
    _wordnum = 0;
  }

  void copyState(const CStateItem* from) {
    _strlastWord = from->_strlastWord;
    _lastWordStart = from->_lastWordStart;
    _lastWordEnd = from->_lastWordEnd;
    _prevStackState = from->_prevStackState;
    _prevState = from->_prevState;
    _nextPosition = from->_nextPosition;
    _pCharacters = from->_pCharacters;
    _characterSize = from->_characterSize;
    _lastAction = from->_lastAction;
    _curFeat.copy(from->_curFeat);
    _score = from->_score;
    _wordnum = from->_wordnum;
  }

  const CStateItem* getPrevStackState() const{
    return _prevStackState;
  }

  const CStateItem* getPrevState() const{
    return _prevState;
  }

  std::string getLastWord() {
    return _strlastWord;
  }

public:
  //only assign context
  void separate(CStateItem* next) const{
    if (_nextPosition >= _characterSize) {
      std::cout << "separate error" << std::endl;
      return;
    }
    next->_strlastWord = (*_pCharacters)[_nextPosition];
    next->_lastWordStart = _nextPosition;
    next->_lastWordEnd = _nextPosition;
    next->_prevStackState = this;
    next->_prevState = this;
    next->_nextPosition = _nextPosition + 1;
    next->_pCharacters = _pCharacters;
    next->_characterSize = _characterSize;
    next->_wordnum = _wordnum + 1;
    next->_lastAction.set(CAction::SEP);
  }

  //only assign context
  void finish(CStateItem* next) const{
    if (_nextPosition != _characterSize) {
      std::cout << "finish error" << std::endl;
      return;
    }
    next->_strlastWord = _strlastWord;
    next->_lastWordStart = _lastWordStart;
    next->_lastWordEnd = _lastWordEnd;
    next->_prevStackState = _prevStackState;
    next->_prevState = this;
    next->_nextPosition = _nextPosition + 1;
    next->_pCharacters = _pCharacters;
    next->_characterSize = _characterSize;
    next->_wordnum = _wordnum + 1;
    next->_lastAction.set(CAction::FIN);
  }

  //only assign context
  void append(CStateItem* next) const{
    if (_nextPosition >= _characterSize) {
      std::cout << "append error" << std::endl;
      return;
    }
    next->_strlastWord = _strlastWord + (*_pCharacters)[_nextPosition];
    next->_lastWordStart = _lastWordStart;
    next->_lastWordEnd = _nextPosition;
    next->_prevStackState = _prevStackState;
    next->_prevState = this;
    next->_nextPosition = _nextPosition + 1;
    next->_pCharacters = _pCharacters;
    next->_characterSize = _characterSize;
    next->_wordnum = _wordnum;
    next->_lastAction.set(CAction::APP);
  }

  void move(CStateItem* next, const CAction& ac) const{
    if (ac.isAppend()) {
      append(next);
    } else if (ac.isSeparate()) {
      separate(next);
    } else if (ac.isFinish()) {
      finish(next);
    } else {
      std::cout << "error action" << std::endl;
    }
  }

  bool IsTerminated() const {
    if (_lastAction.isFinish())
      return true;
    return false;
  }

  //partial results
  void getSegResults(std::vector<std::string>& words) const {
    words.clear();
    words.insert(words.begin(), _strlastWord);
    const CStateItem *prevStackState = _prevStackState;
    while (prevStackState != 0 && prevStackState->_wordnum > 0) {
      words.insert(words.begin(), prevStackState->_strlastWord);
      prevStackState = prevStackState->_prevStackState;
    }
  }


  void getGoldAction(const std::vector<std::string>& segments, CAction& ac) const {
    if (_nextPosition == _characterSize) {
      ac.set(CAction::FIN);
      return;
    }
    if (_nextPosition == 0) {
      ac.set(CAction::SEP);
      return;
    }

    if (_nextPosition > 0 && _nextPosition < _characterSize) {
      // should have a check here to see whether the words are match, but I did not do it here
      if (_strlastWord.length() == segments[_wordnum-1].length()) {
        ac.set(CAction::SEP);
        return;
      } else {
        ac.set(CAction::APP);
        return;
      }
    }

    ac.set(CAction::NO_ACTION);
    return;
  }

  // we did not judge whether history actions are match with current state.
  void getGoldAction(const CStateItem* goldState, CAction& ac) const{
    if (_nextPosition > goldState->_nextPosition || _nextPosition < 0) {
      ac.set(CAction::NO_ACTION);
      return;
    }
    const CStateItem *prevState = goldState->_prevState;
    CAction curAction = goldState->_lastAction;
    while (_nextPosition < prevState->_nextPosition) {
      curAction = prevState->_lastAction;
      prevState = prevState->_prevState;
    }
    return ac.set(curAction._code);
  }

  void getCandidateActions(vector<CAction> & actions) const{
    actions.clear();
    static CAction ac;
    if(_nextPosition == 0){
      ac.set(CAction::SEP);
      actions.push_back(ac);
    }
    else if(_nextPosition == _characterSize){
      ac.set(CAction::FIN);
      actions.push_back(ac);
    }
    else if(_nextPosition > 0 && _nextPosition < _characterSize){
      ac.set(CAction::SEP);
      actions.push_back(ac);
      ac.set(CAction::APP);
      actions.push_back(ac);
    }
    else{

    }

  }

  inline std::string str() const{
    stringstream curoutstr;

    curoutstr << "score: " << _score << " ";
    curoutstr << "seg:";
    std::vector<std::string> words;
    getSegResults(words);
    for(int idx = 0; idx < words.size(); idx++){
      curoutstr << " " << words[idx];
    }

    return curoutstr.str();
  }

};


class CScoredStateAction {
public:
  CAction action;
  const CStateItem *item;
  dtype score;
  Feature feat;

public:
  CScoredStateAction() :
      item(0), action(-1), score(0) {
    feat.setFeatureFormat(false);
    feat.clear();
  }


public:
  bool operator <(const CScoredStateAction &a1) const {
    return score < a1.score;
  }
  bool operator >(const CScoredStateAction &a1) const {
    return score > a1.score;
  }
  bool operator <=(const CScoredStateAction &a1) const {
    return score <= a1.score;
  }
  bool operator >=(const CScoredStateAction &a1) const {
    return score >= a1.score;
  }


};

class CScoredStateAction_Compare {
public:
  int operator()(const CScoredStateAction &o1, const CScoredStateAction &o2) const {

    if (o1.score < o2.score)
      return -1;
    else if (o1.score > o2.score)
      return 1;
    else
      return 0;
  }
};


#endif /* SEG_STATE_H_ */
