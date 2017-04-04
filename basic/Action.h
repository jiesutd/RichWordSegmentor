/*
 * CAction.h
 *
 *  Created on: Oct 6, 2015
 *      Author: mszhang
 */

#ifndef BASIC_CAction_H_
#define BASIC_CAction_H_



/*===============================================================
 *
 * scored actions
 *
 *==============================================================*/
// for segmentation, there are only threee valid operations
class CAction {
public:
  enum CODE {NO_ACTION=0, SEP=1, APP=2, FIN=3, IDLE=4};
  unsigned long _code;

public:
   CAction() : _code(0){
   }

   CAction(int code) : _code(code){
   }

   CAction(const CAction &ac) : _code(ac._code){
   }

public:
   inline void clear() { _code=0; }

   inline void set(int code){
     _code = code;
   }

   inline bool isNone() const { return _code==NO_ACTION; }
   inline bool isSeparate() const { return _code==SEP; }
   inline bool isAppend() const { return _code==APP; }
   inline bool isFinish() const { return _code==FIN; }
   inline bool isIdle() const { return _code>=IDLE; }

public:
   inline std::string str() const {
     if (isNone()) { return "NONE"; }
     if (isIdle()) { return "IDLE"; }
     if (isSeparate()) { return "SEP"; }
     if (isAppend()) { return "APP"; }
     if (isFinish()) { return "FIN"; }
     return "IDLE";
   }

public:
   const unsigned long &code() const {return _code;}
   const unsigned long &hash() const {return _code;}
   bool operator == (const CAction &a1) const { return _code == a1._code; }
   bool operator != (const CAction &a1) const { return _code != a1._code; }
   bool operator < (const CAction &a1) const { return _code < a1._code; }
   bool operator > (const CAction &a1) const { return _code > a1._code; }

};


inline std::istream & operator >> (std::istream &is, CAction &action) {
  std::string tmp;
  is >> tmp;
  if (tmp=="NONE") {
    action.clear();
  }
  else if(tmp=="IDLE"){
    action._code = CAction::IDLE;
  }
  else if(tmp=="SEP"){
    action._code = CAction::SEP;
  }
  else if(tmp=="APP"){
    action._code = CAction::APP;
  }
  else if(tmp=="FIN"){
    action._code = CAction::FIN;
  }

  return is;
}


inline std::ostream & operator << (std::ostream &os, const CAction &action) {
   os << action.str();
   return os;
}


#endif /* BASIC_CAction_H_ */
