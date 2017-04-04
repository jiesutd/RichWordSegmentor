#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "Instance.h"

class Reader
{
public:
	Reader()
	{
	}

	virtual ~Reader()
	{
		if (m_inf.is_open()) m_inf.close();
	}
	int startReading(const char *filename) {
		if (m_inf.is_open()) {
			m_inf.close();
			m_inf.clear();
		}
		m_inf.open(filename);

    if (!m_inf.is_open()) {
			cout << "Reader::startReading() open file err: " << filename << endl;
			return -1;
		}

		return 0;
	}

	void finishReading() {
		if (m_inf.is_open()) {
			m_inf.close();
			m_inf.clear();
		}
	}
	virtual Instance *getNext() = 0;
protected:
	ifstream m_inf;

	int m_numInstance;

	Instance m_instance;
};

#endif

