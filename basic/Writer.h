#ifndef _JST_WRITER_
#define _JST_WRITER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "Instance.h"

class Writer
{
public:
	Writer()
	{
	}
	virtual ~Writer()
	{
		if (m_outf.is_open()) m_outf.close();
	}

	inline int startWriting(const char *filename) {
		m_outf.open(filename);
		if (!m_outf) {
			cout << "Writerr::startWriting() open file err: " << filename << endl;
			return -1;
		}
		return 0;
	}

	inline void finishWriting() {
		m_outf.close();
	}

	virtual int write(const Instance *pInstance) = 0;
	virtual int write(const vector<string> &curWords) = 0;
protected:
	ofstream m_outf;
};

#endif

