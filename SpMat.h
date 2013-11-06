/*
 * SpMat.h
 *
 *  Created on: Nov 5, 2013
 *      Author: stefan
 */

#ifndef SPMAT_H_
#define SPMAT_H_

#include <vector>

class SpMat
{
public:
	struct entry
	{
		int i;
		int j;
		double val;
	};

public:
	SpMat();
	~SpMat();

	void addEntry( int i, int j, double val );
	const std::vector<entry>& getEntries();

private:
	std::vector<entry> entries;
	int m;
	int n;
};

#endif /* SPMAT_H_ */
