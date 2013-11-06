/*
 * SpMat.h
 *
 *  Created on: Nov 5, 2013
 *      Author: stefan
 */

#ifndef SPMAT_H_
#define SPMAT_H_

#include <vector>

class Vec;

class SpMat
{
public:
	struct entry
	{
		unsigned int i;
		unsigned int j;
		double val;
	};

public:
	SpMat( unsigned int m, unsigned int n );
	~SpMat();

	void addEntry( unsigned int i, unsigned int j, double val );
	const std::vector<entry>& getEntries() const;
	void multiplyVector( const Vec& vec, Vec& res ) const;
	void diag( double val, int offset );

private:
	std::vector<entry> entries;
	unsigned int m;
	unsigned int n;
};

#endif /* SPMAT_H_ */
