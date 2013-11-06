/*
 * Vec.h
 *
 *  Created on: Nov 6, 2013
 *      Author: stefan
 */

#ifndef VEC_H_
#define VEC_H_

class Vec
{
public:
	Vec( unsigned int n );
	~Vec();

	double* getValues();
	const double* getValues() const;
	unsigned int getSize() const;
	void zeros();
	void print();

private:
	double* val;
	unsigned int n;
};

#endif /* VEC_H_ */
