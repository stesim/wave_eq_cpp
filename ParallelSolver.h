#pragma once

#include "Solver.h"
#include <thread>
#include <mutex>
#include <vector>

class ParallelSolver : public Solver
{
private:
	struct Params
	{
		arma::vec** PZ;
		arma::vec** PW;
		arma::sp_mat* M;
		double l2;
		double dt2;
		unsigned int nsteps;
	};

public:
	ParallelSolver();
	virtual ~ParallelSolver();

	virtual void solve(
		double L,
		unsigned int N,
		unsigned int n,
		double T,
		SpacialFunction funu0,
		SpacialFunction funu1,
		SpaciotemporalFunction funsol,
		arma::vec& x,
		arma::vec& numSol,
		arma::vec* exactSol,
		arma::vec* error );

public:
	static unsigned int s_uiDefaultNumThreads;

private:
	/*
	* Generate the finite differences matrix with split off main diagonal.
	*/
	static void genFDMatrices(
			unsigned int np,
			double l2,
			arma::sp_mat& left,
			arma::sp_mat& center,
			arma::sp_mat& right );

	void startThreads(
			std::vector<std::thread>& threads,
			std::vector<Params>& jobQueue );

	void joinThreads( std::vector<std::thread>& threads );

	void threadJob(
			std::vector<Params>& jobQueue,
			unsigned int& curJob,
			std::mutex& mutex );

private:
	unsigned int m_uiNumThreads;
};
