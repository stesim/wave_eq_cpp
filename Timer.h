#pragma once

class Timer
{
public:
	Timer();
	virtual ~Timer();

	void start();
	double stop();
	
	double getElapsedTime();

	bool isRunning();

protected:
	virtual double getTime() const = 0;

private:
	double m_dStartTime;
	double m_dStopTime;
	bool m_bRunning;
};
