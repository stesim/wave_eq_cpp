#pragma once

#include "Timer.h"

class CpuTimer : public Timer
{
public:
	CpuTimer();
	virtual ~CpuTimer();

protected:
	virtual double getTime() const;
};
