#pragma once

#include "Timer.h"

class WallTimer : public Timer
{
public:
	WallTimer();
	virtual ~WallTimer();

protected:
	virtual double getTime() const;
};
