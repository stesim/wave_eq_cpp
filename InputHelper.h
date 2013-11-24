#pragma once

#include <iostream>
#include <sstream>

/*
* Input helper function.
*/
template<typename T>
T inputParam( const char* name, T defVal )
{
	std::cout << name << " [" << defVal << "]: ";
	T val;
	std::string input;
	std::getline( std::cin, input );
	if( !input.empty() )
	{
		std::istringstream stream( input );
		stream >> val;
	}
	else
	{
		val = defVal;
	}
	return val;
}
