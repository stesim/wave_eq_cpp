import numpy
import matplotlib.pyplot as plt

def plot(step,numSteps,x,numSol,exSol,error):
	_numSol = numpy.array(numSol)
	_exSol = numpy.array(exSol)
	plt.subplot(2,1,1)
	plt.plot(x,numSol,x,exSol)
	plt.title("Numerical and analytical solution")
	plt.xlabel("x")
	plt.subplot(2,1,2)
	plt.plot(error)
	plt.title("L2 error")
	plt.xlabel("t")
	plt.show()
