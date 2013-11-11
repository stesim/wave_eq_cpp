import numpy;
import matplotlib.pyplot as plt;

x = numpy.fromfile("plot_data_x.dat",float,-1," ")
y = numpy.fromfile("plot_data_y.dat",float,-1," ")

plt.plot(x,y)
plt.show()
