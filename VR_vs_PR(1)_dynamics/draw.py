import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Define the three colors (RGB or Hex)
colors = ["#ffffff", "#66c2a5", "#8da0cb"]  #FAA0A0 "Pastel red"
cmap = ListedColormap(colors)


S=np.linspace(-2.001,-0.001,100)
T=np.linspace(-2.001,-0.001,100)

data=np.loadtxt("Result_parameter_dependence.txt")

for i in range(len(S)):
	for j in range(len(T)):
		if S[i]<T[j]:
			data[i,j]=np.nan



plt.imshow(data, cmap=cmap,origin='upper',extent=[-2.001,-0.001,-2.001,-0.001],interpolation='nearest')
plt.savefig('Figure1.svg',dpi=500)
plt.show()

