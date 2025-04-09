import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{mathtools}
\usepackage{helvet}
'''



colors = ["#ffffff","#F88379", "#66c2a5", "#F5F5DC"]  #FAA0A0 "Pastel red"
# Define the three colors (RGB or Hex)
colors = ["#ffffff", "#66c2a5", "#8da0cb"]  #FAA0A0 "Pastel red"
cmap = ListedColormap(colors)


data_VR1_VR2=np.loadtxt("payoff_VR1_VR2_ar.txt")
data_VR1_PR2=np.loadtxt("payoff_VR1_PR2_ar.txt")
data_PR1_VR2=np.loadtxt("payoff_PR1_VR2_ar.txt")
data_PR1_PR2=np.loadtxt("payoff_PR1_PR2_ar.txt")


S=np.linspace(-2.001,-0.001,100)
T=np.linspace(-2.001,0.999,100)
Parameter_dependence=np.zeros((len(S),len(T)))
ESS_VR=np.zeros((len(S),len(T)))
ESS_PR=np.zeros((len(S),len(T)))
ESS=np.zeros((len(S),len(T)))



for i1 in range(len(S)):
	# print(i1)
	for j1 in range(len(T)):
		payoff_VR1_PR2=data_VR1_PR2[i1,j1]
		payoff_PR1_VR2=data_PR1_VR2[i1,j1]
		payoff_VR1_VR2=data_VR1_VR2[i1,j1]
		payoff_PR1_PR2=data_PR1_PR2[i1,j1]
		# print(payoff_VR1_PR2,payoff_VR1_VR2,payoff_PR1_VR2,payoff_PR1_PR2)

		if payoff_VR1_VR2>payoff_PR1_VR2:
			ESS_VR[i1,j1]=1.0
			ESS[i1,j1]=1.0
		elif payoff_VR1_VR2==payoff_PR1_VR2 and payoff_VR1_PR2>payoff_PR1_PR2:
			ESS_VR[i1,j1]=1.0
			ESS[i1,j1]=1.0
		if payoff_PR1_PR2 > payoff_VR1_PR2:
			ESS_PR[i1,j1]=1.0
			ESS[i1,j1]=2.0
		elif payoff_PR1_PR2 == payoff_VR1_PR2 and payoff_PR1_VR2>payoff_VR1_VR2:
			ESS_PR[i1,j1]=1.0
			ESS[i1,j1]=2.0
		if ESS_VR[i1,j1] ==1 and ESS_PR[i1,j1] ==1:
			ESS[i1,j1]=3.0
		if ESS_VR[i1,j1] ==0 and ESS_PR[i1,j1] ==0:
			ESS[i1,j1]=4.0

		# if  T[j1]> 2*S[i1]:
		# 	ESS[i1,j1]=0.0


plt.imshow(ESS, origin='lower', extent=[-2.001, 0.999, -2.001, -0.001],cmap=cmap)
# plt.plot(S, S, '--k')
# plt.plot(2*S, S, '--b')

# plt.plot(T, T - 1, '-k', label=r'$\frac{0-S}{1-T}=1$')
# plt.axvline(x=0, color='k', linestyle='--')
# plt.axhline(y=-1,color='b', linestyle='--')

# Set x and y ticks
plt.xticks([-2.0, -0.5, 1.0])
plt.yticks([-2.0, -1.0, 0.0])

plt.xlim([-2.0, 1.0])
plt.ylim([-2.0, 0.0])

# Get the current axis
ax = plt.gca()

# Move y-axis label and ticks to the right side
ax.yaxis.tick_right()  # Move ticks to the right
ax.yaxis.set_label_position("right")  # Move label to the right
ax.yaxis.set_ticks_position('right')  # Ensure ticks are only on the right

# Remove everything on the left side
ax.spines['left'].set_visible(False)

# Adjust label styles
plt.tick_params(axis='x', labelsize=17)
plt.tick_params(axis='y', labelsize=17)
plt.xlabel('$T$', fontsize=17)
plt.ylabel('$S$', fontsize=17, rotation=270, labelpad=15)  # Rotate for alignment

plt.legend(fontsize=17)
plt.savefig("ESS.svg", dpi=500)
plt.show()