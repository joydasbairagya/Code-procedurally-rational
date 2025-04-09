import numpy as np
import matplotlib.pyplot as plt
from numba import njit,jit,prange
import time

start=time.time()
#We haveassume first player strategey is x and 2nd player's is y
@njit(nogil=True)
def PR1(x,y,s,t):
	if s>t:
		sam_func=y*(2-y)-x
	if s<t:
		sam_func=y-x
	if s==t:
		sam_func=y*(3/2-y)-x
	return(sam_func)

@njit(nogil=True)
def PR2(x,y,s,t):
	if s>t:
		sam_func=x*(2-x)-y
	if s<t:
		sam_func=x-y
	if s==t:
		sam_func=x*(3/2-x)-y
	return(sam_func)

@njit(nogil=True)
def VR2(x,y,s,t):
	if 1+s>0:
		f=2*x*(1-x)
	elif 1+s<0:
		f=2*x*(1-x)*(x**2+2*x*(1-x))
	elif 1+s==0:
		f=2*x*(1-x)*(x**2+2*x*(1-x)+(1/2)*(1-x)**2)
	if 2*s>t:
		g=(1-x)**2 *(x**2+2*x*(1-x)**2)
	elif 2*s<t:
		g=(1-x)**2 * x**2
	elif 2*s==t:
		g=(1-x)**2 *(x**2+x*(1-x)**2)

	best_func=x**2+f+g-y

	return(best_func)

@njit(nogil=True)
def VR1(x,y,s,t):
	if 1+s>0:
		f=2*y*(1-y)
	elif 1+s<0:
		f=2*y*(1-y)*(y**2+2*y*(1-y))
	elif 1+s==0:
		f=2*y*(1-y)*(y**2+2*y*(1-y)+(1/2)*(1-y)**2)
	if 2*s>t:
		g=(1-y)**2 *(y**2+2*y*(1-y)**2)
	elif 2*s<t:
		g=(1-y)**2 * y**2
	elif 2*s==t:
		g=(1-y)**2 *(y**2+y*(1-y)**2)
		
	best_func=y**2+f+g-x
	return(best_func)

@njit(nogil=True)
def increment(VR1,VR2,x,y,s,t,h):
	k1x=VR1(x,y,s,t)
	k1y=VR2(x,y,s,t)

	k2x=VR1(x+(h/2)*k1x,y+(h/2)*k1y,s,t)
	k2y=VR2(x+(h/2)*k1x,y+(h/2)*k1y,s,t)

	k3x=VR1(x+(h/2)*k2x,y+(h/2)*k2y,s,t)
	k3y=VR2(x+(h/2)*k2x,y+(h/2)*k2y,s,t)

	k4x=VR1(x+(h)*k3x,y+(h)*k3y,s,t)
	k4y=VR2(x+(h)*k3x,y+(h)*k3y,s,t)

	incrementx=(h/6)*(k1x+2*k2x+2*k3x+k4x)
	incrementy=(h/6)*(k1y+2*k2y+2*k3y+k4y)

	return(incrementx,incrementy)

@njit(nogil=True)
def time_evolution(VR1,VR2,x0,y0,s,t):
	h=10**-2
	time=np.arange(0,10**2,h)
	for i in range(len(time)):
		small_increment=increment(VR1,VR2,x0,y0,s,t,h)
		dx=small_increment[0]
		dy=small_increment[1]
		x0 = x0 + dx
		y0 = y0 + dy
	return(x0,y0)




@njit(nogil=True,parallel=True)
def initial_condition(VR1,VR2,s,t):
	X0=np.linspace(0,1,240)
	Y0=np.linspace(0,1,240)
	Intitial_condition_dependence=np.zeros((len(X0),len(Y0)))
	for i in prange(len(X0)):
		for j in prange(len(Y0)):
			#Payoff of the first player is calculated here
			xa=time_evolution(VR1,VR2,X0[i],Y0[j],s,t)[0]
			Intitial_condition_dependence[i,j]=xa*(xa+(1-xa)*s)+(1-xa)*xa*t
	return(np.mean(Intitial_condition_dependence))

# print(initial_condition(PR1,PR2,-0.25,0.6))
# nash=(0.25)/(1+0.25-0.5)
# plt.imshow(initial_condition(PR1,PR2,-0.25,0.5),origin='lower',extent=[0,1,0,1])
# plt.axhline(y=nash,color='k',linestyle='--')
# plt.axvline(x=nash,color='k',linestyle='--')
# plt.show()
# print()
# print("Nash",nash)
@njit(nogil=True)
def parameter_dependence():
	S=np.linspace(-2.001,-0.001,100)
	T=np.linspace(-2.001,0.999,100)
	Parameter_dependence=np.zeros((len(S),len(T)))
	ESS_VR=np.zeros((len(S),len(T)))
	ESS_PR=np.zeros((len(S),len(T)))
	ESS=np.zeros((len(S),len(T)))
	payoff_VR1_PR2_ar=np.zeros((len(S),len(T)))
	payoff_PR1_VR2_ar=np.zeros((len(S),len(T)))
	payoff_VR1_VR2_ar=np.zeros((len(S),len(T)))
	payoff_PR1_PR2_ar=np.zeros((len(S),len(T)))
	

	for i1 in range(len(S)):
		print(i1)
		for j1 in range(len(T)):
			payoff_VR1_PR2=initial_condition(VR1,PR2,S[i1],T[j1])
			payoff_PR1_VR2=initial_condition(PR1,VR2,S[i1],T[j1])
			payoff_VR1_VR2=initial_condition(VR1,VR2,S[i1],T[j1])
			payoff_PR1_PR2=initial_condition(PR1,PR2,S[i1],T[j1])


			payoff_VR1_PR2_ar[i1,j1]=payoff_VR1_PR2
			payoff_PR1_VR2_ar[i1,j1]=payoff_PR1_VR2
			payoff_VR1_VR2_ar[i1,j1]=payoff_VR1_VR2
			payoff_PR1_PR2_ar[i1,j1]=payoff_PR1_PR2

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




			# Parameter_dependence[i1,j1]= initial_condition(VR1,PR2,S[i1],T[j1])-initial_condition(PR1,PR2,S[i1],T[j1])
	return(ESS,payoff_VR1_PR2_ar,payoff_PR1_VR2_ar,payoff_VR1_VR2_ar,payoff_PR1_PR2_ar)










result=parameter_dependence()
#print(np.mean(result))
np.savetxt("ESS.txt",result[0])
np.savetxt('payoff_VR1_PR2_ar.txt',result[1])
np.savetxt('payoff_PR1_VR2_ar.txt',result[2])
np.savetxt('payoff_VR1_VR2_ar.txt',result[3])
np.savetxt('payoff_PR1_PR2_ar.txt',result[4])







end=time.time()
print("time=",end-start)






# s=-1.0
# t=-1.5

# nash=-s/(1-s-t)
# T=np.linspace(-2.001,-0.001,100)
# plt.imshow(result,origin='lower',extent=[-2.001,-0.001,-2.001,-0.001])
# plt.axhline(y=nash,color='k',linestyle='--')
# plt.axvline(x=nash,color='k',linestyle='--')

# plt.plot(T,T,'--k')
# plt.savefig('parameter_dependence.svg',dpi=900)
# plt.show()
