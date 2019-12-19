import numpy as np 
from matplotlib import pyplot as plt 
import argparse


def get_t_max(period, t_0, c):
	"""
	Find the moment of the maximum brightness (corresponds to the minimum stellar magnitude)
	
	Parameters
	---------
	period: float    
			Period
	
	t_0:    float    
			The first observation moment
	
	c:      array   
			Fourier coefficients

	Returns
	---------------------
	float

	Example
	------
	Find the cosine minimum argument 
	
	>>> import numpy as np
	>>> from lightcurve_module import get_t_max
	>>> t_0 = 3*np.pi
	>>>	period = 2*np.pi
	>>>	c = np.array([0, 1, 0, 0, 0, 0, 0])
	>>> get_t_max(period, t_0, c)
	3.141592653589793
 
	"""
	t = np.arange(t_0 - period, t_0, 0.01)
	w = 2*np.pi/period
	func = c[0] + c[1]*np.cos(w*t) + c[2]*np.sin(w*t) + c[3]*np.cos(2*w*t) + c[4]*np.sin(2*w*t) + c[5]*np.cos(3*w*t) + c[6]*np.sin(3*w*t)
	return t[np.argmin(func)] 


def get_phase(t_max, P, t):
	"""
	Return phases of observations	
	
	Parameters
	---------
	t_max: 	float  
			Moment of the maximum
	
	P:     	float  
			Period
	
	t:     	array  
			Moments of observations

	Returns
	-------
	array
	"""

	num_P = (t - t_max)/P
	return num_P - np.trunc(num_P)

def LS_Fourier3(x, y, period):
	
	"""
	Find Fourier coefficients up to 3-rd order with Least Squares method

	Parameters
	----------
	x:      array  
			argument values
	
	y:      array  
			function values
	
	period: float  
			period

	Returns
	-------
	array
	"""

	w = 2*np.pi/period
	
	N = len(x)
	n = 7              
	CONST = np.ones(N)
	cx = np.cos(w*x)
	sx = np.sin(w*x)
	c2x = np.cos(2*w*x)
	s2x = np.sin(2*w*x)
	c3x = np.cos(3*w*x)
	s3x = np.sin(3*w*x)
	array = np.array([CONST, cx, sx, c2x, s2x, c3x, s3x])

	
	big_array = np.repeat(array, n, axis = 0).reshape(n, n, N)*array
	little_array = array*y
	
	matrix = np.sum(big_array, 2)
	right_part = np.sum(little_array, 1)

	c = np.linalg.solve(matrix, right_part)

	return c

def get_Fourier3(t, period, c):
	"""
	Return values of 3-rd order Fourier series correspond to input arguments

	Parameters
	----------
	t:	array 
		Input arguments

	period:	float
			Period

	c:	array
		Fourier coefficients

	Returns
	-------
	array
	"""

	w = 2*np.pi/period
	return c[0] + c[1]*np.cos(w*t) + c[2]*np.sin(w*t) + c[3]*np.cos(2*w*t) + c[4]*np.sin(2*w*t) + c[5]*np.cos(3*w*t) + c[6]*np.sin(3*w*t) 

def extend_phase(phase, value):
	"""
	Return phases from -0.25 to 1.25 and corresponding array values

	Parameters
	----------
	phase:	array
			Phases
	
	value:	array
			Magnitudes

	Returns
	-------
	dict {'Phase': phase, 'Value': value}
	"""

	d = dict.fromkeys(['Phase', 'Value'])

	ind_more_0 = phase>0
	phase_more_0 = phase[ind_more_0]
	value_more_0 = value[ind_more_0]

	ind_less_025 = phase_more_0<=0.25
	phase_less_025 = phase_more_0[ind_less_025]
	value_less_025 = value_more_0[ind_less_025]

	ind_less_1 = phase<1
	phase_less_1 = phase[ind_less_1]
	value_less_1 = value[ind_less_1]

	ind_more_075 = phase_less_1>=0.75
	phase_more_075 = phase_less_1[ind_more_075]
	value_more_075 = value_less_1[ind_more_075]

	d['Phase'] = np.concatenate([phase_more_075 - 1, phase, phase_less_025 + 1])
	d['Value'] = np.concatenate([value_more_075, value, value_less_025])

	return d

def show_curve(obs_curve, av_curve):

	"""
	Plot observed data and average curve

	Parameters
	----------
	obs_curve:	dict {'Phase': phase, 'Value': value}
				Observed values

	av_curve:	dict {'Phase': phase, 'Value': value}
				Approximated values
	
	"""
	
	fig = plt.figure()
	ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

	ax.plot(av_curve['Phase'], av_curve['Value'], 'k-')
	ax.plot(obs_curve['Phase'], obs_curve['Value'], 'b.')
	ax.tick_params(top = 'on', right = 'on', direction = 'in', which = 'both')
	ax.set_xlim(-0.3, 1.3)
	ax.set_xticks(np.arange(-0.3, 1.3, 0.05), minor = 'True')
	ax.set_xlabel('Phase')
	ax.set_ylabel('Magnitude')
	ax.invert_yaxis()
	
	plt.show()

def parse_arguments():

	"""
	Parse command line arguments
	"""
	parser = argparse.ArgumentParser(add_help = True)
	parser.add_argument('-f', '--file', action='store', dest='file', help = 'Data file', type = str) 
	parser.add_argument('-p', '--period', action='store', dest='period', help = 'Period', type = float) 
	parser_args = parser.parse_args()

	return parser_args
