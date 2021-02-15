import numpy as np
import scipy.special as spspec
import math

def gaussian(mu,sigma,t):
	#x = (1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-((t-mu)/(np.sqrt(2)*sigma))**2)
	x = np.exp(-((t-mu)/(np.sqrt(2)*sigma))**2) # unscaled version, but who cares?
	x = x/np.amax(x)
	return x

def phi(alpha,t):
    #center_t = t[int(len(t)/2)]
    #t2 = t-center_t # center t array around 0
    x = 0.5*(1+spspec.erf(alpha*t/np.sqrt(2)))
    return x

def EuclDistance(pt1,pt2):
	''' Calculate the Euclidian distance between two points defined by their coordinates
	$d = \sqrt{(pt2[0]-pt1[0])^2+(pt2[1]-pt1[1])^2}$ in the 2-d case
	$d = \sqrt{\sum_{i=0}^{n-1}{pt2[i]-pt1[i])^2}}$ in the n-d case

	IN:
		- pt1: a point defined by a list (or array) of coordinates
		- pt2: another point defined by a list (or array) of coordinates
	OUT:
		- dist: the Euclidian distance between these two points
	'''
	n_dims = len(pt1) # number of dimensions
	sum__ = 0

	for i in range(n_dims):
		sum__ = sum__ + (pt2[i]-pt1[i])**2
	dist = np.sqrt(sum__)
	return dist

def Coordinates_Polygon(center,n_vertices,radius,angle_shift):
	''' Calculate the coordinates of the vertices of a polygon, given a center, a radius, and a number of vertices
	For now it only works in the 2d-case, more dimensions to come!

	IN:
		- center: the coordinates (2-element array or list) of the center of the polygon
		- n_vertices: the number of vertices (e.g. 5 for a pentagon, 6 for a hexagon, & please update your ancient Greek for more information)
		- radius: all vertices will be on a circle of this radius, centered at the center of the polygon
		- angle_shift: used to rotate the circle (angle_shift=0 will place the first point directly above the center), should be in radians

	OUT: 
		- x_coords: a list of x-coordinates of the n_vertices vertices
		- y_coords: a list of yx-coordinates of the n_vertices vertices
	'''
	x_coords = np.zeros(n_vertices)
	y_coords = np.zeros(n_vertices)
	
	for k in range(n_vertices):
		x_coords[k] = center[0] + radius * np.sin(k * 2*np.pi/n_vertices + angle_shift)
		y_coords[k] = center[1] + radius * np.cos(k * 2*np.pi/n_vertices + angle_shift)
	
	return x_coords, y_coords

def AverageFilter(x,M):
    ''' M-point moving-average filter on input array x

    IN:
        - x: input array
        - M: width (in samples) of the moving-average window

    OUT:
        - xavg: filtered array
    '''
    xavg = np.zeros(len(x))
    
    for ind in range(M,len(x)-M):
        start_ind = ind-math.floor(M/2) # utiliser int au lieu de math.floor devrait marcher aussi
        end_ind = start_ind+M
        xavg[ind] = np.mean(x[start_ind:end_ind])

    # What value on the left?
    for ind in range(M):
        xavg[ind] = xavg[M]
    
    # What value on the right?
    for ind in range(len(x)-M,len(x)):
        xavg[ind] = xavg[len(x)-M-1]

    return xavg

def fade(inputsig, dur_fadein, dur_fadeout, fadetype, datatype):
	'''This function applies a fade in and a fade out of specific duration onto the input signal
	
	IN: 
		- inputsig: the raw signal
		- dur_fadein: the duration of the fade in, in SAMPLES
		- dur_fadeout: the duration of the fade out, in SAMPLES
		- fadetype: the type of fade in and out ("lin" resp. "log" for a linear resp. logarithmic slope between 0 and 1 or 1 and 0)
		- datatype: the type of data in the numpy array Amplitudes (such as np.float32), in order to avoid to increase the size of resulting sounds (default datatype for numpy is float64...)
	OUT: 
		- outputsig the signal with fades'''
	
	Amplitudes = np.ones(len(inputsig), dtype=datatype)
	
	if fadetype == 'lin':
		Amplitudes[0:int(dur_fadein)] = np.linspace(0, 1, int(dur_fadein))
		Amplitudes[len(inputsig)-int(dur_fadeout):len(inputsig)] = np.linspace(1, 0, num=int(dur_fadeout))
	elif fadetype == 'log':
		Amplitudes[0:int(dur_fadein)] = np.logspace(-100, 0, num=int(dur_fadein))
		Amplitudes[len(inputsig)-int(dur_fadeout):len(inputsig)] = np.logspace(0, -100, num=int(dur_fadeout))
		
	outputsig = Amplitudes*inputsig
	
	return outputsig

def linmap(x,in_min,in_max,out_min,out_max):
    '''Just a linear mapping of incoming data "x" assumed to range within [in_min:in_max] into range [out_min:out_max]
    '''
    slope = (out_max-out_min)/(in_max-in_min)
    intercept = out_max - slope*in_max
  
    mapped_x = slope*x+intercept
    
    return mapped_x
