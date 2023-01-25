import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Defining variables:
#path = = "2_depth.txt", path = "cubo_depth_02.png"

def impot_data(path = "Sensor_data/cubo_depth_02.png"):
	
	try:
		#Import Depth
		sensor_data = np.genfromtxt(path, delimiter=',') 
		sensor_data = cv2.rotate(sensor_data, cv2.ROTATE_90_CLOCKWISE)
	except:
		#Import RGB
		sensor_data = cv2.imread(path)
		sensor_data = cv2.cvtColor(sensor_data, cv2.COLOR_BGR2GRAY)

		#percent by which the image is resized
		scale_percent = 10

		#calculate the X percent of original dimensions
		width = int(sensor_data.shape[1] * scale_percent / 100)
		height = int(sensor_data.shape[0] * scale_percent / 100)

		# dsize
		dsize = (width, height)
		
		# resize image
		sensor_data = cv2.resize(sensor_data, dsize)

	return sensor_data

def click_event(event, x, y, flags, params): 
	if len(seed) >= 3:
		cv2.destroyAllWindows()
		return None

	if event == cv2.EVENT_LBUTTONDOWN:   
	    seed.append([y,x])
	    cv2.circle(arr1, (x,y), radius=0, color=(0, 0, 255), thickness=5)
	    cv2.imshow('Defina os pontos',arr1)
	    return seed

def choose_point():

	cv2.namedWindow("Defina os pontos",  cv2.WINDOW_KEEPRATIO) 
	cv2.imshow('Defina os pontos', arr1) 
	cv2.setMouseCallback('Defina os pontos', click_event)
	print(seed)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	print(depth.shape)

def print_import(depth, img):

	plt.subplot(1,2,1)
	plt.imshow(depth)
	plt.subplot(1,2,2)
	plt.imshow(img)
	plt.show()

def normal_vector(depth, seed):
	
	for point in seed:
		point.append(depth[point[0], point[1]])
	print(seed)
	P1, P2, P3 = np.array(seed)

	A = P1 - P2
	B = P1 - P3

	n = list(np.cross(A,B))
	i, j, k = n
	n = n/(np.sqrt(i**2 + j**2 + k**2))

	print("Normal vector", n)

	constant = -np.dot(n,P1)

	return n, constant

def Euler_Angles(n):
	
	a, b, c = n
	cos_alpha = a/(np.sqrt(a**2 + b**2 + c**2))
	cos_beta = b/(np.sqrt(a**2 + b**2 + c**2))
	cos_gamma = c/(np.sqrt(a**2 + b**2 + c**2))

	eulers = [np.arccos(cos_alpha), np.arccos(cos_beta), np.arccos(cos_gamma)]
	print("Eulers Angles in radians: ", eulers)

	return eulers

def set_axes_radius(ax, origin, radius):
    '''
        From StackOverflow question:
        https://stackoverflow.com/questions/13685386/
    '''
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax, zoom=1.):
    '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect("equal") and ax.axis("equal") not working for 3D.
        input:
          ax:   a matplotlib axis, e.g., as output from plt.gca().

    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis = 1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0])) / zoom
    set_axes_radius(ax, origin, radius)

def plot_3d(n, constant, depth, arrow_size = 50, step = 20):

	xx, yy = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]

	# calculate Z for the plane equation
	z = (-n[0] * xx - n[1] * yy - constant)/n[2]

	fig = plt.figure()
	ax = fig.gca(projection = '3d')

	ax.set_zlabel("Z", color = 'b', size = 18)

	
	ax.quiver(
		xx[0:depth.shape[0]:step, 
		   0:depth.shape[1]:step],

		yy[0:depth.shape[0]:step,
		   0:depth.shape[1]:step], 
		
		z[0:depth.shape[0]:step,
		  0:depth.shape[1]:step], 
		
		n[0], n[1], n[2],

		color = "m", length = arrow_size)
	set_axes_equal(ax)

	ax.plot_surface(xx, yy, z, rstride = 1, cstride = 1, color = "r",
        linewidth = 0, alpha = 1)
	ax.plot_surface(xx, yy, depth ,rstride = 1, cstride = 1, cmap = plt.cm.viridis,
        linewidth = 0)
	ax.set_zlim(depth.max(), depth.min())
	ax.view_init(45, 90)

	

	
	
	# show it
	plt.show()

def copy8bits(depth):
	arr1 = depth.copy()
	arr1 = cv2.normalize(arr1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
	arr1 = arr1.astype(np.uint8)
	return arr1



if __name__ == '__main__':

	# Define the data path
	path = "Sensor_data/2_depth.txt"
	# Import data
	depth = impot_data(path)

	# When you need to print both depth and RGB data
	## print_import(depth, img)
	
	# Define global list for the 3 coplanar points
	seed = []

	# Make a copy of the depth to preserve data
	arr1 = copy8bits(depth)

	# Event to choose point
	choose_point()
	print(seed)

	# Calculate the normal vector and the constant parameter
	n, constant = normal_vector(depth, seed)

	# Convert the normal vector in Euler angles
	eulers = Euler_Angles(n)

	# Plot the results
	plot_3d(n, constant, depth, 0.1)

