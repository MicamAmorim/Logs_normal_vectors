import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Defining variables:


def impot_data():
	z_path = "2_depth.txt"
	RGB_path = "cubo_depth_02.png"
	intr_path = "2_intrinsics.txt"

	depth = np.genfromtxt(z_path, delimiter=',') 
	depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
	print(depth.shape)
	M,N = depth.shape

	img = cv2.imread(RGB_path)

	M_rgb, N_rgb,_ = img.shape

	Sh = M_rgb/M
	Sw = N_rgb/N

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#percent by which the image is resized
	scale_percent = 20

	#calculate the 50 percent of original dimensions
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)

	# dsize
	dsize = (width, height)

	# resize image
	img = cv2.resize(img, dsize)

	intrinsics = np.genfromtxt(intr_path, delimiter=',')
	print(intrinsics)

	return depth, img, [Sh, Sw], intrinsics

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

def plot_3d(n, constant, depth):
	xx, yy = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]

	# calculate Z for the plane equation
	z = (-n[0] * xx - n[1] * yy - constant)/n[2]

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(xx, yy, depth ,rstride=1, cstride=1, cmap=plt.cm.viridis,
        linewidth=0)
	ax.set_zlim(depth.max(), depth.min())
	ax.view_init(45, 90)

	ax.plot_surface(xx, yy, z, rstride=1, cstride=1, color = "r",
        linewidth=0, alpha = 1)

	# show it
	plt.show()

def copy8bits(depth):
	arr1 = depth.copy()
	arr1 = cv2.normalize(arr1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
	arr1 = arr1.astype(np.uint8)
	return arr1



if __name__ == '__main__':


	# invert the position of img with depth to import the raw data
	img, depth, scale, intrinsics = impot_data()
	#print_import(depth, img)
	seed = []
	
	arr1 = copy8bits(depth)

	choose_point()
	print(seed)
	n, constant = normal_vector(depth, seed)
	eulers = Euler_Angles(n)

	plot_3d(n, constant, depth)

