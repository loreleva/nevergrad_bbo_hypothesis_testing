import math
import numpy as np


# FUNCTIONS IMPLEMENTATION

"""
MANY LOCAL MINIMA
"""

def ackley_function(x, param):
	a = param["a"]
	b = param["b"]
	c = param["c"]

	dim = len(x)
	sum1 = np.square(x).sum()
	sum2 = np.cos(np.multiply(c, x)).sum()

	term1 = -a * np.power(math.e, -b * np.sqrt(sum1 / dim))
	term2 = -np.power(math.e, sum2 / dim)

	return term1 + term2 + a + math.e


def bukin_function_n_6(x):
	return 100 * np.sqrt(abs(x[1] - 0.01*(x[0]**2))) + 0.01 * abs(x[0] + 10)


def cross_in_tray_function(x):
	abs1 = abs(100 - (np.sqrt(x[0]**2 + x[1]**2) / math.pi))
	abs2 = abs(np.sin(x[0]) * np.sin(x[1]) * np.power(math.e, abs1))
	return -0.0001 * np.power(abs2 + 1, 0.1)


def drop_wave_function(x):
	num = 1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))
	denom = 0.5 * (x[0]**2 + x[1]**2) + 2
	return -(num / denom)


def eggholder_function(x):
	term1 = - (x[1] + 47) * np.sin(np.sqrt(abs(x[1] + x[0]/2 + 47)))
	term2 = x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
	return term1 - term2


def gramacy_lee_2012_function(x):
	return (np.sin(10 * math.pi * x) / (2 * x)) + np.power(x-1, 4)


def griewank_function(x):
	dim = len(x)
	sum1 = (np.square(x)/4000).sum()
	prod = np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1))))
	return sum1 - prod + 1


def holder_table_function(x):
	return -abs(np.sin(x[0]) * np.cos(x[1]) * np.power(math.e, abs(1 - (np.sqrt(x[0]**2 + x[1]**2) / math.pi))))


def langermann_function(x, param):
	m = param["m"]
	c = param["c"]
	A = np.asarray(param["A"])
	res = 0
	for i in range(m):
		term1 = np.power(math.e, -(1 / math.pi) * np.square(x - A[i]).sum())
		term2 = np.cos(math.pi * np.square(x - A[i]).sum())
		res += c[i] * term1 * term2
	return res


def levy_function(x):
	x = np.asarray(x)
	w = np.asarray(1 + (x-1) / 4)
	w_d = w[-1]
	term1 = (w_d - 1)**2 * (1 + np.sin(2 * math.pi * w_d)**2)
	w = w[:-1]
	sum1 = (w-1)**2 * (1 + 10 * np.sin(math.pi * w + 1)**2)
	return np.sin(math.pi * w[0])**2 + (sum1 + term1).sum()


def levy_function_n_13(x):
	return np.sin(3 * math.pi * x[0])**2 + (x[0] - 1)**2 * (1 + np.sin(3 * math.pi * x[1])**2) + (x[1] - 1)**2 * (1 + np.sin(2 * math.pi * x[1])**2)


def rastrigin_function(x):
	dim = len(x)
	x = np.asarray(x)
	return 10 * dim + (x**2 - 10 * np.cos(2 * math.pi * x)).sum()


def schaffer_function_n_2(x):
	num = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
	denom = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
	return 0.5 + num / denom

def schaffer_function_n_4(x):
	num = np.cos(np.sin(abs(x[0]**2 - x[1]**2)))**2 - 0.5
	denom = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
	return 0.5 + num / denom


def schwefel_function(x):
	dim = len(x)
	x = np.asarray(x)
	return 418.9829 * dim - (x * np.sin(np.sqrt(np.absolute(x)))).sum()


def shubert_function(x):
	term1 = sum([i * np.cos((i + 1) * x[0] + i) for i in range(1,6)])
	term2 = sum([i * np.cos((i + 1) * x[1] + i) for i in range(1,6)])
	return term1 * term2


"""
BOWL-SHAPED
"""


def bohachevsky_functions(x):
	return x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * math.pi * x[0]) - 0.4 * np.cos(4 * math.pi * x[1]) + 0.7


def perm_function_o_d_b(x, param):
	dim = len(x)
	b = param["b"]
	res = 0
	for i in range(1, dim+1):
		inner_sum = 0
		for j in range(1, dim+1):
			inner_sum += (j + b) * (x[j-1]**i - (1 / j**i))
		res += inner_sum**2
	return res


def rotated_hyper_ellipsoid_function(x):
	dim = len(x)
	res = 0
	for i in range(1, dim+1):
		for j in range(i):
			res += x[j]**2
	return res


def sphere_function(x):
	return np.square(x).sum()


def sum_of_different_powers_function(x):
	dim = len(x)
	res = 0
	for i in range(1, dim+1):
		res += abs(x[i-1])**(i+1)
	return res


def sum_squares_function(x):
	dim = len(x)
	res = 0
	for i in range(1, dim+1):
		res += i * x[i-1]**2
	return res


def trid_function(x):
	dim = len(x)
	x = np.asarray(x)
	sum1 = np.square(x-1).sum()
	sum2 = 0
	for i in range(2, dim+1):
		sum2 += x[i-1] * x[i-2]
	return sum1 - sum2


"""
PLATE-SHAPED
"""


def booth_function(x):
	return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def matyas_function(x):
	return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]


def mccormick_function(x):
	return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1


def power_sum_function(x, param):
	dim = len(x)
	b = param["b"]
	res = 0
	for i in range(1, dim+1):
		sum1 = 0
		for j in range(i):
			sum1 += x[j]**i
		res += (sum1 - b[i-1])**2
	return res


def zakharov_function(x):
	dim = len(x)
	sum1 = 0
	for i in range(1, dim+1):
		sum1 += 0.5 * i * x[i-1]
	return np.square(x).sum() + sum1**2 + sum1**4


"""
VALLEY-SHAPED
"""


def three_hump_camel_function(x):
	return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6 / 6) + x[0] * x[1] + x[1]**2


def six_hump_camel_function(x):
	return (4 - 2.1 * x[0]**2 + (x[0]**4 / 3)) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2


def dixon_price_function(x):
	dim = len(x)
	res = 0
	for i in range(2, dim+1):
		res += i * (2 * x[i-1]**2 - x[i-2])**2
	return (x[0] - 1)**2 + res


def rosenbrock_function(x):
	dim = len(x)
	res = 0
	for i in range(1, dim):
		res += 100 * (x[i] - x[i-1]**2)**2 + (x[i-1] - 1)**2
	return res


"""
STEEP RIDGES/DROPS
"""


def de_jong_function_n_5(x):
	dim = len(x)
	base = [-32, -16, 0, 16, 32]
	a1 = base * 5
	a2 = [x for x in base for i in range(5)]
	a = np.asarray([a1, a2])

	sum1 = 0
	for i in range(1, 26):
		sum1 += 1 / (i + (x[0] - a[0][i-1])**6 + (x[1] - a[1][i-1])**6)
	return (0.002 + sum1)**(-1)


def easom_function(x):
	return -np.cos(x[0]) * np.cos(x[1]) * np.power(math.e, - (x[0] - math.pi)**2 - (x[1] - math.pi)**2)


def michalewicz_function(x):
	dim = len(x)
	res = 0
	for i in range(1, dim+1):
		res += np.sin(x[i-1]) * np.sin((i * x[i-1]**2) / math.pi)**(2*10)
	return -res


"""
OTHER
"""


def beale_function(x):
	return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2


def branin_function(x):
	a = 1
	b = 5.1 / (4 * math.pi**2)
	c = 5 / math.pi
	r = 6
	s = 10
	t = 1 / (8 * math.pi)

	return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s


def colville_function(x):
	return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2 + (x[2] - 1)**2 + 90 * (x[2]**2 - x[3])**2 + 10.1 * ((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8 * (x[1] - 1) * (x[3] - 1)


def forrester_et_al_2008_function(x):
	return (6 * x - 2)**2 * np.sin(12*x - 4)


def goldstein_price_function(x):
	term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)
	term2 = 30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2)
	return term1 * term2


def hartmann_3_d_function(x):
	a = np.asarray([1., 1.2, 3.0, 3.2])
	A = np.asarray([[3.0, 10, 30], 
		            [0.1, 10, 35], 
		            [3.0, 10, 30], 
		            [0.1, 10, 35]])
	P = 10**(-4) * np.asarray([[3689, 1170, 2673], 
							   [4699, 4387, 7470], 
							   [1091, 8732, 5547], 
							   [381,  5743, 8828]])
	res = 0
	for i in range(4):
		sum1 = 0
		for j in range(3):
			sum1 += A[i][j] * (x[j] - P[i][j])**2
		res += a[i] * np.power(math.e, -sum1)
	return -res


def hartmann_4_d_function(x):
	a = np.asarray([1., 1.2, 3.0, 3.2])
	A = np.asarray([[10,   3,   17,   3.50, 1.7, 8 ], 
					[0.05, 10,  17,   0.1,  8,   14], 
					[3,    3.5, 1.7,  10,   17,  8 ], 
					[17,   8,   0.05, 10,   0.1, 14]])
	P = 10**(-4) * np.asarray([[1312, 1696, 5569, 124,  8283, 5886], 
							   [2329, 4135, 8307, 3736, 1004, 9991], 
							   [2348, 1451, 3522, 2883, 3047, 6650], 
							   [4047, 8828, 8732, 5743, 1091, 381]])

	sum1 = 0
	for i in range(4):
		sum2 = 0
		for j in range(4):
			sum2 += A[i][j] * (x[j] - P[i][j])**2
		sum1 += a[i] * np.power(math.e, -sum2)
	return (1 / 0.839) * (1.1 - sum1)


def hartmann_6_d_function(x):
	a = np.asarray([1., 1.2, 3.0, 3.2])
	A = np.asarray([[10,   3,   17,   3.50, 1.7, 8 ], 
					[0.05, 10,  17,   0.1,  8,   14], 
					[3,    3.5, 1.7,  10,   17,  8 ], 
					[17,   8,   0.05, 10,   0.1, 14]])
	P = 10**(-4) * np.asarray([[1312, 1696, 5569, 124,  8283, 5886], 
							   [2329, 4135, 8307, 3736, 1004, 9991], 
							   [2348, 1451, 3522, 2883, 3047, 6650], 
							   [4047, 8828, 8732, 5743, 1091, 381]])

	sum1 = 0
	for i in range(4):
		sum2 = 0
		for j in range(6):
			sum2 += A[i][j] * (x[j] - P[i][j])**2
		sum1 += a[i] * np.power(math.e, -sum2)
	return -sum1


def perm_function_d_b(x, param):
	dim = len(x)
	b = param["b"]

	sum1 = 0
	for i in range(1, dim+1):
		sum2 = 0
		for j in range(1, dim+1):
			sum2 += (j**i + b) * ((x[j-1] / j)**i - 1)
		sum1 += sum2**2
	return sum1


def powell_function(x):
	dim = len(x)
	res = 0
	for i in range(1, (dim//4)+1):
		res += (x[(4*i-3)-1] + 10 * x[(4*i-2)-1])**2 + 5 * (x[(4*i-1)-1] - x[4*i-1])**2 + (x[(4*i-2)-1] - 2*x[(4*i-1)-1])**4 + 10 * (x[(4*i-3)-1] - x[4*i - 1])**4
	return res


def shekel_function(x):
	b = np.asarray([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) * 0.1
	C = np.asarray([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7. ],
					[4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],
					[4., 1., 8., 6., 3., 2., 5., 8., 6., 7. ],
					[4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],])

	res = 0
	for i in range(10):
		sum1 = 0
		for j in range(4):
			sum1 += (x[j] - C[j][i])**2
		res += (sum1 + b[i])**(-1) 
	return -res


def styblinski_tang_function(x):
	return 0.5 * (np.power(x, 4) - 16 * np.square(x) + 5 * np.asarray(x)).sum()