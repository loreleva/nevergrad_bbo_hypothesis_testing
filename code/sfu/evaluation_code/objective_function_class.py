import math, os, json
from rpy2 import robjects
from subprocess import check_output
import shutil
import numpy as np

class JsonNotLoaded(Exception):
	pass

class sfuFunctionError(Exception):
	pass

json_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "functions/") + "functions.json"

def get_json_functions():
	"""
	Returns the dictionary of all the functions inside the functions' json
	"""
	global json_filepath
	f = open(json_filepath)
	json_functions = json.load(f)
	f.close()
	return json_functions

def get_json_function(name):
	"""
	Returns the dictionary of the selected function inside the functions' json
	"""
	global json_filepath
	f = open(json_filepath)
	json_functions = json.load(f)
	f.close()
	try:
		res_json = json_functions[name]
	except KeyError:
		raise sfuFunctionError(f"The selected function does not exists: \"{name}\"")
	return res_json

def search_function(filters=None):
	"""
	Search a function in the given json

	Parameters
	---------
	filters: dict
		Dictionary with the json fields as keys.

	Returns
	------
	list
		The list of function names which satisfy the filters.
		If no filter is given in input, all the functions are returned.
	"""
	json_functions = get_json_functions()
	results = []
	for function in json_functions:
		ok = 1
		if filters!=None:
			for filt in filters:
				if type(filters[filt]) == bool:
					temp_value_filter = False
					if type(json_functions[function][filt]) != bool and json_functions[function][filt] != None:
						temp_value_filter = True
					if temp_value_filter != filters[filt]:
						ok = 0
				elif json_functions[function][filt] != str(filters[filt]):
					ok = 0
		if ok == 1:
			results.append(function)
	return results


class objective_function():
	"""Objective function's class
	
	Parameters
	----------
	name: str
		Name of the objective function
	dimension: int
		Dimension of the objective function.
	has_input_domain: bool
		It is True when the range of the input domain is available.
	input_lb: list
		Each index of the list represents a dimension, a value at that index represents the lower bound of the range of that dimension.
	input_ub: list
		Each index of the list represents a dimension, a value at that index represents the upper bound of the range of that dimension.
	input_opt: int, float, list, None
		Function's input value of the global optimum. When the value is list of lists, it represents the multiple possible input values for the global optimum.
	has_parameters: bool
		It is True when the function accepts parameters.
	parameters_description: str
		Description of the function's parameters.
	parameters: dict
		Dictionary of parameters' values of the objective function.
	parameters_names: list
		List of the parameters names.
	param_opt: tuple, None
		Tuple with parameter name at idx 0, and dict of values for which opt is defined at idx 1, if any.
	opt: float, None
		Function's global optimum.
	R_code: str
		Function's implementation in R.
	implementation_name: str
		R function's name of the objective function implementation.

	Functions
	---------
	evaluate(inp, param): evaluate the function on "inp" input values.
	"""

	def __init__(self, name, dim=None, param=None):
		"""
		Parameters
		----------
		name: str
			Name of the objective function
		dim: str, int, None
			Dimension of the objective function.
			If "dim" is different from "d", it can be either an int or str type (e.g., 5 or "5").
			If the dimension is "d", this means that the objective function accepts on input values of any dimension.
			"dimension" can be also a None value if the objective function accepts inputs of only one dimension value (e.g., only input of size 5).
		param: dict, None
			Values for the function's parameters.
			If given in input, the keys of the dictionary are the parameters names.
			If nothing is given in input, the function's parameters values will be setted to default ones, if any.
		"""
		# name of the objective function
		self.name = name
		# obtain function json dictionary
		json_functions = get_json_functions()
		if not self.name in json_functions:
			raise sfuFunctionError("The function selected does not exist")
		json_func = json_functions[self.name]
		
		# define function dimension
		json_dim = json_func["dimension"]
		# if dim is given, check that it is an integer
		try:
			if dim != None:
				dim = int(dim)
		except ValueError:
			raise sfuFunctionError(f"The given dimension is not an integer: {dim=}")

		if json_dim == "d":
			if dim == None:
				raise sfuFunctionError("The function selected needs dimension definition")
			self.dimension = dim
		else:
			if dim != None and dim != int(json_dim):
				raise sfuFunctionError(f"The given dimension is not accepted by the selected function. The selected function supports only this dimension: {json_dim}")
			self.dimension = int(json_dim)


		# define input domain range
		if json_func["input_domain"] == None:
			self.has_input_domain = False
		else:
			self.has_input_domain = True
			self.input_lb = []
			self.input_ub = []
			# if input domain is defined for each dimension
			if len(json_func["input_domain"]) == self.dimension:
				for range_domain in json_func["input_domain"]:
					# if the range is python code, evaluate it
					if type(range_domain[0]) == str:
						local_var = {"d" : self.dimension}
						exec(range_domain[0], globals(), local_var)
						self.input_lb.append(local_var["input_domain"][0])
						self.input_ub.append(local_var["input_domain"][1])
					else:
						self.input_lb.append(range_domain[0])
						self.input_ub.append(range_domain[1])
			# if input domain has to be expanded for each dimension
			else:
				range_domain = json_func["input_domain"][0]
				# if the range is python code, evaluate it
				if type(range_domain[0]) == str:
					local_var = {"d" : self.dimension}
					exec(range_domain[0], globals(), local_var)
					temp_lb = local_var["input_domain"][0]
					temp_ub = local_var["input_domain"][1]
				else:
					temp_lb = range_domain[0]
					temp_ub = range_domain[1]
				for x in range(self.dimension):
					self.input_lb.append(temp_lb)
					self.input_ub.append(temp_ub)


		# DEFINE INPUT VALUE(S) OF THE GLOBAL OPTIMUM
		if type(json_func["minimum_x"]) == str:
			local_var = {"d" : self.dimension}
			exec(json_func["minimum_x"], globals(), local_var)
			self.input_opt = local_var["minimum_x"]
		elif type(json_func["minimum_x"]) == int or type(json_func["minimum_x"]) == float:
			self.input_opt = [json_func["minimum_x"] for x in range(self.dimension)]
		elif json_func["minimum_x"] == None:
			self.input_opt = None
		elif len(json_func["minimum_x"]) == 1:
			self.input_opt = json_func["minimum_x"][0]
		else:
			self.input_opt = json_func["minimum_x"]

		# DEFINE FUNCTION'S PARAMETERS
		if json_func["parameters"] == None:
			self.has_parameters = False
		else:
			self.has_parameters = True
			self.parameters_description = json_func["parameters"]
			self.parameters_names = json_func["parameters_names"]
			# if parameters are given in input, set them
			if param != None:
				# check if the parameters given in input are parameters accepted by the function
				for param_name in param:
					if param_name not in self.parameters_names:
						raise sfuFunctionError(f"The selected function does not have such parameter: \"{param_name}\"")
				self.parameters = param
			else:
				# set default parameters values
				self.parameters = json_func["default_parameters"]
				for param in self.parameters:
					# if the parameter definition is python code, evaluate it
					if type(self.parameters[param]) == str:
						local_var = {}
						exec(self.parameters[param], globals(), local_var)
						self.parameters[param] = local_var[param]


		# DEFINE GLOBAL OPTIMUM
		self.param_opt = None
		# if definition of optimum is python code, evaluate it
		if type(json_func["minimum_f"]) == str:
			local_var = {"d" : self.dimension}
			exec(json_func["minimum_f"], globals(), local_var)
			self.opt = local_var["minimum_f"]
		# if minimum_f is a dict, then the function's optimum is defined only for those parameter's values.
		# The key of this dict is the parameter's name and its value is a dict with
		# parameter values in str format as key, and the global optimum as value.
		# e.g., {"m" : {"5": -10.1532, "7" : -10.4029}} will represents the fact that with m=5, the function's optimum is -10.1532 
		elif type(json_func["minimum_f"]) == dict:
			param_opt_dict = json_func["minimum_f"]
			param_opt_name = list(param_opt_dict.keys())[0]
			# overwrite the variable with the dict associated to the parameter
			param_opt_dict = param_opt_dict[param_opt_name]
			self.param_opt = (param_opt_name, param_opt_dict)
			# if optimum depends on function dimension, select the optimum corresponding to the function's dimension
			if self.param_opt[0] == "dimension":
				# if optimum is not defined for the chosen function dimension
				if str(self.dimension) not in self.param_opt[1]:
					self.opt = None
				else:
					self.opt = param_opt_dict[str(self.dimension)]
			# if optimum depends on function's parameter, select the optimum corresponding to such parameter value
			else:
				self.opt = self.param_opt[1][str(self.parameters[self.param_opt[0]])]
		# optimum is a float value
		else:
			self.opt = json_func["minimum_f"]	

		# keep string of R implementation of the function
		self.implementation = functions_by_name[self.name]

	def set_parameters(self, parameters):
		if not self.has_parameters:
			raise sfuFunctionError("Function does not accept parameters")
		for param in parameters:
			self.parameters[param] = parameters[param]

		# if changed parameter modify optimum, update it
		if self.param_opt != None and self.param_opt[0] in list(parameters.keys()):
			self.opt = self.param_opt[1][str(self.parameters[self.param_opt[0]])]

	# evaluate the function on input values
	def evaluate(self, inp):
		"""
		Parameters
		----------
		inp: list
			List of int or float values.

		Returns
		-------
		float
			Value of the function on input point "inp".
		"""
		# check if the input is valid
		if self.dimension == 1 and type(inp) != int and type(inp) != float:
			raise sfuFunctionError("Function input must be int or float")
		if self.dimension != 1 and (type(inp) != list or len(inp) != self.dimension):
			raise sfuFunctionError("Function input does not match function dimension")

		# run function
		if self.has_parameters:
			return self.implementation(inp, self.parameters)
		else:
			return self.implementation(inp)



# FUNCTIONS IMPLEMENTATIONS

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

functions_by_name = {
	"ackley_function" : ackley_function,
	"bukin_function_n._6" : bukin_function_n_6,
	"cross-in-tray_function": cross_in_tray_function,
	"drop-wave_function" : drop_wave_function,
	"eggholder_function" : eggholder_function,
	"gramacy_&_lee_(2012)_function" : gramacy_lee_2012_function,
	"griewank_function" : griewank_function,
	"holder_table_function" : holder_table_function,
	"langermann_function" : langermann_function,
	"levy_function" : levy_function,
	"levy_function_n._13" : levy_function_n_13,
	"rastrigin_function" : rastrigin_function,
	"schaffer_function_n._2" : schaffer_function_n_2,
	"schaffer_function_n._4" : schaffer_function_n_4,
	"schwefel_function" : schwefel_function,
	"shubert_function" : shubert_function,
	"bohachevsky_functions" : bohachevsky_functions,
	"perm_function_0,_d,_B" : perm_function_o_d_b,
	"rotated_hyper-ellipsoid_function" : rotated_hyper_ellipsoid_function,
	"sphere_function" : sphere_function,
	"sum_of_different_powers_function" : sum_of_different_powers_function,
	"sum_squares_function" : sum_squares_function,
	"trid_function" : trid_function,
	"booth_function" : booth_function,
	"matyas_function" : matyas_function,
	"mccormick_function" : mccormick_function,
	"power_sum_function" : power_sum_function,
	"zakharov_function" : zakharov_function,
	"three-hump_camel_function" : three_hump_camel_function,
	"six-hump_camel_function" : six_hump_camel_function,
	"dixon-price_function" : dixon_price_function,
	"rosenbrock_function" : rosenbrock_function,
	"de_jong_function_n._5" : de_jong_function_n_5,
	"easom_function" : easom_function,
	"michalewicz_function" : michalewicz_function,
	"beale_function" : beale_function,
	"branin_function" : branin_function,
	"colville_function" : colville_function,
	"forrester_et_al._(2008)_function" : forrester_et_al_2008_function,
	"goldstein-price_function" : goldstein_price_function,
	"hartmann_3-d_function" : hartmann_3_d_function,
	"hartmann_4-d_function" : hartmann_4_d_function,
	"hartmann_6-d_function" : hartmann_6_d_function,
	"perm_function_d,_B" : perm_function_d_b,
	"powell_function" : powell_function,
	"shekel_function" : shekel_function,
	"styblinski-tang_function" : styblinski_tang_function
}