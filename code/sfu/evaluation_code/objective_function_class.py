import json, math, os
from rpy2 import robjects
from subprocess import check_output
import shutil

from error_classes import *

json_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "functions/") + "functions.json"


def load_json():
	global json_filepath
	f = open(json_filepath)
	json_functions = json.load(f)
	f.close()
	return json_functions


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
	json_functions = load_json()
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
	input_domain: bool
		It is True when the range of the input domain is available.
	input_lb: list
		Each index of the list represents a dimension, a value at that index represents the lower bound of the range of that dimension.
	input_ub: list
		Each index of the list represents a dimension, a value at that index represents the upper bound of the range of that dimension.
	minimum_x: int, float, list, None
		Function's input value of the global optimum. When the value is list of lists, it represents the multiple possible input values for the global optimum.
	has_parameters: bool
		It is True when the function accepts parameters.
	parameters_description: str
		Description of the function's parameters.
	parameters: dict
		Dictionary of parameters' values of the objective function.
	parameters_names: list
		List of the parameters names.
	minimum_f_param: str, None
		Parameter name on which the global minimum depends.
	minimum_f: float, None
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
		self.name = name

		# obtain function json dictionary
		json_functions = load_json()
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
			self.input_domain = False
		else:
			self.input_domain = True
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


		# define input value of the global optimum
		if type(json_func["minimum_x"]) == str:
			local_var = {"d" : self.dimension}
			exec(json_func["minimum_x"], globals(), local_var)
			self.minimum_x = local_var["minimum_x"]
		elif type(json_func["minimum_x"]) == int or type(json_func["minimum_x"]) == float:
			self.minimum_x = [json_func["minimum_x"] for x in range(self.dimension)]
		elif json_func["minimum_x"] == None:
			self.minimum_x = None
		elif len(json_func["minimum_x"]) == 1:
			self.minimum_x = json_func["minimum_x"][0]
		else:
			self.minimum_x = json_func["minimum_x"]

		# define function's parameters
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


		# define function's global optimum
		self.minimum_f_param = None
		# if definition of global optimum is python code, evaluate it
		if type(json_func["minimum_f"]) == str:
			local_var = {"d" : self.dimension}
			exec(json_func["minimum_f"], globals(), local_var)
			self.minimum_f = local_var["minimum_f"]
		# if optimum depends on parameters/dimensions
		elif type(json_func["minimum_f"]) == dict:
			self.minimum_f_dict = json_func["minimum_f"]
			self.minimum_f_param = list(self.minimum_f_dict.keys())[0]
			self.minimum_f_param = (self.minimum_f_param, list(self.minimum_f_dict[self.minimum_f_param].keys()))

			# if optimum depends on function dimension, select the optimum corresponding to the function's dimension
			if self.minimum_f_param[0] == "dimension":
				# if optimum is not defined for the chosen function dimension
				if str(self.dimension) not in list(self.minimum_f_dict[self.minimum_f_param[0]].keys()):
					self.minimum_f = None
				else:
					self.minimum_f = self.minimum_f_dict[self.minimum_f_param[0]][str(self.dimension)]
			# if optimum depends on function parameter, select the optimum corresponding to such parameter value
			else:
				self.minimum_f = self.minimum_f_dict[self.minimum_f_param[0]][str(self.parameters[self.minimum_f_param[0]])]
		# optimum is a float value
		else:
			self.minimum_f = json_func["minimum_f"]	

		# keep string of R implementation of the function
		path_implementation = os.path.join(os.path.dirname(os.path.dirname(__file__)), "functions/") + json_func["filepath_r"]
		with open(path_implementation, 'r') as r:
			self.R_code = r.read()
			self.implementation_name = self.R_code.split('\n')[0].split()[0]
			r.close()

	# evaluate the function on input values
	def evaluate(self, inp):
		"""
		Parameters
		----------
		inp: list
			List of float values.

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
		if type(inp) != list:
			inp = [inp]

		# modify the R code to run it
		call = "\n" + self.implementation_name
		if self.dimension == 1:
			call = call + "({}".format(inp[0])
		else:
			inp = [str(x) for x in inp]
			call = call + "(c(" + ",".join(tuple(inp)) + ")"
		if self.has_parameters == True:
			for par in self.parameters_names:
				call = call + ",{}={}".format(par, self.parameters[par])
		call = call + ")"
		return robjects.r(self.R_code + call)[0]