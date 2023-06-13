import os, json, math
import sfu.functions_dataset.functions as functions_implementation

class JsonNotLoaded(Exception):
	pass

class sfuFunctionError(Exception):
	pass

# Define path of JSON functions data
json_filepath = os.path.join("functions_dataset", "functions_data.json")

def get_json_functions(name=None) -> dict:
	"""
	Returns the dictionary of all the functions inside the functions' json or of just the function "name"
	"""
	global json_filepath
	f = open(json_filepath)
	json_functions = json.load(f)
	f.close()
	if name != None:
		try:
			json_functions = json_functions[name]
		except KeyError:
			raise sfuFunctionError(f"The selected function does not exists: \"{name}\"")
	return json_functions

def search_functions(filters=None) -> list:
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
	
	Attributes
	----------
	name: str
		Name of the objective function
	dimension: int
		Dimension of the objective function
	has_input_domain_range: bool
		It is True when the range of the input domain is available
	input_lb: list
		Each index of the list represents a dimension, a value at that index represents the lower bound of the range of that dimension
	input_ub: list
		Each index of the list represents a dimension, a value at that index represents the upper bound of the range of that dimension
	input_opt: int, float, list, None
		Function's input value of the global optimum. When the value is list of lists, it represents the multiple possible input values for the global optimum
	has_parameters: bool
		It is True when the function accepts parameters
	parameters_description: str
		Description of the function's parameters
	parameters: dict
		Dictionary of parameters' values of the objective function
	parameters_names: list
		List of the parameters names
	param_opt: tuple, None
		Tuple with parameter name at idx 0, and dict of values for which opt is defined at idx 1.
	opt: float, None
		Function's global optimum
	implementation: function
		Implementation of the objective function

	Methods
	---------
	evaluate(inp): evaluate objective function on "inp" input values.
	update_parameters(parameters): update parameters of the objective function
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

		# save dictionary of data of the function
		json_func = json_functions[self.name]
		
		# DEFINE FUNCTION INPUT DIMENSION
		self.__set_input_dimension(dim, json_func["dimension"])

		# DEFINE INPUT DOMAIN RANGE
		self.__set_input_domain_range(json_func["input_domain_range"])
		
		# DEFINE INPUT VALUE(S) OF THE GLOBAL OPTIMUM
		self.__set_input_global_optimum(json_func["minimum_x"])

		# DEFINE FUNCTION'S PARAMETERS
		self.__set_parameters(param, json_func["parameters"])

		# DEFINE GLOBAL OPTIMUM
		self.__set_global_optimum(json_func["minimum_f"])

		# Obtain function implementation
		self.implementation = getattr(functions_implementation, self.name)


	def __set_input_dimension(self, dim, json_dim) -> None:
		"""
		Set function input dimension

		Parameters
		----------
		dim: int
			Function input dimension
		json_dim: int/str
			Default function input dimension taken defined in json functions file

		Returns
		-------
		None
		"""

		# if function input is d-dimensional by default, check the value dim given in input
		if json_dim == "d":
			# if dim is not given, raise an error
			if dim == None:
				raise sfuFunctionError("The function selected needs dimension definition")	
			else:
				# if dim is not an integer, raise an error
				try:
					dim = int(dim)
				except ValueError:
					raise sfuFunctionError(f"The given dimension is not an integer: {dim=}")

			# set the input dimension to dim
			self.dimension = dim
		# function input dimension is defined by default
		else:
			# if function input is not d-dimensional and a dim different from the default one is given in input, raise an error
			if dim != None and dim != int(json_dim):
				raise sfuFunctionError(f"The given dimension is not accepted by the selected function. The selected function supports only this dimension: {json_dim}")
			# set the default dimension contained in json file
			self.dimension = int(json_dim)


	def __set_input_domain_range(self, json_input_domain_range) -> None:
		"""
		Set input domain range if it exists

		Parameters
		----------
		json_input_domain_range: list/str
			List or str that defines the lower and upper bound for each function input dimension

		Returns
		-------
		None
		"""

		# input domain is not defined
		if json_input_domain_range == None:
			self.has_input_domain_range = False
		else:
			self.has_input_domain_range = True
			# init lists of input lower and upper bounds
			self.input_lb = []
			self.input_ub = []

			# if input domain range is defined explicitly for each dimension
			if len(json_input_domain_range) == self.dimension:
				# iterate over each function input dimension
				for range_domain in json_input_domain_range:
					# if the range is python code, evaluate it
					if type(range_domain[0]) == str:
						local_var = {"d" : self.dimension}
						exec(range_domain[0], globals(), local_var)
						self.input_lb.append(local_var["input_domain_range"][0])
						self.input_ub.append(local_var["input_domain_range"][1])
					# otherwise range is given in a 2 dimensional list [lb, ub]
					else:
						self.input_lb.append(range_domain[0])
						self.input_ub.append(range_domain[1])

			# if input domain range is defined for a single dimension and has to be expanded for each dimension
			else:
				range_domain = json_input_domain_range[0]
				# if the range is python code, evaluate it
				if type(range_domain[0]) == str:
					local_var = {"d" : self.dimension}
					exec(range_domain[0], globals(), local_var)
					temp_lb = local_var["input_domain_range"][0]
					temp_ub = local_var["input_domain_range"][1]
				# else take the lb and ub values
				else:
					temp_lb = range_domain[0]
					temp_ub = range_domain[1]

				# iterate over each dimension to assign lb and ub
				for x in range(self.dimension):
					self.input_lb.append(temp_lb)
					self.input_ub.append(temp_ub)


	def __set_input_global_optimum(self, json_min_x) -> None:
		"""
		Set function input (or inputs) corresponding to the global optimum

		Parameters
		----------
		json_min_x: int/float/str/list
			If str the input is described as python code to be evaluated.
			If list, it can contain a single list describing a unique minimum input, or it contain multiple lists corresponding to the multiple inputs associated to the global optimum

		Returns
		-------
		None
		"""

		# if python code evaluate it
		if type(json_min_x) == str:
			local_var = {"d" : self.dimension}
			exec(json_min_x, globals(), local_var)
			self.input_opt = local_var["minimum_x"]
		# if it is a single int or float, expand it for each function input dimension
		elif type(json_min_x) == int or type(json_min_x) == float:
			self.input_opt = [json_min_x for x in range(self.dimension)]
		# min_x does not exists
		elif json_min_x == None:
			self.input_opt = None
		# min_x is singular and given expicitly for each dimension
		elif len(json_min_x) == 1:
			self.input_opt = json_min_x[0]
		# min_x is more than one and given expicitly for each dimension
		else:
			self.input_opt = json_min_x

	
	def __set_parameters(self, param, json_parameters) -> None:
		"""
		Set function parameters

		Parameters
		----------
		param: dict
			Dictionary of "parameter_name : value" defined by the user
		json_parameters: dict
			Dictionary of the default parameters values 

		Returns
		-------
		None
		"""

		# function does not accept parameters
		if json_parameters == None:
			self.has_parameters = False
		# set function parameters
		else:
			self.has_parameters = True
			self.parameters_description = json_parameters["description"]
			self.parameters_names = json_parameters["parameters_names"]
			
			# if parameters are given in input, set them
			if param != None:
				# check if the parameters given in input are parameters accepted by the function
				for param_name in param:
					if param_name not in self.parameters_names:
						raise sfuFunctionError(f"The selected function does not have such parameter: \"{param_name}\"")
				self.parameters = param
			# otherwise set the fault ones
			else:
				self.parameters = json_parameters["default_parameters"]
				# check if the parameter value is python code, and evaluate it
				for param in self.parameters:
					if type(self.parameters[param]) == str:
						local_var = {}
						exec(self.parameters[param], globals(), local_var)
						self.parameters[param] = local_var[param]


	def __set_global_optimum(self, opt) -> None:
		"""
		Set global optimum

		Parameters
		----------
		opt: str/dict/float
			If str, global optimum is a python code to be evaluated
			If dict, global optimum is defined only for some paramters/input dimensions

		Returns
		-------
		None
		"""

		# init variable referencing to parameters that influences the global optimum
		self.param_opt = None

		# if definition of optimum is python code, evaluate it
		if type(opt) == str:
			local_var = {"d" : self.dimension}
			exec(opt, globals(), local_var)
			self.opt = local_var["minimum_f"]
		# if global optimum is a dict, then the function's optimum is defined only for those parameter's values.
		# The key of this dict is the parameter's name and its value is a dict with
		# parameter values in str format as key, and the global optimum as value.
		# e.g., {"m" : {"5": -10.1532, "7" : -10.4029}} will represents the fact that with m=5, the function's optimum is -10.1532 etc...
		elif type(opt) == dict:
			# take the name of the parameter that influences the opt
			param_opt_name = list(opt.keys())[0]
			# take the dict associated to the parameter
			param_opt_dict = opt[param_opt_name]
			# save the tuple which contains (parameter_name, dict with "parameter value: opt")
			self.param_opt = (param_opt_name, param_opt_dict)
			# if optimum depends on function input dimension, select the optimum corresponding to the function's dimension
			if self.param_opt[0] == "dimension":
				# if optimum is not defined for the chosen function dimension set it to None
				if str(self.dimension) not in self.param_opt[1]:
					self.opt = None
				# otherwise select the optimum corresponding to the function dimension
				else:
					self.opt = param_opt_dict[str(self.dimension)]
			# if optimum depends on function's parameter, select the optimum corresponding to the current parameter value
			else:
				self.opt = self.param_opt[1][str(self.parameters[self.param_opt[0]])]
		# otherwise optimum is a float value
		else:
			self.opt = opt	



	def update_parameters(self, parameters) -> None:
		"""
		Update parameter values

		Parameters
		----------
		parameters: dict
			Dictionary of "parameter_name : value"

		Returns
		-------
		None
		"""

		if not self.has_parameters:
			raise sfuFunctionError("Function does not accept parameters")
		for param in parameters:
			self.parameters[param] = parameters[param]

		# if changed parameter influences global optimum, update it
		if self.param_opt != None and self.param_opt[0] in list(parameters.keys()):
			self.opt = self.param_opt[1][str(self.parameters[self.param_opt[0]])]


	def evaluate(self, inp) -> float:
		"""
		Evaluate function on given input values

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