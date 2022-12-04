import json, math, os
from rpy2 import robjects
from subprocess import check_output
import shutil

json_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "functions/") + "functions.json"


def load_json():
	global json_filepath
	f = open(json_filepath)
	json_functions = json.load(f)
	f.close()
	return json_functions


# HANDLE INPUT RANGE AND CHECK HOW NEVERGRAD BEHAVE WITH TRID FUNCTION WITH OPTIMUM AT -30

class objective_function():

	def __init__(self, name, dim=None, param=None):		
		self.name = name
		json_functions = load_json()
		if not self.name in json_functions.keys():
			raise sfuFunctionError("The function selected does not exists")
		json_func = json_functions[self.name]
		
		# Assign dimension
		self.dimension = json_func["dimension"]
		if self.dimension == "d":
			if dim == None:
				raise sfuFunctionError("The function selected needs dimension definition")
			self.dimension = dim
		else:
			self.dimension = int(self.dimension)

		# Assign input range
		if json_func["input_domain"] == None:
			self.input_domain = None
		else:
			self.input_domain = True
			self.input_lb = []
			self.input_ub = []
			# if input domain is defined for each dimension
			if len(json_func["input_domain"]) == self.dimension:
				for x in json_func["input_domain"]:
					if type(x[0]) == str:
						local_var = {"d" : self.dimension}
						exec(x[0], globals(), local_var)
						self.input_lb.append(local_var["input_domain"][0])
						self.input_ub.append(local_var["input_domain"][1])
					else:
						self.input_lb.append(x[0])
						self.input_ub.append(x[1])
			# if input domain is to be expanded for each dimension
			else:
				ran = json_func["input_domain"][0]
				if type(ran[0]) == str:
					local_var = {"d" : self.dimension}
					exec(ran[0], globals(), local_var)
					temp_lb = local_var["input_domain"][0]
					temp_ub = local_var["input_domain"][1]
				else:
					temp_lb = ran[0]
					temp_ub = ran[1]
				for x in range(self.dimension):
					self.input_lb.append(temp_lb)
					self.input_ub.append(temp_ub)


		# Assign minimum_x
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

		# Assign parameters
		if json_func["parameters"] == None:
			self.parameters = None
		else:
			self.parameters = True
			self.parameters_description = json_func["parameters"]
			
			if param != None:
				self.parameters_values = param
			else:
				# set default parameters values
				self.parameters_values = json_func["default_parameters"]
				for param in self.parameters_values:
					if type(self.parameters_values[param]) == str:
						local_var = {}
						exec(self.parameters_values[param], globals(), local_var)
						self.parameters_values[param] = local_var[param]
			self.parameters_names = json_func["parameters_names"]


		# Assign minimum_f value
		self.minimum_f_param = None
		if type(json_func["minimum_f"]) == str:
			local_var = {"d" : self.dimension}
			exec(json_func["minimum_f"], globals(), local_var)
			self.minimum_f = local_var["minimum_f"]
		elif type(json_func["minimum_f"]) == dict:
			self.minimum_f_dict = json_func["minimum_f"]
			self.minimum_f_param = list(self.minimum_f_dict.keys())[0]

			# if optimum depends on function dimension
			if self.minimum_f_param == "dimension":
				if str(self.dimension) not in list(self.minimum_f_dict[self.minimum_f_param].keys()):
					self.minimum_f = None
				else:
					self.minimum_f = self.minimum_f_dict[self.minimum_f_param][str(self.dimension)]

			else: 
				self.minimum_f = self.minimum_f_dict[self.minimum_f_param][str(self.parameters_values[self.minimum_f_param])]

		else:
			self.minimum_f = json_func["minimum_f"]	

		# Keep string of R implementation of the function
		path_implementation = os.path.join(os.path.dirname(os.path.dirname(__file__)), "functions/") + json_func["filepath_r"]

		with open(path_implementation, 'r') as r:
			self.R_code = r.read()
			self.implementation_name = self.R_code.split('\n')[0].split()[0]
			r.close()

	def set_parameters(self, parameters):
		if not self.parameters:
			raise sfuFunctionError("Function does not accept parameters")
		for param in parameters:
			self.parameters_values[param] = parameters[param]

		# if changed parameter modify optimum, update it
		if self.minimum_f_param != None and self.minimum_f_param in list(parameters.keys()):
			self.minimum_f = self.minimum_f_dict[self.minimum_f_param][str(self.parameters_values[self.minimum_f_param])]



	def evaluate(self, inp, param=None):
		if self.dimension == 1 and type(inp) != int and type(inp) != float:
			raise sfuFunctionError("Function input must be int or float")
		if self.dimension != 1 and (type(inp) != list or len(inp) != self.dimension):
			raise sfuFunctionError("Function input does not match function dimension")
		if type(inp) != list:
			inp = [inp]

		call = "\n" + self.implementation_name
		if self.dimension == 1:
			call = call + "({}".format(inp[0])
		else:
			inp = [str(x) for x in inp]
			call = call + "(c(" + ",".join(tuple(inp)) + ")"
		if self.parameters != None:
			for par in self.parameters_names:
				call = call + ",{}={}".format(par, self.parameters_values[par])
		call = call + ")"
		return robjects.r(self.R_code + call)[0]

class JsonNotLoaded(Exception):
	pass

class sfuFunctionError(Exception):
	pass

	

def search_function(filters=None):
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