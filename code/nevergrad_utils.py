import nevergrad as ng
import random
from utils import *

def run_nevergrad(function_obj, range_stopping_criteria, num_points):
	# define parameters for nevergrad optimizator
	if function_obj.has_input_domain:
		param = ng.p.Array(shape=(function_obj.dimension,), lower=function_obj.input_lb, upper=function_obj.input_ub)
	else:
		param = ng.p.Array(shape=(function_obj.dimension,))

	# init nevergrad optimizer
	optimizer = ng.optimizers.NGOpt10(parametrization=param, num_workers=num_points, budget=10**6)
	# for the first point, if input range is defined sample from it and suggest it to the optimizer
	if function_obj.input_lb != None and function_obj.input_ub != None:
		for n in range(num_points):
			point = []
			for dim in range(param.dimension):
				point.append(random.uniform(function_obj.input_lb[dim], function_obj.input_ub[dim]))
			optimizer.suggest(point)
	else:
		# else sample from range -100,100
		for n in range(num_points):
			point = []
			for dim in range(param.dimension):
				point.append(random.uniform(-100,100))
			optimizer.suggest(point)

	results = []
	# execute the first "range_stopping_criteria" executions
	for x in range(range_stopping_criteria):
		# obtain input points
		input_points = obtain_queries(optimizer, num_points)
		# evaluate objective function on input points
		computed_points = compute_points(function_obj, input_points)
		# save results and update optimizer
		results.append(min(computed_points))
		update_optimizer(optimizer, input_points, computed_points, num_points)

	# while stopping criteria is not met and the number of asks is below 100k, run nevergrad
	while (not stopping_criteria(results) and optimizer.num_ask <= 100000):
		#print(optimizer.num_ask)
		input_points = obtain_queries(optimizer, num_points)
		computed_points = compute_points(function_obj, input_points)
		results.append(min(computed_points))
		results = results[1:]
		#print(f"RESULT : {results}")
		update_optimizer(optimizer, input_points, computed_points, num_points)
	# return results
	recommendation = optimizer.provide_recommendation()
	return {"num evaluations" : optimizer.num_ask, "input opt" : list(*recommendation.args), "opt" : function_obj.evaluate(list(*recommendation.args))}

def obtain_queries(optimizer, num_points):
	input_points = []
	for i in range(num_points):
		query = optimizer.ask()
		input_points.append(list(*query.args))
	return input_points

def compute_points(function_obj, input_points):
	results = []
	for inp in input_points:
		results.append(function_obj.evaluate(inp))
	return results

def update_optimizer(optimizer, input_points, computed_points, num_points):
	for i in range(num_points):
		candidate = optimizer.parametrization.spawn_child(new_value = input_points[i])
		optimizer.tell(candidate, computed_points[i])

def stopping_criteria(samples):
	if std_dev(samples) < 1e-6:
		return True
	return False