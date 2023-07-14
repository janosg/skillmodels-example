from skillmodels.likelihood_function import get_maximization_inputs
import pandas as pd
import numpy as np
import yaml
from time import time
from estimagic import maximize

# %%
import jax
jax.devices()


# %%
with open("model2.yaml") as y:
    model_dict = yaml.load(y, Loader=yaml.SafeLoader)

# %%
data = pd.read_stata("model2_simulated_data.dta")
data.set_index(["caseid", "period"], inplace=True)

# %% [markdown]
# ## Getting the inputs for ``estimagic.maximize``
#
# Skillmodels basically just has one public function called ``get_maximization_inputs``. When called with a model specification and a dataset it contains a dictionary with everything you need to maximize the likelihood function using estimagic.
#
# By everything you need I mean everything model specific. You should still use the optional arguments of ``maximize`` to tune the optimization.

# %%
max_inputs = get_maximization_inputs(model_dict, data)

# %% [markdown]
# ## Filling the Params Template
#
# Often you can greatly reduce estimation time by choosing good start parameters. What are good start parameters depends strongly on the model specifications, the scaling of your variables and the normalizations you make.
#
# If you have strong difficulties to pick good start values, you probably want to think again about the interpretability of your model parameters and possibly change the normalizations and scaling of your
# measurements.
#
# As a rule of thumb: If all measurements are standardized and, all fixed loadings are 1 and all fixed intercepts are 0 then one is a good start value for all free loadings and 0 is a good start value for all free intercepts.
#
# Measurement and shock standard deviations are better started slightly larger than you would expect them.
#
# Below I just load start parameters for the CHS example model that I filled out manually.

# %%
params_template = max_inputs["params_template"]
params_template.head()

# %%
index_cols = ["category", "period", "name1", "name2"]
chs_path = "chs_results.csv"
chs_values = pd.read_csv(chs_path)
chs_values.set_index(index_cols, inplace=True)
chs_values = chs_values[["chs_value", "good_start_value", "bad_start_value"]]
chs_values.head()

# %%
params = params_template.copy()
params["value"] = chs_values["chs_value"]
params.head()

# %% [markdown]
# ## Time compilation speed
#
# Skillmodels uses jax to just-in-time compile the numerical code and get a gradient of the likelihood function by automatic differentiation.
#
# There are several versions of the log likelihood function and its gradient:
#
# - **debug_loglike**: Is not compiled, can be debugged with a debugger, returns a lot of intermediate outputs and is slow.
# - **loglike**: Is compiled and fast but does not return intermediate outputs
# - **gradient**: Is compiled and fast, returns the gradient of loglike
# - **loglike_and_gradient**: Is compiled and fast and exploits synergies between loglike and gradient calculation. This is the most important one for estimation.

# %%
debug_loglike = max_inputs["debug_loglike"]
loglike = max_inputs["loglike"]
gradient = max_inputs["gradient"]
jacobian = max_inputs["jacobian"]
loglike_and_gradient = max_inputs["loglike_and_gradient"]

# %%
start = time()
debug_loglike_value = debug_loglike(params)
print(time() - start)
debug_loglike_value

# %%
start = time()
loglike_value = loglike(params)
print(time() - start)
loglike_value

# %%
start = time()
gradient_value = gradient(params)
print(time() - start)

# %%
start = time()
jacobian_value = jacobian(params)
print(time() - start)

# %%
start = time()
loglike_and_gradient_value = loglike_and_gradient(params)
print(time() - start)


# %%
constraints = max_inputs["constraints"]

additional_constraints = [
    {"query": "category == 'transition' & name1 == 'fac2' & name2 != 'fac2'",
     "type": "fixed", "value": 0},
    {"loc": "initial_states", "type": "fixed", "value": 0},
    {"queries": [f"period == {i} & name1 == 'Q1_fac1'" for i in range(8)],
     "type": "pairwise_equality"}
]


# %%
constraints = constraints + additional_constraints

# %% [markdown]
# ## Estimating the model

# %%
params["value"] = chs_values["good_start_value"]
loc = params.query("category == 'shock_sds' & name1 == 'fac3'").index
params.loc[loc, "lower_bound"] = 0.00
loglike(params)

# %%
res = maximize(
    criterion=loglike,
    params=params,
    algorithm="scipy_lbfgsb",
    criterion_and_derivative=loglike_and_gradient,
    constraints=constraints,
    logging=False,
    algo_options={"convergence.relative_criterion_tolerance": 1e-9}
)

# %%
res


