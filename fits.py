import numpy as np
import pandas as pd
from iminuit import Minuit
from probfit import Chi2Regression
import matplotlib.pyplot as plt


lin_fun                = lambda x, a, b: (a * x) + b
power_fun              = lambda x, a, b: a * (x ** (b))
exp_fun                = lambda x, a, b: a * (np.exp(b * x))
cos_fun                = lambda x, a, b: a * (np.cos(b* x))
sin_fun                = lambda x, a, b: a * (np.sin(b * x))
poly2_fun              = lambda x, a, b, c: a * (x ** 2) + b * x + c
poly3_fun              = lambda x, a, b, c, d: a * (x ** 3) + b * (x ** 2) + c * x + d
const_fun              = lambda x, a: x*0+a
#cos2_fun               = lambda x,a,b: a * (np.cos(b*x))**2
# normalised_gauss_fun   = lambda x,a,b: 1/b*(np.sqrt(2*np.pi))*np.exp(-0.5*((x-a)/(b))**2)
# gauss_fun              = lambda x,a,b,C: C*np.exp(-0.5*((x-a)**2/b**2))
#normalised_poisson_fun = lambda x,a: (( a ** x ) * np.exp((-1)*a)) /(np.math.factorial((x)))
# poisson_fun            = lambda x,a,b: b *(( a ** x ) * np.exp((-1)*a)) /(np.math.factorial((x)))

names = ['linear', 'power', 'exp', 'cos', 'sin', 'poly2', 'poly3', 'constant']
latex_names = ['$a \\cdot x+b$', '$a \\cdot x^{b}$', '$a \\cdot e^{b \\cdot x}$', '$a \\cdot \\cos{(bx)}$', '$a \\cdot \\sin{(bx)}$',
               '$ax^{2}+bx+c$', '$ax^3+bx^2+cx+d$', '$a$']
models = [lin_fun, power_fun, exp_fun, cos_fun, sin_fun, poly2_fun, poly3_fun, const_fun]

# This is a change..

"""
# Code that generates fits for px0 and mean_20 (unsuccessful teleportation, successful measurement probability).

# mean_20, unsuccessful teleportation counts vs. beta - linear.

# Get needed vars for the fit.
file_name = 'mean_20.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'linear'
graph_title = 'mean_20 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\\frac{1}{K}$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:163]
ys = fd[y_col].values[:163]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # plt.savefig('mean20_till_163_unsuccessful_teleportation_vs_beta_fit_'+model_str)
    reg.show()


generate_fit(model_str, xs, ys)




# mean_20, successful measurement probability vs. beta - linear.

# Get needed vars for the fit.
file_name = 'mean_20.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'linear'
graph_title = 'mean_20 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\\frac{1}{K}$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:163]
ys = fd[y_col].values[:163]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()
#    plt.savefig('mean20_till_163_successful_measurement_probability_vs_beta_fit_'+model_str)


generate_fit(model_str, xs, ys)



# px0, successful measurement probability vs. beta - poly3.

# Get needed vars for the fit.
file_name = 'whole_25_px0.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'linear'
graph_title = 'px0 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\\frac{1}{K}$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values
ys = fd[y_col].values


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # plt.savefig('px0_successful_measurement_probability_vs_beta_fit_poly3')
    reg.show()


generate_fit(model_str, xs, ys)



# px0, unsuccessful teleportation counts vs. beta - poly3.

# Get needed vars for the fit.
file_name = 'whole_25_px0.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'linear'
graph_title = 'px0 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\\frac{1}{K}$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values
ys = fd[y_col].values


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # plt.savefig('px0_unsuccessful_teleportation_vs_beta_fit_poly3')
    reg.show()


generate_fit(model_str, xs, ys)
"""

























