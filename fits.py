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

names = ['linear', 'power', 'exp', 'cos', 'sin', 'poly2', 'poly3', 'constant']
latex_names = ['$a \\cdot x+b$', '$a \\cdot x^{b}$', '$a \\cdot e^{b \\cdot x}$', '$a \\cdot \\cos{(bx)}$', '$a \\cdot \\sin{(bx)}$',
               '$ax^{2}+bx+c$', '$ax^3+bx^2+cx+d$', '$a$']
models = [lin_fun, power_fun, exp_fun, cos_fun, sin_fun, poly2_fun, poly3_fun, const_fun]

# mean_20, state fidelity vs. beta - Linear.

# Get needed vars for the fit.
file_name = 'mean_20_csv.csv'
x_col = 'beta'
y_col = 'state_fid'
model_str = 'poly3'
graph_title = 'mean_20 state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'fidelity'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:125]
ys = fd[y_col].values[:125]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)


"""
# Here is all the code I used to generate ALL THE FITS OF THE PROJECT (by order of appearance in the paper).


# px0, successful measurement probability vs. beta - linear.

# Get needed vars for the fit.
file_name = 'whole_25_px0.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'linear'
graph_title = 'px0 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)


# px0, successful measurement probability vs. beta - poly2.

# Get needed vars for the fit.
file_name = 'whole_25_px0.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'poly2'
graph_title = 'px0 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)




# px0, unsuccessful teleportation counts vs. beta - poly2.

# Get needed vars for the fit.
file_name = 'whole_25_px0.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'poly2'
graph_title = 'px0 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)



# px0, state fidelity vs. beta - poly3.

# Get needed vars for the fit.
file_name = 'px0_fid.csv'
x_col = 'beta'
y_col = 'state_fid'
model_str = 'poly3'
graph_title = 'px0 state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'fidelity'

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
    reg.show()


generate_fit(model_str, xs, ys)





# px1, unsuccessful teleportation counts vs. beta - poly2.

# Get needed vars for the fit.
file_name = 'whole_25_px1.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'poly2'
graph_title = 'px1 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)



# py0, successful measurement probability vs. beta - linear.

# Get needed vars for the fit.
file_name = 'whole_25_py0.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'linear'
graph_title = 'py0 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)


# py0, successful measurement probability vs. beta - poly2.

# Get needed vars for the fit.
file_name = 'whole_25_py0.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'poly2'
graph_title = 'py0 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)




# py0, unsuccessful teleportation counts vs. beta - poly2 (square).

# Get needed vars for the fit.
file_name = 'whole_25_py0.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'poly2'
graph_title = 'py0 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)



# py0, state fidelity vs. beta - poly3.

# Get needed vars for the fit.
file_name = 'py0_fid.csv'
x_col = 'beta'
y_col = 'state_fid'
model_str = 'poly3'
graph_title = 'py0 state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'fidelity'

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
    reg.show()


generate_fit(model_str, xs, ys)



# py1, successful measurement probability vs. beta - linear.

# Get needed vars for the fit.
file_name = 'whole_25_py1.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'linear'
graph_title = 'py1 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)


# py1, successful measurement probability vs. beta - poly2.

# Get needed vars for the fit.
file_name = 'whole_25_py1.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'poly2'
graph_title = 'py1 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)



# py1, unsuccessful teleportation counts vs. beta - poly2 (square).

# Get needed vars for the fit.
file_name = 'whole_25_py1.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'poly2'
graph_title = 'py1 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)



# py1, state fidelity vs. beta - poly3.

# Get needed vars for the fit.
file_name = 'py1_fid.csv'
x_col = 'beta'
y_col = 'state_fid'
model_str = 'poly3'
graph_title = 'py1 state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'fidelity'

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
    reg.show()


generate_fit(model_str, xs, ys)




# pz1, successful measurement probability vs. beta - linear.

# Get needed vars for the fit.
file_name = 'whole_25_pz1.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'linear'
graph_title = 'pz1 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)


# pz1, successful measurement probability vs. beta - poly2.

# Get needed vars for the fit.
file_name = 'whole_25_pz1.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'poly2'
graph_title = 'pz1 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)



# pz1, unsuccessful teleportation counts vs. beta - poly2 (square).

# Get needed vars for the fit.
file_name = 'whole_25_pz1.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'poly2'
graph_title = 'pz1 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:42]
ys = fd[y_col].values[:42]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)



# pz1, state fidelity vs. beta - poly3.

# Get needed vars for the fit.
file_name = 'pz1_fid.csv'
x_col = 'beta'
y_col = 'state_fid'
model_str = 'poly3'
graph_title = 'pz1 state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'fidelity'

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
    reg.show()


generate_fit(model_str, xs, ys)



# mean_20, successful measurement probability vs. beta - linear.

# Get needed vars for the fit.
file_name = 'mean_20.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'linear'
graph_title = 'mean_20 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:86]
ys = fd[y_col].values[:86]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)



# mean_20, unsuccessful teleportation counts vs. beta - poly2 (square).

# Get needed vars for the fit.
file_name = 'mean_20.csv'
x_col = 'beta'
y_col = 'bad_counts'
model_str = 'poly2'
graph_title = 'mean_20 unsuccessful teleportation vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'unsuccessful teleportation counts'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:86]
ys = fd[y_col].values[:86]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)



# mean_20, state fidelity vs. beta - poly3.

# Get needed vars for the fit.
file_name = 'mean_20_csv.csv'
x_col = 'beta'
y_col = 'state_fid'
model_str = 'poly3'
graph_title = 'mean_20 state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'fidelity'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:125]
ys = fd[y_col].values[:125]


def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()


generate_fit(model_str, xs, ys)


# mean_20, successful measurement probability vs. beta - poly2.

# Get needed vars for the fit.
file_name = 'mean_20.csv'
x_col = 'beta'
y_col = 'probability_of_0000000'
model_str = 'poly2'
graph_title = 'mean_20 successful measurement probability vs. $\\beta$'
x_title = '$\\beta$ $\left[\\frac{1}{J}\\right]$'
y_title = 'successful measurement probability'

# Generate fit & graph.
fd = pd.read_csv(file_name)
xs = fd[x_col].values[:86]
ys = fd[y_col].values[:86]

def generate_fit(model, xs, ys):
    reg = Chi2Regression(models[names.index(model)], xs, ys)
    opt = Minuit(reg)
    opt.migrad()
    tit = graph_title + ', ' + 'fit: f(x)=' + latex_names[names.index(model)]
    plt.title(tit)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    reg.show()

generate_fit(model_str, xs, ys)
"""




























