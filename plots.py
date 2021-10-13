import matplotlib.pyplot as plt
import pandas as pd


def bad_counts_plot(file_name):
    fd = pd.read_csv(file_name)
    ys = fd['bad_counts'].values
    xs = fd['beta'].values
    plt.scatter(xs, ys)
    #plt.title('Bad counts vs. $\\beta$')
    plt.ylabel('unsuccessful teleportation counts')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
    #plt.show()

def prob_plot(file_name):
    fd = pd.read_csv(file_name)
    ys2 = fd['probability_of_0000000'].values
    xs2 = fd['beta'].values
    plt.scatter(xs2, ys2)
    #plt.title('Success vs. $\\beta$')
    plt.ylabel('successful measurement probability')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
    #plt.show()


def state_fid_plot(file_name):
    fd = pd.read_csv(file_name)
    ys = fd['state_fid'].values
    xs = fd['beta'].values
    plot1 = plt.figure(1)
    plt.scatter(xs, ys)
    plt.title('state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$')
    plt.ylabel('fidelity')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
    plt.savefig(file_name[:len(file_name)-4])
    # plt.show()
    plt.close(1)

def generate_graphs(file_name):
    plot1 = plt.figure(1)
    bad_counts_plot(file_name)
    # plt.title(file_name[len(file_name)-7:len(file_name)-4] + ' unsuccessful teleportation vs. $\\beta$')    # For constant initial states.
    plt.title(file_name[:len(file_name) - 4] + ' unsuccessful teleportation vs. $\\beta$')    # For means (5, 10, 20)
    plt.savefig(file_name[:len(file_name)-4] + ' unsuccessful teleportation')

    plot2 = plt.figure(2)
    prob_plot(file_name)
    # plt.title(file_name[len(file_name)-7:len(file_name)-4] + ' Successful measurement probability vs. $\\beta$')    # For constant initial states.
    plt.title(file_name[:len(file_name) - 4] + ' Successful measurement probability vs. $\\beta$')    # For means (5, 10, 20)
    plt.savefig(file_name[:len(file_name)-4] + ' Successful measurement probability')

    # plt.show()

    plt.close('all')



# generate_graphs('mean_20.csv')

"""
for str_num in ['5', '10', '20']:
    # generate_graphs('mean_'+str_num+'.csv')
    state_fid_plot('mean_'+str_num+'_csv.csv')
"""

"""
for eig_state in ['px0', 'px1', 'py0', 'py1', 'pz0', 'pz1']:
    file_name = 'whole_25_'+eig_state+'.csv'
    generate_graphs(file_name)
"""





















