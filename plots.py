import matplotlib.pyplot as plt
import pandas as pd


def bad_counts_plot(file_name):
    fd = pd.read_csv(file_name)
    ys = fd['bad_counts'].values
    xs = fd['beta'].values
    plt.scatter(xs, ys)
    #plt.title('Bad counts vs. $\\beta$')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{K}\\right]$')
    plt.ylabel('unsuccessful teleportation counts')
    #plt.show()

def prob_plot(file_name):
    fd = pd.read_csv(file_name)
    ys2 = fd['probability_of_0000000'].values
    xs2 = fd['beta'].values
    plt.scatter(xs2, ys2)
    #plt.title('Success vs. $\\beta$')
    plt.ylabel('successful measurement probability')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{K}\\right]$')
    #plt.show()


def state_fid_plot(file_name):
    fd = pd.read_csv(file_name)
    ys = fd['state_fid'].values
    xs = fd['beta'].values
    plt.scatter(xs, ys)
    plt.title('state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$')
    plt.xlabel('$\\beta$')
    plt.ylabel('fidelity')
    plt.savefig(file_name[:len(file_name)-4])
    # plt.show()


def generate_graphs(file_name):
    plot1 = plt.figure(1)
    bad_counts_plot(file_name)
    plt.title(file_name[len(file_name)-7:len(file_name)-4] + ' unsuccessful teleportation vs. $\\beta$')
    # plt.title(file_name[:len(file_name) - 4] + ' unsuccessful teleportation vs. $\\beta$')
    plt.savefig(file_name[:len(file_name)-4] + ' unsuccessful teleportation')

    plot2 = plt.figure(2)
    prob_plot(file_name)
    plt.title(file_name[len(file_name)-7:len(file_name)-4] + ' Successful measurement probability vs. $\\beta$')
    # plt.title(file_name[:len(file_name) - 4] + ' Successful measurement probability vs. $\\beta$')
    plt.savefig(file_name[:len(file_name)-4] + ' Successful measurement probability')

    # plt.show()

    plt.close('all')



for str_num in ['5', '10', '20']:
    state_fid_plot('mean_'+str_num+'_csv.csv')


























