import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def find_last_value(file_path,name):
    """
    Search for the last occurrence of 'Fimp=' in a file and return the float value following it.
    
    :param file_path: Path to the file to be searched.
    :return: The float value following the last occurrence of 'Fimp=' or None if not found.
    """
    last_fimp_value = None
    if os.path.exists(file_path)==False:
        print('cannnot find the file {} !'.format(file_path))
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if name in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith(name):
                            # Try to extract and convert the number following 'Fimp='
                            try:
                                last_fimp_value = float(part.split('=')[1])
                            except ValueError:
                                # If conversion fails, continue to the next occurrence
                                continue
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    # print(last_fimp_value)
    return last_fimp_value


def doubleyaxis(B,U,T,order,maxit=20):

    filename='./data/{}_{}_{}_{}.dat'.format(U,T,B,order)
    data = np.loadtxt(filename).T
    m_arr=data[0,:maxit]
    diff_arr=data[1,:maxit]
    if maxit<=np.shape(m_arr)[0]:
        it=np.arange(maxit)
    else:
        it=np.arange(np.shape(m_arr)[0])


    fig, ax1 = plt.subplots()

    # Plot the first array
    ax1.plot(it, m_arr, 'r-')  # 'g-' is for green solid line
    ax1.set_xlabel('# of iteration')
    ax1.set_ylabel('Magnetization', color='r')
    # ax1.set_ylim(-1.1, 1.1)

    # Create a second y-axis for the same subplot
    ax2 = ax1.twinx()

    # Plot the second array
    ax2.plot(it, diff_arr, 'b-')  # 'b-' is for blue solid line
    ax2.set_ylabel('Difference', color='b')
    # ax2.set_yscale('log')
    ax1.set_title('Magnetization & Convergence: U={} T={} order={} B={}'.format(U,T,order,B))
    # Show the plot
    plt.show()

def mvsdiff(B,U,T,order):
    filename='./data/{}_{}_{}_{}.dat'.format(U,T,B,order)
    data = np.loadtxt(filename).T
    m_arr=data[0,:maxit]
    diff_arr=data[1,:maxit]
    plt.scatter(diff_arr, m_arr)
    # plt.axhline(y=n0loc22-n0loc11, color='r', linestyle='--')
    plt.xlabel("diff")
    plt.ylabel("magnetization")
    plt.title('magnetization: order={},U={},T={},B={}'.format(order,U,T,B))
    plt.show()

def all_orders(B,U,T,maxit=20):
    for order in np.arange(3)+1:
        filename='./data/{}_{}_{}_{}.dat'.format(U,T,B,order)
        data = np.loadtxt(filename).T
        m_arr=data[0,:maxit]
        diff_arr=data[1,:maxit]
        if maxit<=np.shape(m_arr)[0]:
            it=np.arange(maxit)
        else:
            it=np.arange(np.shape(m_arr)[0])
        plt.plot(it, m_arr, label='order {}'.format(order))
    plt.legend()
    plt.show()

U=10.0
T=0.24
B=0.01
order=3
maxit=20
all_orders(B,U,T)