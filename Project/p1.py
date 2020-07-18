from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage
import matplotlib.pyplot as plt
import numpy as np

from ime_fgs.plot import draw_graph
from ime_fgs.utils import col_vec, row_vec


def generate_msg(z_12, z_13, z_23, meas_var):
    z_12_msg = GaussianMeanCovMessage(np.atleast_2d(z_12), [[meas_var]])
    z_13_msg = GaussianMeanCovMessage(np.atleast_2d(z_13), [[meas_var]])
    z_23_msg = GaussianMeanCovMessage(np.atleast_2d(z_23), [[meas_var]])
    
    return z_12_msg, z_13_msg, z_23_msg

def plot_pos(Anchor1, Anchor2, Anchor3, X_est, X_real):
    '''
    Function to plot the environment
    
    @input: 
        Anchor1: the pos of the first Anchor
        Anchor2: the pos of the second Anchor
        Anchor3: the pos of the third Anchor
        X_est: the estimated pos of the drone
        X_real: the real pos of the drone
    '''
    plt.plot([Anchor1[0], Anchor2[0], Anchor3[0]], [Anchor1[1], Anchor1[2], Anchor1[3]], 'ro')
    plt.plot([X_real[0]], [X_real[1]], 'go')
    plt.plot([X_est[0]], [X_est[1]], 'bo')
    plt.axis([0, 6, 0, 20])
    plt.legend('Anchors','Real Position', 'Estimated Pos')
    plt.show()

def calculate_error(X_est, X_real):
    '''
    Function to calculate the error between an estimated 2d pos 
    and the real 2d pos
    
    @input: 
        X_est: the estimated position of dim: 1 x 2
        X_real: the real position of dim: 1 x 2
        
    @output:
        error: a scalar error feedback
    '''
    errors = np.absolute(X_real - X_est)
    error = errors[0] + errors[1]
    
    return error

def plot_error(iterations, error):
    '''
    Function to plot error calculations over some iterations
    
    @input 
        iterations: number of iterations
        error: error vector of dim: num_iterations x 1
    '''
    iteration = np.linspace(0,iterations)
    plt.plot(iteration, error, color = 'r')
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.show()

def calculate_dist(x_d, x_A1, x_A2, x_A3):
    z_12 = np.linalg.norm(x_A1-x_d)- np.linalg.norm(x_A2-x_d)
    z_13 = np.linalg.norm(x_A1-x_d)- np.linalg.norm(x_A3-x_d)
    z_23 = np.linalg.norm(x_A2-x_d)- np.linalg.norm(x_A3-x_d)
    
    return z_12, z_13, z_23

def add_noise(meas_var, z_12, z_13, z_23):
    # add noise to 3 measurements
    z_12 = z_12 + np.random.multivariate_normal([0], [[meas_var]])
    z_13 = z_13 + np.random.multivariate_normal([0], [[meas_var]])
    z_23 = z_23 + np.random.multivariate_normal([0], [[meas_var]])
    
    return z_12, z_13, z_23

def add_nodes(x_A1,x_A2, x_A3):
    x_dtilted = np.array([3, 3])
    D_12 = np.linalg.norm(x_A1-x_dtilted)- np.linalg.norm(x_A2-x_dtilted) - row_vec((((x_A2-x_dtilted)/np.linalg.norm(x_A2-x_dtilted))-((x_A1-x_dtilted)/np.linalg.norm(x_A1-x_dtilted))))@col_vec(x_dtilted)
    B_12 = row_vec((x_A2-x_dtilted)/np.linalg.norm(x_A2-x_dtilted)-(x_A1-x_dtilted)/np.linalg.norm(x_A1-x_dtilted))
    D_13 = np.linalg.norm(x_A1-x_dtilted)- np.linalg.norm(x_A3-x_dtilted) - row_vec((((x_A3-x_dtilted)/np.linalg.norm(x_A3-x_dtilted))-((x_A1-x_dtilted)/np.linalg.norm(x_A1-x_dtilted))))@col_vec(x_dtilted)
    B_13 = row_vec(((x_A3-x_dtilted)/np.linalg.norm(x_A3-x_dtilted)-(x_A1-x_dtilted)/np.linalg.norm(x_A1-x_dtilted)))
    D_23 = np.linalg.norm(x_A2-x_dtilted)- np.linalg.norm(x_A3-x_dtilted) - row_vec((((x_A3-x_dtilted)/np.linalg.norm(x_A3-x_dtilted))-((x_A2-x_dtilted)/np.linalg.norm(x_A2-x_dtilted))))@col_vec(x_dtilted)
    B_23 = row_vec(((x_A3-x_dtilted)/np.linalg.norm(x_A3-x_dtilted)-(x_A2-x_dtilted)/np.linalg.norm(x_A2-x_dtilted)))
    
    z_12_node = PriorNode(name = "z_12_node")
    z_13_node = PriorNode(name = "z_13_node")
    z_23_node = PriorNode(name = "z_23_node")
    D_12_node = PriorNode(name="D_12")
    D_13_node = PriorNode(name="D_13")
    D_23_node = PriorNode(name="D_23")
    B_12_node = MatrixNode(B_12, name = "B_12")
    B_13_node = MatrixNode(B_13, name = "B_13")
    B_23_node = MatrixNode(B_23, name = "B_23")
    
    add_function12_node = AdditionNode(name = "add_function_node")
    add_function13_node = AdditionNode(name = "add_function_node")
    add_function23_node = AdditionNode(name = "add_function_node")
    equality_node = EqualityNode(name="=", number_of_ports=4)
    x_d_node = PriorNode(name="x_d")
    
    return x_d_node, equality_node, add_function23_node, add_function13_node, add_function12_node, B_23_node, B_13_node, B_12_node, D_23_node, D_13_node, D_12_node, z_23_node, z_13_node, z_12_node, D_12, D_13, D_23 
    
if __name__ == "__main__":
    ## Initial position of drone
    x_d = np.array([1, 2])
    # x_d = x_d.reshape(1,2)
    print(x_d)
    ## Initial position of anchors
    x_A1 = np.array([0,0])
    x_A2 = np.array([4,0])
    x_A3 = np.array([2,4])
    ## Ground truth distance: Here x=(x_co-ordinate, y_co-ordinate)
    z_12,z_13,z_23 = calculate_dist(x_d, x_A1, x_A2, x_A3)
    print(z_12,z_13,z_23)
    
    ## Measurement Variance
    meas_var = 0.1**2
    
    #Sample noise for each measurement
    z_12, z_13, z_23 = add_noise(meas_var, z_12, z_13, z_23)
    
    z_12_msg, z_13_msg, z_23_msg = generate_msg(z_12, z_13, z_23, meas_var)
    
    pos_var = 1 * np.eye(2)
    x_d0 = x_d + np.random.multivariate_normal([0, 0], pos_var)
    
    #initial position with the corresponding covariance matrix for the message
    x_d_msg = GaussianMeanCovMessage(col_vec(x_d0), pos_var)
    
    # Create all relevant nodes
    x_d_node, equality_node, add_function23_node, add_function13_node, add_function12_node, B_23_node, B_13_node, B_12_node, D_23_node, D_13_node, D_12_node, z_23_node, z_13_node, z_12_node, D_12, D_13, D_23 = add_nodes(x_A1,x_A2, x_A3)    
    
    # Connect the nodes together with the .connect function
    
    x_d_node.port_a.connect(equality_node.ports[0])
    z_12_node.port_a.connect(add_function12_node.port_c)
    z_13_node.port_a.connect(add_function13_node.port_c)
    z_23_node.port_a.connect(add_function23_node.port_c)
    D_12_node.port_a.connect(add_function12_node.port_b)
    D_13_node.port_a.connect(add_function13_node.port_b)
    D_23_node.port_a.connect(add_function23_node.port_b)
    B_12_node.port_b.connect(add_function12_node.port_a)
    B_13_node.port_b.connect(add_function13_node.port_a)
    B_23_node.port_b.connect(add_function23_node.port_a)
    B_12_node.port_a.connect(equality_node.ports[1])
    B_13_node.port_a.connect(equality_node.ports[2])
    B_23_node.port_a.connect(equality_node.ports[3])
    draw_graph(x_d_node)
    
    ## pass message through graph and relinearize
    x_d_msg_est_list = []
    
    #for  in ():
    #print('z_12', z_12, 'z_12_msg',z_12_msg, 'D_12', D_12, 'x_d_msg', x_d_msg)
    for i in range(10):
        
        D_12_msg = GaussianMeanCovMessage(np.atleast_2d(D_12), [[0]])
        z_12_node.update_prior(z_12_msg)
        D_12_node.update_prior(D_12_msg)
        add_function12_node.port_a.update(GaussianWeightedMeanInfoMessage)
        B_12_node.port_a.update()
        
        D_13_msg = GaussianMeanCovMessage(np.atleast_2d(D_13), [[0]])
        z_13_node.update_prior(z_13_msg)
        D_13_node.update_prior(D_13_msg)
        add_function13_node.port_a.update(GaussianWeightedMeanInfoMessage)
        B_13_node.port_a.update()
    
        D_23_msg = GaussianMeanCovMessage(np.atleast_2d(D_23), [[0]])
        z_23_node.update_prior(z_23_msg)
        D_23_node.update_prior(D_23_msg)
        add_function23_node.port_a.update(GaussianWeightedMeanInfoMessage)
        B_23_node.port_a.update()
    
        
        equality_node.ports[0].update()
        
        x_d_node = equality_node.ports[0]
        X_est = equality_node.ports[0].marginal(GaussianMeanCovMessage).mean
        x_d_msg_est_list.append(X_est)
        z_12,z_13,z_23 = calculate_dist(X_est, x_A1, x_A2, x_A3)
        z_12_msg, z_13_msg, z_23_msg = generate_msg(z_12, z_13, z_23, 0)
        
        print(equality_node.ports[0].marginal(GaussianMeanCovMessage).mean)
    
