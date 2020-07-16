from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage
import matplotlib.pyplot as plt
import numpy as np

from ime_fgs.plot import draw_graph
from ime_fgs.utils import col_vec, row_vec

## Initial position of drone
x_d = np.array([1, 2])
# x_d = x_d.reshape(1,2)
print(x_d)
## Initial position of anchors
x_A1 = np.array([0,0])
x_A2 = np.array([4,0])
x_A3 = np.array([2,4])
## Ground truth distance: Here x=(x_co-ordinate, y_co-ordinate)
z_12 = np.linalg.norm(x_A1-x_d)- np.linalg.norm(x_A2-x_d)
z_13 = np.linalg.norm(x_A1-x_d)- np.linalg.norm(x_A3-x_d)
z_23 = np.linalg.norm(x_A2-x_d)- np.linalg.norm(x_A3-x_d)
print(z_12,z_13,z_23 )

## Measurement Variance
meas_var = 0.1**2

##Sample noise for each measurement
z_12 =z_12 + np.random.multivariate_normal([0], [[meas_var]])
z_13 =z_13 + np.random.multivariate_normal([0], [[meas_var]])
z_23 =z_23 + np.random.multivariate_normal([0], [[meas_var]])

z_12_msg = GaussianMeanCovMessage(np.atleast_2d(z_12), [[meas_var]])
z_13_msg = GaussianMeanCovMessage(np.atleast_2d(z_13), [[meas_var]])
z_23_msg = GaussianMeanCovMessage(np.atleast_2d(z_23), [[meas_var]])

pos_var = 1 * np.eye(2)
x_d0 = x_d + np.random.multivariate_normal([0, 0], pos_var)
#initial position with the corresponding covariance matrix for the message
x_d_msg = GaussianMeanCovMessage(col_vec(x_d0), pos_var)

# Create all relevant nodes
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
#D_12_node = MatrixNode(D_12, name = "D_12")
#D_13_node = MatrixNode(D_13, name = "D_13")
#D_23_node = MatrixNode(D_23, name = "D_23")

B_12_node = MatrixNode(B_12, name = "B_12")
B_13_node = MatrixNode(B_13, name = "B_13")
B_23_node = MatrixNode(B_23, name = "B_23")

add_function12_node = AdditionNode(name = "add_function_node")
add_function13_node = AdditionNode(name = "add_function_node")
add_function23_node = AdditionNode(name = "add_function_node")
equality_node = EqualityNode(name="=", number_of_ports=4)
x_d_node = PriorNode(name="x_d")

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

# pass message through graph and relinearize
estimated_state_list12 = []
estimated_state_list13 = []
estimated_state_list23 = []

# use last state estimation for new state estimation
z = z_12_node.update_prior(z_12_msg)
add_function12_node.port_c.update(z)
d = D_12_node.port_a.update(D_12)
add_function12_node.port_b.update(d)
add_function12_node.port_a.update(z+d)
B_12_node.port_a.update(z+d)
b = B_12_node.port_b.update((z-d)/B_12)
equality_node.ports[1].update(b)

z = z_13_node.update_prior(z_13_msg)
add_function13_node.port_c.update(z)
d = D_13_node.port_a.update(D_13)
add_function13_node.port_b.update(d)
add_function13_node.port_a.update(z+d)
B_13_node.port_a.update(z+d)
b = B_13_node.port_b.update((z-d)/B_13)
equality_node.ports[2].update(b)

z = z_23_node.update_prior(z_23_msg)
add_function23_node.port_c.update(z)
d = D_23_node.port_a.update(D_23)
add_function23_node.port_b.update(d)
add_function23_node.port_a.update(z+d)
B_23_node.port_a.update(z+d)
b = B_23_node.port_b.update((z-d)/B_23)
equality_node.ports[3].update(b)

equality_node.ports[0].update(x_d_msg)

