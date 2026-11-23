import numpy as np
import matplotlib.pyplot as plt
import copy
import math
# matplotlib inline
def load_data():
    data = np.loadtxt("WuEnda_ML/dataset/ex1data1.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y
def load_data_multi():
    data = np.loadtxt("WuEnda_ML/dataset/ex1data2.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y
def compute_cost(x, y, w, b): 
    m = x.shape[0]
    cost = 0
    for i in range (m):
        fw = w*x[i] + b
        cost += (fw - y[i])**2
    total_cost = (1/(2*m))*cost
    return total_cost
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    djdw = 0
    djdb = 0
    for i in range (m):
         fwb = w*x[i]+b
         cost_w_sum = (fwb-y[i])*x[i]
         cost_b_sum = (fwb-y[i])
         djdw += cost_w_sum
         djdb += cost_b_sum
    djdw = djdw/m
    djdb = djdb/m
    return djdw,djdb
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing

x_train, y_train = load_data()
w_init = 0.
b_init = 0.
alpha = 0.01
iterations = 5000
w,b,J_h,p_h = gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations,compute_cost,compute_gradient)
print(w,b)
m = x_train.shape[0]
x_pre = np.zeros(m)
for i in range (m):
    x_pre = w*x_train+b
plt.plot(x_train,x_pre,c = 'b')
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Predict vs. True")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()