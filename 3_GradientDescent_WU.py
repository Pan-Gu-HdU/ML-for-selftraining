import numpy as np
import matplotlib.pyplot as plt
import math, copy

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2, 3.5, 4.1, 4.4, 5.2, 5.6, 5.7])
true_w = 200
true_b = 50
noise_scale = 50
noise = np.random.randn(len(x_train)) * noise_scale
y_train = true_w * x_train + true_b + noise
# y_train = np.array([250, 300, 480, 430, 630, 730])
def compute_cost(x, y, w, b): 
    m = x.shape[0]
    cost = 0
    cost_sum = 0
    for i in range (m):
        fwb = w*x[i]+b
        cost = (fwb-y[i])**2
        cost_sum += cost
    total_cost = (1/(2*m))*cost_sum
    return total_cost
def compute_GD(x,y,w,b):
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
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
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


w_init = 0
b_init = 0
# 迭代次数 iterations
iterations = 10000
tmp_alpha = 0.005

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_GD)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


# plot cost versus iteration 
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            
ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  
ax2.set_xlabel('iteration step') 

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, marker='x', c='r', s=100, label='Actual Data')
y_pred = w_final * x_train + b_final
plt.plot(x_train, y_pred, c='b', linewidth=2, label=f'Fitted Line: y={w_final:.2f}x+{b_final:.2f}')

plt.title('Linear Regression Fit using Gradient Descent')
plt.xlabel('X (Feature)')
plt.ylabel('Y (Target)')
plt.legend()
plt.grid(True)
plt.show()