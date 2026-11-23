import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math, copy
#手搓有点难，直接去网页端看教程
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target 

# print("数据形状:", X.shape)
# print("特征列:", list(X.columns))
# print("房价形状:", y.shape)
# print("房价列:", y)
# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\n训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
X_features = ['medic','age','avrooms','avebedrms']
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
# fig,ax=plt.subplots(1, 4,figsize=(15,6),sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:,i],y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")
# plt.show()

def zscore_normalization(X):
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X-mu)/sigma
    return (X_norm,mu,sigma)    
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


X_norm, X_mu, X_sigma = zscore_normalization(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

fig,ax=plt.subplots(1, 4, figsize=(15, 6))
for i in range(len(ax)):
    ax[i].hist(X_train[:, i], bins=20) 
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count")
fig.suptitle("Distribution of features before normalization")
plt.tight_layout()
plt.show()

fig,ax=plt.subplots(1,4,figsize=(15,6))
for i in range(len(ax)):
    ax[i].hist(X_norm[:, i], bins=20) 
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle(f"Distribution of features after normalization")
plt.tight_layout()
plt.show()
