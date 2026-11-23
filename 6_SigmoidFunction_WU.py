import numpy as np
import matplotlib.pyplot as plt

'''
    z = wx + b
    g(z) = 1/(1+exp(-z))
    指数函数调用
    input_array = np.array([1,2,3])
    exp_array = np.exp(input_array)
'''
def sigmoid(z):
    gz = 1/(1+np.exp(-z))
    return gz

z_tmp = np.arange(-10,11)
y = sigmoid(z_tmp)
np.set_printoptions(precision=3)
print(np.c_[z_tmp,y])

fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(z_tmp,y,c = "b")
ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
plt.show()