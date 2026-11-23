import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=2)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math, copy

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target 
def data_cleaning_HOUSE(X,y):
    df = X.copy()
    df['MedHouseVal'] = y
    print(f"清洗前数据量: {len(df)} 条")
    df_clean = df[df['MedHouseVal'] < 5.0].copy()
    # 剔除掉极端的低价房
    df_clean = df[(df['MedHouseVal'] >= 0.15) & (df['MedHouseVal'] < 5.0)]
    print(f"清洗后数据量: {len(df_clean)} 条 (剔除了 {len(df) - len(df_clean)} 条封顶数据)")
    # 重新分离出干净的 X 和 y
    X_clean = df_clean.drop('MedHouseVal', axis=1)
    y_clean = df_clean['MedHouseVal']
    return (X_clean,y_clean)
X_clean,y_clean = data_cleaning_HOUSE(X,y)


X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=2
)
print(f"\n训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
X_features = ['medic','age','avrooms','avebedrms','population','aveOccup','Latitude','Longitude']
X_train = X_train.to_numpy()


scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
X_train_norm = scaler.fit_transform(X_train) 
X_test_norm = scaler.transform(X_test)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
print("特征之间的数量级差距明显下降\n")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}\n ")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}\n")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")
## 作图部分
fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharey=True) # 2行4列，加大画布高度
axs = axs.flatten() # 将2x4的矩阵展平成一维数组，方便循环

for i in range(len(X_features)):
    axs[i].scatter(X_train[:, i], y_train, label='target')
    axs[i].scatter(X_train[:, i], y_pred, label='predict')
    axs[i].set_xlabel(X_features[i])
axs[0].set_ylabel("Price")
axs[4].set_ylabel("Price") 
axs[0].legend(loc='upper right') 

fig.suptitle("target versus prediction using z-score normalized model")
plt.tight_layout() # 防止 8 张图的标签重叠
plt.show()