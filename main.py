import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import copy, math
from sklearn.model_selection import train_test_split
from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from math import e

heart = pd.read_csv('heart.csv')
print("FIRST 5 ROWS\n")
print(heart.head(5),'\n')

print("DATA TYPES: \n")
print(heart.dtypes, '\n')

print("INFO: \n")
print(heart.info(),'\n')

print("DESCRIBE: \n")
print(heart.describe(),'\n')

print("CHECK IF NULL: \n")
print(heart.isnull().any(axis=1),'\n')

#correlation matrix

corr = heart.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,annot=True, cmap=sns.diverging_palette(220,20,as_cmap=True))
plt.show()
plt.close()
continousdata = heart[['age','trtbps','chol','thalachh','oldpeak']]
sns.pairplot(continousdata)
plt.show()
plt.close()
sns.catplot(x='output',y='oldpeak',hue='slp',kind='bar',data=heart)
plt.title('ST depression v. Heart disease')
plt.xlabel('Heart Disease',size=20)
plt.ylabel('ST Depression',size=20)
plt.show()
plt.close()

plt.figure(figsize=(12,8))
sns.violinplot(x='output',y='thalachh',hue='sex',inner='quartile',data=heart)
plt.title('Thalach Level v Heart Disease',fontsize=20)
plt.xlabel('Heart Disease Target' , fontsize = 16)
plt.ylabel('Thalach Level', fontsize = 16)
plt.show()
plt.close()

plt.figure(figsize=(12,8))
sns.boxplot(x='output',y='oldpeak',hue='sex',data=heart)
plt.title("ST depression level v Heart Disease",fontsize=20)
plt.xlabel('Heart Disease Target',fontsize=16)
plt.ylabel(ylabel='ST depression',fontsize=16)
plt.show()
plt.close()

posdata = heart[heart['output'] ==1]
posdata.describe()

negdata = heart[heart['output'] ==0]
posdata.describe()

print('Mean of positive patients ST depression: ',posdata['oldpeak'].mean())
print('Mean of negative patients ST depression: ',negdata['oldpeak'].mean())

print('Mean of positive patients thalach: ',posdata['thalachh'].mean())
print('Mean of negative patients thalach: ',negdata['thalachh'].mean())

X = heart[['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']]
y = heart[['output']]

#print(X)
#print(y)


#features = heartencoded[encodedcolnames]



#print(finalX.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train=np.array(y_train)
y_test=np.array(y_test)
encoder = BinaryEncoder(cols=['sex','cp','fbs','restecg','exng','slp','caa','thall'],drop_invariant=True)
X_train= encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
#print(heartencoded)
print("INFO: \n")
print(X_train.info(),'\n')
colnames = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
encodedcolnames = X_train.columns


ct = ColumnTransformer([('somename',StandardScaler(),['age','trtbps','chol','thalachh','oldpeak'])],remainder='passthrough')
ct.set_output(transform='pandas')
X_train = ct.fit_transform(X_train)
X_train.rename(columns = {'somename__age':'age','somename__trtbps':'trtbps','somename__chol':'chol','somename__thalachh':'thalach','somename__oldpeak':'oldpeak'}, inplace = True)
X_train.rename(columns= {'remainder__sex_0':'sex_0','remainder__sex_1':'sex_1','remainder__cp_0':'cp_0','remainder__cp_1':'cp_1','remainder__cp_2':'cp_2','remainder__fbs_0':'fbs_0','fbs_1':'fbs_1','remainder__restecg_0':'restecg_0','remainder__restecg_1':'restecg_1','remainder__exng_0':'exng_0','remainder__exng_1':'exng_1','remainder__slp_0':'slp_0','remainder__slp_1':'slp_1','remainder__caa_0':'caa_0','remainder__caa_1':'caa_1','remainder__caa_2':'caa_2','remainder__thall_0':'thall_0','remainder__thall_1':'thall_1','remainder__thall_2':'thall_2'},inplace=True)
#print(X_train[0])
#print(y_train[0])
X_test = ct.transform(X_test)
X_train=np.array(X_train)
X_test=np.array(X_test)
def compute_cost(x,y,w,b,regterm):
    m = x.shape[0]
    n = x.shape[1]
    cost = 0.0
    regcost = 0.0
    for i in range(m):
        f_wb_i = 1/(1+e **-(np.dot(x[i],w)+b))
        cost = cost + (y[i]*math.log(f_wb_i) + (1-y[i]) * math.log(1-f_wb_i))
    cost = (-cost / m)
    for i in range(n):
        regcost = regcost + (w[i] ** 2)
    regcost = regcost * (regterm / (2*m))
    return cost

def compute_gradient(x,y,w,b,regterm):
    m,n = x.shape # (number of examples, number of features)
    dj_dw = np.zeros(n,)
    dj_db = 0.

    for i in range(m):
        err = 1/( 1 + e **-(np.dot(x[i],w)+b)) - y[i]
        for j in range(n):
            dj_dw[j] = (dj_dw[j]+err[0] * x[i,j]) +(regterm/m) *w[j]
        dj_db = dj_db+err
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,regterm_in,cost_function,gradient_function,alpha,num_iters):

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    regterm = regterm_in

    for i in range(num_iters):
        dj_dw,dj_db = gradient_function(x,y,w,b,regterm)
        w = w-alpha*dj_dw
        b = b-alpha*dj_db

        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b,regterm))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration (i:4d) : Cost{J_history[-1]}")
    return w,b,J_history

initial_w = np.zeros(24)
initial_b = 0.

iterations = 10000
alpha = 1.0e-2
regularization_term = 0.5
w_final,b_final,J_hist = gradient_descent(X_train,y_train,initial_w,initial_b,regularization_term,compute_cost,compute_gradient,alpha,iterations)
print(w_final,b_final)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(8000 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()
plt.close()
predict = []

m,_ = X_test.shape
for i in range(m):
    predition = (1/(1+e **-(np.dot(X_test[i],w_final)+b_final)))
    predition = 1 if predition > 0.5 else 0
    predict.append(predition)
    print(predict[i])
    print("target value:", y_test[i])
predict = np.array(predict)
sum = 0
for i in range(len(predict)):
    if predict[i] == y_test[i][0]:
        sum+=1
    else:
        sum+=0
avg = sum/(len(predict)-1)


print('Train Accuracy:', avg * 100)









