# importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")


# checking for null values

data.isnull().sum()

# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')
plt.title('Study Hours vs Percentage Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

# plotting regressor plot to determine the relationship between feature and target
sns.regplot(x=data['Hours'], y=data['Scores'], data=data)
plt.title('Study Hours vs Percentage Scores')
plt.xlabel('Study Hours')
plt.ylabel('Percentage')
plt.show()

X = data.iloc[:, :-1].values  # Attribute
y = data.iloc[:, 1].values  # Labels

# Using Scikit-Learn's built-in train_test_split() method:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_new = y_train.reshape(-1, 1)
ones = np.ones([X_train.shape[0], 1])  # create a array containing only ones
X_train_new = np.concatenate([ones, X_train], 1)  # concatenate the ones to X matrix

# creating the theta matrix
# notice small alpha value
alpha = 0.01
iters = 5000

theta = np.array([[1.0, 1.0]])
print(theta)


def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# Gradient Descent
def gradientDescent(X, y, theta, alpha, iters):
    m = len(X)
    for i in range(iters):
        theta = theta - (alpha / m) * np.sum(((X @ theta.T) - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        # if i % 10 == 0:
        # print(cost)
    return theta, cost


g, cost = gradientDescent(X_train_new, y_train_new, theta, alpha, iters)
print("Intercept -", g[0][0])
print("Coefficient- ", g[0][1])
print("The final cost obtained after optimisation - ", cost)

# Plotting scatter points
plt.scatter(X, y, label='Scatter Plot')
axes = plt.gca()

# Plotting the Line
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1] * x_vals  # the line equation

plt.plot(x_vals, y_vals, color='red', label='Regression Line')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")

print("Coefficient -", regressor.coef_)
print("Intercept - ", regressor.intercept_)
# Plotting the regression line
line = regressor.coef_ * X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line, color='red', label='Regression Line')
plt.legend()
plt.show()

print(X_test)  # Testing data - In Hours
y_pred = regressor.predict(X_test)  # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

# Estimating training and test score
print("Training Score:", regressor.score(X_train, y_train))
print("Test Score:", regressor.score(X_test, y_test))

# plotting the grid to depict the actual and predicted value
df.plot(kind='bar', figsize=(7, 7))
plt.grid(which='major', linewidth='0.5', color='green')
plt.grid(which='minor', linewidth='0.5', color='black')
plt.show()

hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))
print()
