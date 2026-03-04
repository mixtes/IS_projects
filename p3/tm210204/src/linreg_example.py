import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from matplotlib import cm

from sklearn.linear_model import LinearRegression


class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff = None        # wi: changes through iterations; at the beginning it is: [[0.]
        #                                                                                   [0.]]
        self.features = None     # X
        self.target = None       # y
        self.mse_history = None  # history of mean-square errors (J(w)) during the algorithm

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)  # reshape the 1D array into a 2D [shape (n + 1) x 1
        # reshape used like this makes arrays in this form: [1 2 3 4 5] (1 x 5)
        # into arrays in this form: [[1]
        #                            [2]
        #                            [3]
        #                            [4]
        #                            [5]] (5 x 1)

    def cost(self):  # J(w) error function
        predicted = self.features.dot(self.coeff)  # h(xi)
        s = pow(predicted - self.target, 2).sum()  # sum of (h(xi) - yi)^2
        return (0.5 / len(self.features)) * s      # J(w)

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self, learning_rate):  # one iteration of the algorithm
        predicted = self.features.dot(self.coeff)  # h(xi) matrix multiplication; in first iteration, coeff is zeros

        s = self.features.T.dot(predicted - self.target)  # T is transpose of the matrix;
        gradient = (1. / len(self.features)) * s  # calculated gradient using s
        self.coeff = self.coeff - learning_rate * gradient  # wi = wi - alpha * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate, num_iterations=100):
        self.mse_history = []
        for k in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    #
    def fit(self, features, target):
        self.features = features.copy(deep=True)  # features is type DataFrame; save X

        coeff_shape = len(features.columns) + 1  # shape of the numpy array, in this case 1 x 2 [one dimensional]
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)  # reshape to 2 x 1; all zeros

        self.features.insert(0, 'c0', np.ones((len(features), 1)))  # add a column of ones before the column of X

        self.features = self.features.to_numpy()  # make it into a (n + 1) x 2 array
        self.target = target.to_numpy().reshape(-1, 1)  # save y


dataframe = pd.read_csv('../data/house_prices_train_original.csv')

X = dataframe.loc[:, ['area']]  # X is a DataFrame with one column : 2D
y = dataframe['price']  # y is a Series : 1D
y = y / 1000

plt.figure('House prices')
plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
            edgecolors='black', linewidths=2, label='houses')

plt.xlabel('Area in m^2', fontsize=13)
plt.ylabel('Price in $', fontsize=13)
plt.title('House price for area')

plt.legend()

plt.tight_layout()
# plt.show()

spots = 200
estates = pd.DataFrame(data=np.linspace(0, max(X['area']), num=spots), columns=['area'])

lrgd = LinearRegressionGradientDescent()
lrgd.fit(X, y)
learning_rates = np.array([[0.17], [0.0000475]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 20)

plt.figure(1)
line, = plt.plot(estates['area'], lrgd.predict(estates), lw=5, c='red')
line.set_label('LRGD model')

lr_model = LinearRegression()
lr_model.fit(X, y)

line, = plt.plot(estates['area'], lr_model.predict(estates), lw=2, c='blue')
line.set_label('Ordinary LR model')

plt.legend(loc='upper left')
plt.show()


plt.figure('MS Error')
plt.plot(np.arange(0, len(mse_history), 1), mse_history)
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('MS error value', fontsize=13)
plt.xticks(np.arange(0, len(mse_history), 2))
plt.title('Mean-sqare error function')
plt.tight_layout()
plt.legend(['MS Error'])
plt.show()


spots = 100
c0, c1 = np.meshgrid(np.linspace(-50, 50, spots), np.linspace(0, 2, spots))

c0 = c0.flatten()
c1 = c1.flatten()
mse_values = []
for i in range(len(c0)):
    lrgd.set_coefficients(c0[i], c1[i])
    mse_values.append(lrgd.cost())

fig = plt.figure('MSE hyperplane')
ax = plt.subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(c0.reshape(spots, spots), c1.reshape(spots, spots),
                       np.array(mse_values).reshape(spots, spots),
                       cmap=cm.coolwarm)

min_mse_ind = mse_values.index(min(mse_values))
ax.scatter(c0[min_mse_ind], c1[min_mse_ind], mse_values[min_mse_ind],
           c='r', s=250, marker='^')
fig.colorbar(surf, shrink=0.5)
ax.set_xlabel('c0')
ax.set_ylabel('c1')
ax.set_zlabel('mse')
plt.title('MSE error')
plt.tight_layout()
plt.show()


example_estate_sqm = 122
example_estate = pd.DataFrame(data=[example_estate_sqm], columns=['area'])
lrgd.set_coefficients(res_coeff)
print(f'LRGD price for {example_estate_sqm}sqm house is '
      f'{lrgd.predict(example_estate)[0]:.2f} thousand $')
print(f'LRGD c0: {lrgd.coeff.flatten()[0]:2f}, '
      f'c1: {lrgd.coeff.flatten()[1]:.2f}')
print(f'LR price for {example_estate_sqm}sqm house is '
      f'{lr_model.predict(example_estate)[0]:.2f} thousand $')
print(f'LR c0: {lr_model.intercept_:.2f}, '
      f'c1: {lr_model.coef_[0]:.2f}')

lrgd.set_coefficients(res_coeff)
print(f'LRGD MSE: {lrgd.cost():.2f}')

c = np.concatenate((np.array([lr_model.intercept_]), lr_model.coef_))
lrgd.set_coefficients(c)
print(f'LR MSE: {lrgd.cost():.2f}')

lrgd.set_coefficients(res_coeff)

data_test = pd.read_csv('../data/house_prices_test_original.csv')
X = data_test[['area']]
y = data_test['price'] / 1000

lr_coef_ = lr_model.coef_
lr_int_ = lr_model.intercept_
lr_model.coef_ = lrgd.coeff.flatten()[1:]
lr_model.intercept_ = lrgd.coeff.flatten()[0]
print(f'LRGD score: {lr_model.score(X, y):.2f}')

lr_model.coef_ = lr_coef_
lr_model.intercept_ = lr_int_
print(f'LR score: {lr_model.score(X, y):.2f}')
