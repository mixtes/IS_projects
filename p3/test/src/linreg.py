import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
        self.features = features.copy(deep=True)

        self.features.insert(0, 'c0', np.ones((len(features), 1)))

        self.features = self.features.to_numpy()
        self.coeff = np.zeros((self.features.shape[1], 1))

        self.target = target.to_numpy().reshape(-1, 1)


def plot_model_lines(feature_name, input_df, lrgd_plot, lr_model_plot):
    f_vals = np.linspace(input_df[feature_name].min(), input_df[feature_name].max(), 100)

    fixed_values = input_df.mean().to_dict()
    x_plot = pd.DataFrame([{**fixed_values, feature_name: val} for val in f_vals])

    y_sklearn = lr_model_plot.predict(x_plot)

    x_plot_gd = x_plot.copy()
    x_plot_gd.insert(0, 'c0', 1.0)
    x_plot_np = x_plot_gd.to_numpy()
    y_gd = x_plot_np @ lrgd_plot.coeff

    plt.figure(figsize=(8, 5))
    plt.plot(f_vals, y_sklearn, label='Sklearn Prediction', linewidth=2)
    plt.plot(f_vals, y_gd, label='Gradient Descent Prediction', linestyle='--')
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Price")
    plt.title(f"Prediction Line vs {feature_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


data = pd.read_csv("../data/house_prices_train.csv")
print("First five elements of the dataframe: \n" + str(data.head()) + "\n")
print("Last five elements of the dataframe: \n" + str(data.tail()) + "\n")

print("Dataframe info: ")
data.info()
print("\n")

print("Statistical information about every attribute: ")
print(data.describe())
print("\n")

data['Price'] = data['Price'] / 1000

input_attributes = ['Year_built', 'Area', 'Bath_no', 'Bedroom_no']
output_attribute = 'Price'

plt.figure(figsize=(12, 10))
for i, feature in enumerate(input_attributes):
    plt.subplot(2, 2, i+1)
    plt.scatter(data[feature], data[output_attribute], s=25, c='blue', edgecolors='black', alpha=0.7)
    plt.xlabel(feature, fontsize=11)
    plt.ylabel('Price (in $1000)', fontsize=11)
    plt.title(f'{output_attribute} vs {feature}', fontsize=12)

plt.tight_layout()
plt.show()

X = data.loc[:, input_attributes]
y = data[output_attribute]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()  # x' = (x - u) / o ---> u is the mean of the feature and o is standard deviation
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=input_attributes)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=input_attributes)

lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train_scaled_df, y_train)
learning_rate_const = 0.001
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rate_const, 20000)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled_df, y_train)

lrgd.features = X_val_scaled_df.copy()
lrgd.features.insert(0, 'c0', np.ones((len(X_val_scaled_df), 1)))
lrgd.features = lrgd.features.to_numpy()
lrgd.target = y_val.to_numpy().reshape(-1, 1)

custom_cost_val = lrgd.cost()

lr_model_coef_backup = lr_model.coef_.copy()
lr_model_intercept_backup = lr_model.intercept_

lr_model.coef_ = lrgd.coeff.flatten()[1:]
lr_model.intercept_ = lrgd.coeff.flatten()[0]
custom_score_val = lr_model.score(X_val_scaled_df, y_val)

lr_model.coef_ = lr_model_coef_backup
lr_model.intercept_ = lr_model_intercept_backup

sklearn_score_val = lr_model.score(X_val_scaled_df, y_val)

print("\n--- Gradient Descent Linear Regression ---")
print(f"Coefficients (including intercept): {lrgd.coeff.flatten()}")
print(f"Final cost (MSE/2): {mse_history[-1]:.6f}")
print(f"Validation R^2 score: {custom_score_val:.6f}")

print("\n--- sklearn Linear Regression ---")
print(f"Intercept: {lr_model.intercept_}")
print(f"Coefficents: {lr_model.coef_}")
print(f"Validation R^2 score: {sklearn_score_val:.6f}")

plot_model_lines("Area", X, lrgd, lr_model)

plt.figure(figsize=(10, 5))
plt.plot(mse_history, label='Cost (MSE/2) over iterations')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

y_pred_gd = lrgd.predict(X_val_scaled_df)
y_pred_sklearn = lr_model.predict(X_val_scaled_df)

plt.scatter(y_val, y_pred_gd, color='blue', alpha=0.7, label='Gradient Descent Predictions')
plt.scatter(y_val, y_pred_sklearn, color='red', alpha=0.5, label='sklearn Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)

plt.xlabel('Actual prices (Validation)')
plt.ylabel('Predicted prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

coeff_names = ['Intercept'] + input_attributes
gd_coeffs = lrgd.coeff.flatten()
sklearn_coeffs = [lr_model.intercept_] + list(lr_model.coef_)

x = np.arange(len(coeff_names))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, gd_coeffs, width, label='Gradient Descent')
plt.bar(x + width / 2, sklearn_coeffs, width, label='sklearn')

plt.xticks(x, coeff_names, rotation=15)
plt.ylabel("Coefficent Value")
plt.title("Model Coefficients Comparison")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
