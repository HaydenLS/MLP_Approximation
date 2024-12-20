import numpy as np
import matplotlib.pyplot as plt


class AF:
    # Sigmoid
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return AF.sigmoid(x) * (1 - AF.sigmoid(x))

    # Tanh
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    # ReLU
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        # Производная ReLU: 1 для x > 0, иначе 0
        return np.where(x > 0, 1, 0)

    # Leaky reLU
    @staticmethod
    def leaky_relu(x):
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)


class MLP:
    def __init__(self, input_dim, l1_dim, output_dim):
        # Инициализация весов
        # Веса входного нейрона
        self.weights_input_hidden = np.random.rand(input_dim, l1_dim) - 0.5
        # Веса выходного нейрона у которой количество строк - количество нейронов в скрытом слое
        # Т.е. данная матрица имеет hidden_dim весов, например если нейронов в скр слое 5, то будет 5 весов
        self.weights_hidden_output = np.random.rand(l1_dim, output_dim) - 0.5

        # Сдвиги
        # Сдвиг для входного слоя - матрица 1 x hidden_dim, т е для каждого нейрона свой сдвиг.
        self.bias_hidden = np.zeros((1, l1_dim))
        # Сдвиг для выходного слоя, если в выходном слое 1 нейрон то сдвиг тоже 1 параметр.
        self.bias_output = np.zeros((1, output_dim))

    def forward(self, X, activate_function):
        # Прямое распространение

        # Значения, попадающие на вход в скрытый слой
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # Значения, на выходе из скрытого слоя, то есть функция hi = f(X*wi), где i - это i-ое значение матрицы входных точек
        self.hidden_output = activate_function(self.hidden_input)

        # Входные значения в выходной слой
        # otput_input будет выглядеть примерно так для 3 нейронов - out = h1*w4 + h2*w5 + h3*w6 + b1
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.output_input  # Линейная активация для регрессии
        return self.output

    def backward(self, X, y, lr, activate_derivative):
        # Ошибка = y - y_pred
        error = self.output - y
        # Градиенты
        # Градиент функции активации рассчитываем по цепному правилу.
        output_grad = error
        hidden_error = np.dot(output_grad, self.weights_hidden_output.T)
        hidden_grad = hidden_error * activate_derivative(self.hidden_input)

        # Обновление весов
        self.weights_hidden_output -= lr * np.dot(self.hidden_output.T, output_grad)
        self.bias_output -= lr * np.sum(output_grad, axis=0, keepdims=True)

        self.weights_input_hidden -= lr * np.dot(X.T, hidden_grad)
        self.bias_hidden -= lr * np.sum(hidden_grad, axis=0, keepdims=True)


    # Mean Squared Error
    def MSE(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # R^2 Score
    def score(self, y_true, y_pred):
        y_true_mean = np.mean(y_true)
        SS_tot = np.sum((y_true - y_true_mean)**2)
        Ss_res = np.sum((y_true - y_pred)**2)

        return 1 - Ss_res / SS_tot

    # Стандартный градиентный спуск
    def train(self, X, y, lr, epochs, activate_function=AF.tanh, activate_derivative=AF.tanh_derivative):
        for epoch in range(epochs):

            predictions = self.forward(X, activate_function)

            self.backward(X, y, lr, activate_derivative)

            if epoch % 500 == 0:
                loss = np.mean((y - predictions) ** 2)
                print(f"Epoch {epoch}, MSE Loss: {loss:.4f}")

        predictions = self.forward(X, activate_function)
        score = self.score(y, predictions)
        print(f"Model R^2 Score: {score:.4f}")

    # Метод SDG
    def train_SDG(self, X, y, lr, epochs, batch_size, activate_function=AF.tanh, activate_derivative=AF.tanh_derivative):
        for epoch in range(epochs):
            # Перемешиваем данные
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Обучение на батчах
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward и backward для текущего батча
                self.forward(X_batch, activate_function)
                self.backward(X_batch, y_batch, lr, activate_derivative)

            # Логирование потерь

            if epoch % 500 == 0:
                predictions = self.forward(X, activate_function)
                loss = self.MSE(y_true=y, y_pred=predictions)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # Итоговая точность нашей модели.
        predictions = self.forward(X, activate_function)
        score = self.score(y, predictions)
        print(f"Model R^2 Score: {score:.4f}")


class DoubleLayerMLP(MLP):
    def __init__(self, input_dim, l1_dim, l2_dim, output_dim):
        # Матрица весов 1 скрытого слоя.
        self.weights_l1 = np.random.rand(input_dim, l1_dim) - 0.5
        # Матрица весов 2 скрытого слоя.
        self.weights_l2 = np.random.rand(l1_dim, l2_dim) - 0.5
        # Матрица весов выходного слоя - один нейрон.
        self.weights_output = np.random.rand(l2_dim, output_dim) # output_dim - это всегда 1


        self.bias_l1 = np.zeros((1, l1_dim))
        self.bias_l2 = np.zeros((1, l2_dim))
        self.bias_output = np.zeros((1, output_dim))

    def forward(self, X, activate_function):
        self.activate_0 = X

        # Для 1 скрытого слоя
        self.z_1 = self.activate_0 @ self.weights_l1 + self.bias_l1
        self.activate_1 = activate_function(self.z_1)

        # Для 2 скрытого слоя
        self.z_2 = self.activate_1 @ self.weights_l2 + self.bias_l2
        self.activate_2 = activate_function(self.z_2)

        # Для выходного слоя
        self.z_3 = self.activate_2 @ self.weights_output + self.bias_output
        self.activate_output = self.z_3 # Не используем функцию активации на выходе

        return self.activate_output


    def backward(self, X, y, lr, activate_derivative):
        # Дельта на выходе
        delta_output = self.activate_output - y

        # Скрытые слои

        # Дельта на 2 слое
        delta_2_input = delta_output @ self.weights_output.T
        delta_2 = delta_2_input * activate_derivative(self.z_2)

        # Дельта на 1 слое
        delta_1_input = delta_2 @ self.weights_l2.T
        delta_1 = delta_1_input * activate_derivative(self.z_1)

        # Обновление весов
        self.weights_l1 -= lr * (self.activate_0.T @ delta_1)
        self.bias_l1 -= lr * np.sum(delta_1, axis=0, keepdims=True)

        self.weights_l2 -= lr * (self.activate_1.T @ delta_2)
        self.bias_l2 -= lr * np.sum(delta_2, axis=0, keepdims=True)

        self.weights_output -= lr * (self.activate_2.T @ delta_output)
        self.bias_output -= lr * np.sum(delta_output, axis=0, keepdims=True)







# Пример функции
def func(x):
    #return np.sin(x / 2) + np.cos(x) ** 2
    return np.sin(x**2) + x



# Генерация данных
X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)  # Входные данные
y_true = func(X)  # Выходные данные

# Иницализация модели
input_dim = 1
l1_dim = 5
l2_dim = 5
output_dim = 1
#model = MLP(input_dim, l1_dim, output_dim)
model = DoubleLayerMLP(input_dim, l1_dim, l2_dim, output_dim)


# Обучение модели
activate_function = AF.tanh
activate_derivative = AF.tanh_derivative
model.train(X, y_true, 0.001, 5000, activate_function, activate_derivative)
#model.train_SDG(X, y_true, 0.01, 5000, 20, activate_function, activate_derivative)

# График функции после обучения
plt.plot(X, y_true, label="Actual function")
plt.plot(X, model.forward(X, activate_function), label="MLP approximation")
plt.legend()
plt.show()


