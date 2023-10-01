import numpy as np
import pandas as pd

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# シグモイド関数の微分
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 多層パーセプトロンのクラス
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float64)  # 入力層から隠れ層への重み行列
        self.b1 = np.zeros(hidden_size,dtype=np.float64)  # 隠れ層のバイアスベクトル
        self.W2 = np.random.randn(hidden_size, output_size).astype(np.float64)  # 隠れ層から出力層への重み行列
        self.b2 = np.zeros(output_size,dtype=np.float64)  # 出力層のバイアスベクトル
        self.learning_rate = learning_rate  # 学習率

    def forward(self, x):
        # 入力層から隠れ層への順伝播
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # 隠れ層から出力層への順伝播
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, x, y):
        m = x.shape[0]  # バッチサイズ
        
        # 出力層の誤差（損失関数として二乗和誤差を使用）
        delta2 = (self.a2 - y) * sigmoid_derivative(self.z2)
        
        # 隠れ層の誤差
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.z1)
        
        # 出力層の重みとバイアスの更新
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        # 隠れ層の重みとバイアスの更新
        dW1 = np.dot(x.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        # 重みとバイアスの更新
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def accuracy(self, x, y):
        output = self.forward(x)
        predicted_labels = np.argmax(output, axis=1)
        true_labels = np.argmax(y, axis=1)
        correct_count = np.sum(predicted_labels == true_labels)
        total_count = x.shape[0]
        accuracy = correct_count / total_count
        return accuracy

    def get_weights(self):
        return self.W1, self.b1, self.W2, self.b2

# CSVファイルから訓練データを読み込む
print("write input file for learn : ")
input_data = pd.read_csv(input(),header=None,dtype=float)

print("write output file for learn : ")
output_data = pd.read_csv(input(),header=None,dtype=float)

x_train = input_data.values  # 入力データ
y_train = output_data.values  # 出力データ

# テストデータ
print("write input file for test : ")
x_test = pd.read_csv(input(),header=None,dtype=float)

print("write output file for test : ")
y_test = pd.read_csv(input(),header=None,dtype=float)

# MLPのインスタンス化
print("write node number of input layer : ")
inp=int(input())

print("write node number of middle layer : ")
mid=int(input())

print("write node number of output layer : ")
out=int(input())

print("write learning rate : ")
rate=int(input())

mlp = MLP(inp, mid, out, learning_rate=rate) 

# 学習の実行
print("write epoch number : ")
epochs = int(input())

for epoch in range(epochs):
    output = mlp.forward(x_train)
    mlp.backward(x_train, y_train)
    if epoch % 100 == 0:
        loss = np.mean((output - y_train) ** 2)
        accuracy = mlp.accuracy(x_train, y_train)
        print(f"Epoch: {epoch}, Loss: {loss}, Train Accuracy: {accuracy}")

# テストデータでの識別率の計算と表示
test_accuracy = mlp.accuracy(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# 学習後の重みとバイアスの出力
W1, b1, W2, b2 = mlp.get_weights()
print("学習後の重みとバイアス:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)


