# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 检查数据
#
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)

# 对训练集和测试集进行预处理，然后再训练网络

train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示训练集中的前 25 张图像，并在每张图像下显示类别名称。验证确保数据格式正确无误

if 0:

    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap='gray')
        plt.xlabel(class_names[train_labels[i]])

    plt.show()

# Linear stack of layers

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),             # 第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素）。可以将该层视为图像中像素未堆叠的行，并排列这些行。
    keras.layers.Dense(128, activation=tf.nn.relu),         # 在扁平化像素之后，该网络包含两个 tf.keras.layers.Dense 层的序列。这些层是密集连接或全连接神经层。第一个 Dense 层具有 128 个节点（或神经元）。
    keras.layers.Dense(10, activation=tf.nn.softmax)        # 第二个（也是最后一个）层是具有 10 个节点的 softmax 层，该层会返回一个具有 10 个概率得分的数组，这些得分的总和为 1。每个节点包含一个得分，表示当前图像属于 10 个类别中某一个的概率。
])

model.compile(optimizer=tf.train.AdamOptimizer(),           # 损失函数 - 衡量模型在训练期间的准确率。我们希望尽可能缩小该函数，以“引导”模型朝着正确的方向优化。
              loss='sparse_categorical_crossentropy',       # 优化器 - 根据模型看到的数据及其损失函数更新模型的方式。
              metrics=['accuracy'])                         # 指标 - 用于监控训练和测试步骤。以下示例使用准确率，即图像被正确分类的比例。

# 训练模型
# 将训练数据馈送到模型中。
# 模型学习将图像与标签相关联。
# test_images 数组会验证预测结果是否与 test_labels 数组中的标签一致。
model.fit(train_images, train_labels, epochs=5)


# 评估准确率

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 做出预测
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap='gray')

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()