import numpy as np
from sklearn import model_selection
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

def iris_type(name):
    map = {
        b'Iris-setosa': 0,
        b'Iris-versicolor': 1,
        b'Iris-virginica': 2
    }
    return map[name]


def loadData(filename, split_rate=0.8):
    """
    :param filename: input file path
    :param split_rate: train_test_split rate
    :return: x_train, x_test, y_train, y_test
    """
    data = np.loadtxt(
        fname=filename,
        dtype=float,  # 数据类型
        delimiter=',',  # 分隔符
        converters={4: iris_type}  # 将第5列文字分类替换为数字
    )
    x, y = np.split(data, (4,), axis=1)  # 按列切分，前四列与后面切开
    x = x[:, :2]  # iris数据集只有前两维是线性可分的
    return model_selection.train_test_split(x, y, random_state=1, train_size=split_rate)


def svm_classifier():
    return svm.SVC(
        kernel='linear',  # 设定为线性核,备选项：linear,rbf,sigmoid
        decision_function_shape='ovr',  # 决策函数
        C=0.8 # 误差惩罚系数
    )


def train(classifier, x_train, y_train):
    classifier.fit(x_train, y_train.ravel())


def print_eval(classifier, x_train, y_train, x_test, y_test):
    # 输出准确率
    print("train set prediction:%.3f" % classifier.score(x_train, y_train))
    print("test set prediction:%.3f" % classifier.score(x_test, y_test))
    # 计算决策函数的值 表示x到各个分割平面的距离
    print('decision_function:\n', classifier.decision_function(x_train)[:2])


def draw(classifier, x, y):
    """
    将模型的分割平面以二维图着色的方式呈现，将样本点与真实标签放入染色平面中供对比
    :param classifier: svm_classifier
    :param x: iris sepal length && width
    :param y: iris type
    :return: None
    """
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 底色
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])  # 样本点颜色

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 作为二维图的x轴
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 作为二维图的y轴
    # 生成网格采样点,将每个点染色来表示分类的结果
    epsilon = 0.01  # 格点精度
    x1, x2 = np.mgrid[x1_min:x1_max:epsilon, x2_min:x2_max:epsilon]
    # 将这些格点压缩为 n*2 数组，再交给模型预测
    grid_set = np.stack((x1.flat, x2.flat), axis=1)
    colored_set = classifier.predict(grid_set)   # 使用训练好的模型给格点标记
    # 使得 预测后的结果序列 和 x1 ，x2的形状一致
    colored_set = colored_set.reshape(x1.shape)
    print(colored_set)
    # 绘制模型预测的底色
    plt.pcolormesh(x1, x2, colored_set, cmap=cm_light)  # pcolormesh()会根据预测之后的colored_set的结果自动在cmap里选择颜色
    # 绘制真实的样本点
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), cmap=cm_dark)
    plt.xlabel('sepal length', fontsize=20)
    plt.ylabel('sepal width', fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Iris data classification via SVM', fontsize=30)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    x_train, x_test, y_train, y_test = loadData('data/iris/iris.data')
    # 训练模型
    model = svm_classifier()
    train(model, x_train, y_train)
    # 输出模型评估结果
    print_eval(model, x_train, y_train, x_test, y_test)
    # 绘制可视化分类图
    draw(classifier=model,
         x=np.vstack((x_train, x_test)),
         y=np.vstack((y_train, y_test)))

