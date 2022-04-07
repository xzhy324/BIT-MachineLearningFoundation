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
        delimiter=',',
        converters={4: iris_type}  # 将第5列文字分类替换为数字
    )
    x, y = np.split(data, (4,), axis=1)  # 按列切分，前四列与后面切开
    x = x[:, :2]  # iris数据集只有前两维是线性可分的
    return model_selection.train_test_split(x, y, random_state=1, train_size=split_rate)


def svm_classifier():
    return svm.SVC(
        kernel='linear',  # 设定为线性核,备选项：linear,rbf,sigmoid
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
    iris_feature = ['sepal length', 'sepal width']
    # 开始画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    # 生成网格采样点
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]

    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    # print('grid_test:\n', grid_test[:2])
    # 输出样本到决策面的距离
    z = classifier.decision_function(grid_test)
    print('the distance to decision plane:\n', z[:2])
    grid_hat = classifier.predict(grid_test)
    # 预测分类值 得到[0, 0, ..., 2, 2]
    # print('grid_hat:\n', grid_hat[:2])
    # 使得grid_hat 和 x1 形状一致
    grid_hat = grid_hat.reshape(x1.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 能够直观表现出分类边界

    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Iris data classification via SVM', fontsize=30)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    x_train, x_test, y_train, y_test = loadData('./data/iris/iris.data')
    # 训练模型
    model = svm_classifier()
    train(model, x_train, y_train)
    # 输出模型评估结果
    print_eval(model, x_train, y_train, x_test, y_test)
    # 绘制可视化分类图
    draw(classifier=model,
         x=np.vstack((x_train, x_test)),
         y=np.vstack((y_train, y_test)))

