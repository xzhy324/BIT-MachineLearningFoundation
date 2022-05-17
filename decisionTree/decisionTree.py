import sys
import os
import csv
from math import log2

# 全局变量 ======================================
# 这些值只会初始化一次，就不再改变

max_depth = 0  # 最大深度
positive_class = ''  # 正分类名称
negative_class = ''  # 负分类名称
positive_label = ''  # 标签的正值
negative_label = ''  # 标签的负值
train_major_class_value = ''  # 训练集中的主要分类


# ==============================================

class TreeNode:
    # leave choice 只有当是叶子节点的时候有效; attr只有当非叶子节点时有效，此时左右子树必然存在
    def __init__(self, label=None, left_node=None, right_node=None, leaf_choice=None):
        # 非叶子节点的有效字段
        self.label: str = label
        self.left_child: TreeNode = left_node
        self.right_child: TreeNode = right_node
        # 叶子节点的有效字段
        self.leaf_choice: str = leaf_choice


# 从tsv文件中加载数据集以及标签名称
def loadTsv(input_file):
    """
    返回值一览：
    labels:用于决策的变量的名称列表,形如[name_of_x1,name_of_x2,...,name_of_xn]
    dataSet:样本的labels以及class组合成的二维矩阵 ， 每一行形如 [x1,x2,...,xn,y]，x是labels（决策变量）的值，y是class(分类)的值
    """
    tsv = []
    with open(input_file, 'r') as fp:
        tsvReader = csv.reader(fp, delimiter='\t')
        for line in tsvReader:
            tsv.append(line)
    labels = tsv[0][:-1]
    dataSet = tsv[1:]
    return dataSet, labels


# 获取数据集中的主要分类的值
def getMajorClassValue(dataSet) -> str:
    pos, neg = countClass(dataSet)
    return positive_class if pos >= neg else negative_class


# 获取数据集中的正负分类的各自的数量
def countClass(dataSet) -> (int, int):
    pos, neg = 0, 0
    for data in dataSet:
        if data[-1] == positive_class:
            pos += 1
        else:
            neg += 1
    return pos, neg


# 根据某一列的取值，将符合条件的若干行取出，并删去用于筛选的列
def splitDataSet(dataSet, axis, value) -> []:
    ret_dataSet = []
    for data in dataSet:
        if data[axis] == value:
            new_data = data[:axis]
            new_data.extend(data[axis + 1:])
            ret_dataSet.append(new_data)
    return ret_dataSet


# 计算基于分类结果的信息熵
def entropy(dataSet) -> float:
    if not dataSet:     # 在计算infoLabel时，可能遇到空数据集，随便返回一个数
        return -1
    p = 0
    for data in dataSet:
        if data[-1] == positive_class:
            p += 1
    p /= len(dataSet)
    return -p * log2(p) - (1 - p) * log2(1 - p) if p != 0 and p != 1 else 0


# 计算某一决策变量的信息熵
def infoLabel(dataSet, labelIndex) -> float:
    dataSet_pos_label = [data for data in dataSet if data[labelIndex] == positive_label]
    dataSet_neg_label = [data for data in dataSet if data[labelIndex] == negative_label]
    Dp_size = len(dataSet_pos_label)
    Dn_size = len(dataSet_neg_label)
    D_size = len(dataSet)
    return Dp_size / D_size * entropy(dataSet_pos_label) + Dn_size / D_size * entropy(dataSet_neg_label)


# 递归生成决策树的核心算法
def buildDecisionTree(dataSet, labels, depth=0) -> TreeNode:
    # 结束条件（叶子节点标志）
    classValues = [data[-1] for data in dataSet]
    if classValues.count(classValues[0]) == len(classValues):  # class列全相同
        return TreeNode(leaf_choice=classValues[0])
    if len(dataSet) == 0:  # 没有样本
        leaf_choice = train_major_class_value
        return TreeNode(leaf_choice=leaf_choice)
    if depth == max_depth:  # 到达最大深度
        leaf_choice = getMajorClassValue(dataSet)
        return TreeNode(leaf_choice=leaf_choice)

    # 找到用于决策的label以及其在data中的列序号
    chosen_label = ''
    max_IG = 0
    info = entropy(dataSet)
    for index, label in enumerate(labels):
        IG = info - infoLabel(dataSet, index)
        if IG > max_IG:     # 严格大于才更新记录，以确保平票的时候，找到的是第一个出现的标签
            max_IG = IG
            chosen_label = label

    # 最大信息增益为0时，直接返回叶节点，此时有
    # 有infoLabel == entropy(dataSet) != 0
    # 可能的例子如下：
    # label | class
    # sunny | out
    # windy | out
    # sunny | in
    # windy | in
    if max_IG == 0:
        leaf_choice = getMajorClassValue(dataSet)
        return TreeNode(leaf_choice=leaf_choice)

    chosen_label_index = labels.index(chosen_label)  # 在删除前保存一下原始的选中的列序号，以供分割函数使用
    labels.remove(chosen_label)  # 删除选定的标签列

    left_dataSet = splitDataSet(dataSet, chosen_label_index, positive_label)  # 根据选定标签的正值划分数据集
    left_labels = labels[:]  # 注意python对于列表传的是引用，需要复制一份再递归
    pos_n, neg_n = countClass(left_dataSet)  # 为打印做准备
    print("| " * (depth + 1), end='')
    print("{label} = {label_value}: [{pos_n} {pos_class}/{neg_n} {neg_class}]".format(
        label=chosen_label,
        label_value=positive_label,
        pos_n=pos_n,
        pos_class=positive_class,
        neg_n=neg_n,
        neg_class=negative_class
    ))
    left_node = buildDecisionTree(left_dataSet, left_labels, depth + 1)

    right_dataSet = splitDataSet(dataSet, chosen_label_index, negative_label)  # 根据选定标签的负值划分数据集
    right_labels = labels[:]  # 注意python对于列表传的是引用，需要复制一份再递归
    pos_n, neg_n = countClass(right_dataSet)  # 为打印做准备
    print("| " * (depth + 1), end='')
    print("{label} = {label_value}: [{pos_n} {pos_class}/{neg_n} {neg_class}]".format(
        label=chosen_label,
        label_value=negative_label,
        pos_n=pos_n,
        pos_class=positive_class,
        neg_n=neg_n,
        neg_class=negative_class
    ))
    right_node = buildDecisionTree(right_dataSet, right_labels, depth + 1)

    return TreeNode(label=chosen_label, left_node=left_node, right_node=right_node)


# 对一组决策变量x根据决策树以及标签进行预测
def predict(labels: [str], treeNode: TreeNode, x: [str]) -> str:
    if not treeNode.left_child:   # 当前已经是叶子节点，直接返回结果
        return treeNode.leaf_choice
    index = labels.index(treeNode.label)
    if x[index] == positive_label:
        return predict(labels, treeNode.left_child, x)
    else:
        return predict(labels, treeNode.right_child, x)


# 计算误差并持久化存储
def calcErrorAndSave(treeRoot: TreeNode, dataSet, output_file: str, labels: [str]) -> float:
    error = 0
    with open(output_file, 'w') as fp:
        for data in dataSet:
            # 使用完整的test_labels和训练好的决策树对训练集预测
            result = predict(labels, treeRoot, data[:-1])
            fp.write(result + '\n')
            if result != data[-1]:
                error += 1
    error /= len(dataSet)
    return error


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("unmatched argument numbers")
        exit(0)

    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_output = sys.argv[6]

    # 处理外围的输入错误
    if not os.path.exists(train_input) or not os.path.exists(test_input):
        print("input file does not exist!")
        exit(0)
    if max_depth < 0:
        print("max_depth must be positive!")
        exit(0)

    # 加载数据和标签
    train_dataSet, train_labels = loadTsv(train_input)
    test_dataSet, test_labels = loadTsv(test_input)

    # 初始化全局变量的值
    positive_class = train_dataSet[0][-1]
    for data in train_dataSet:
        if data[-1] != positive_class:
            negative_class = data[-1]
            break
    if positive_class < negative_class:  # 保证positive_name是字典序较为靠后的那一个，平局时优先选择positive_name作为决策变量
        positive_class, negative_class = negative_class, positive_class
    positive_label = train_dataSet[0][0]
    for data in train_dataSet:
        if data[0] != positive_label:
            negative_label = data[0]
            break
    train_major_class_value = getMajorClassValue(train_dataSet)
    if positive_label < negative_label:  # 由于homework中未规定打印顺序，先按照字典逆序打印
        positive_label, negative_label = negative_label, positive_label
    # 建立训练集决策树
    pos_n, neg_n = countClass(train_dataSet)  # 为打印做准备
    print("[{pos_n} {pos_class}/{neg_n} {neg_class}]".format(
        pos_n=pos_n,
        pos_class=positive_class,
        neg_n=neg_n,
        neg_class=negative_class
    ))
    train_treeRoot = buildDecisionTree(train_dataSet, train_labels)  # 注意此时train_labels已经被破坏了

    # 计算误差
    error_test = calcErrorAndSave(train_treeRoot, test_dataSet, test_output, test_labels)
    error_train = calcErrorAndSave(train_treeRoot, train_dataSet, train_output, test_labels)

    with open(metrics_output, 'w') as fp:
        fp.write("error(train): {}\n".format(error_train))
        fp.write("error(test): {}\n".format(error_test))
