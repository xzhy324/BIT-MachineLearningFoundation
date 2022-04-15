import sys
import os
import csv

max_depth = 0
positive_label = ''
negative_label = ''
positive_attr = ''
negative_attr = ''
train_major_label_of_all = ''
test_major_label_of_all = ''
attr_name_list = []


class TreeNode:
    # leave choice 只有当是叶子节点的时候有效; attr只有当非叶子节点时有效，此时左右子树必然存在
    def __init__(self, attr=None, left_node=None, right_node=None, leaf_choice=None):
        # 非叶子节点的有效字段
        self.attr = attr
        self.left_child = left_node
        self.right_child = right_node
        # 叶子节点的有效字段
        self.leave_choice = leaf_choice


def loadTsv(input_file):
    """
    返回值一览：
    features:属性与标签的名称【一行 (n_attr+1) * 1】;
    data:（多个样本\多维属性）二维矩阵【n_sample * n_attr】;
    labels:(多个样本\一个标签) 一列【n_sample * 1】
    """
    tsv = []
    with open(input_file, 'r') as fp:
        tsvreader = csv.reader(fp, delimiter='\t')
        for line in tsvreader:
            tsv.append(line)
    features = tsv[0]
    data = tsv[1:]
    labels = [line[-1] for line in data]
    data = [line[:-1] for line in data]
    return features, data, labels


def getMajorLabel(labels):
    pos = 0
    neg = 0
    for label in labels:
        if label == positive_label:
            pos += 1
        else:
            neg += 1
    return positive_label if pos >= neg else negative_label


def buildDecisionTree(data, labels, depth, data_type, unchosen_attr_name_list):
    # basecase1
    def all_data_hava_same_label(labels):
        for label in labels:
            if labels[0] != label:
                return False
        return True

    # basecase2
    def no_examples(data):
        return True if len(data) == 0 else False

    # basecase3
    def no_further_split_possible():
        return True if not unchosen_attr_name_list else False


    if all_data_hava_same_label(labels):
        leaf_choice = labels[0]
        return TreeNode(leaf_choice=leaf_choice)
    if no_examples(data):
        leaf_choice = getMajorLabel(labels)
        return TreeNode(leaf_choice=leaf_choice)
    if no_further_split_possible():
        leaf_choice = train_major_label_of_all if data_type == "train" else test_major_label_of_all
        return TreeNode(leaf_choice=leaf_choice)

    # 找到用于决策的attr以及其在data中的列序号 todo
    attr_name = ''
    maxIG = 0
    for attr in unchosen_attr_name_list:
        IG =  1

    attr_index = attr_name_list.index(attr_name)

    unchosen_attr_name_list.remove(attr_name)

    # 根据选出的分类属性的正负值划分训练样本
    left_data = []
    left_labels = []
    right_data = []
    right_labels = []
    left_pos_n = 0
    left_neg_n = 0
    right_pos_n = 0
    right_neg_n = 0
    for i, data_line in enumerate(data):
        if data_line[attr_index] == 1:
            left_data.append(data_line)
            left_labels.append(labels[i])
            if labels[i] == positive_label:
                left_pos_n += 1
            else:
                left_neg_n += 1
        elif data_line[attr_index] == 0:
            right_data.append(data_line)
            right_labels.append(labels[i])
            if labels[i] == positive_label:
                right_pos_n += 1
            else:
                left_neg_n += 1

    # 打印左节点信息，并递归生成左节点
    left_text = "{attr_name} = {attr_value}: [{pos_n} {pos_label}/{neg_n} {neg_label}]".format(
        attr_name=attr_name,
        attr_value=positive_attr,
        pos_n=left_pos_n,
        pos_label=positive_label,
        neg_n=left_neg_n,
        neg_label=negative_label
    )
    print("| " * (depth + 1), end='')
    print(left_text)
    left_node = buildDecisionTree(data=left_data, labels=left_labels, depth=depth + 1, data_type=data_type,unchosen_attr_name_list=unchosen_attr_name_list)

    # 打印右节点信息，并递归生成右节点
    right_text = "{attr_name} = {attr_value}: [{pos_n} {pos_label}/{neg_n} {neg_label}]".format(
        attr_name=attr_name,
        attr_value=negative_attr,
        pos_n=right_pos_n,
        pos_label=positive_label,
        neg_n=right_neg_n,
        neg_label=negative_label
    )
    print("| " * (depth + 1), end='')
    print(right_text)
    right_node = buildDecisionTree(data=right_data, labels=right_labels, depth=depth + 1, data_type=data_type,unchosen_attr_name_list=unchosen_attr_name_list)

    return TreeNode(attr=attr_name, left_node=left_node, right_node=right_node)


# 程序入口点
def entry(train_input, test_input, max_depth, train_output, test_output, metrics_output):
    # 加载数据和标签
    train_features, train_data, train_labels = loadTsv(train_input)
    test_features, test_data, test_labels = loadTsv(test_input)

    # 引用全局变量，并初始化值
    global positive_label, negative_label
    global train_major_label_of_all, test_major_label_of_all, attr_name_list
    global positive_attr, negative_attr
    positive_label = train_labels[0]
    for label in train_labels:
        if label != positive_label:
            negative_label = label
            break
    if positive_label < negative_label:  # 保证positive_name是字典序较为靠后的那一个，平局时优先打印positive_name
        tmp_str = positive_label
        positive_label = negative_label
        negative_label = tmp_str
    train_major_label_of_all = getMajorLabel(train_labels)
    test_major_label_of_all = getMajorLabel(test_labels)
    attr_name_list = train_features[:-1]
    positive_attr = train_data[0][0]
    for line in train_data:
        if line[0] != positive_attr:
            negative_attr = line[0]
            break

    # 清洗数据集的属性字段，转化为0和1
    for line in train_data:
        for i in range(len(line)):
            line[i] = 1 if line[i] == positive_attr else 0
    for line in test_data:
        for i in range(len(line)):
            line[i] = 1 if line[i] == positive_attr else 0

    # 建立决策树
    unchosen_attr_name_list = attr_name_list
    train_rootnode = buildDecisionTree(data=train_data, labels=train_labels, depth=0, data_type="train",unchosen_attr_name_list=unchosen_attr_name_list)
    unchosen_attr_name_list = attr_name_list
    test_rootnode = buildDecisionTree(data=test_data, labels=test_labels, depth=0, data_type="test",unchosen_attr_name_list=unchosen_attr_name_list)


# 处理外围的输入错误
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

    if not os.path.exists(train_input) or not os.path.exists(test_input):
        print("input file does not exist!")
        exit(0)
    if max_depth < 0:
        print("max_depth must be positive!")
        exit(0)

    entry(train_input, test_input, max_depth, train_output, test_output, metrics_output)
