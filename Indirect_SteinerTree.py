import sys
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sko.GA import GA
import pandas as pd


# %% 读取文件
# 实例化
root = tk.Tk()
root.withdraw()

# 获取文件夹路径
f_path = filedialog.askopenfilename()


# %% 图像处理
# 读取图像
image_path = f_path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 原图备份
# 获取原有点的中心坐标
def getCenterPoints(image):
    # 限定下方划定轮廓的图像大小
    point_area = image.shape[0] * image.shape[1] / 100
    print(image.size)
    # 二值化处理（根据需要调整阈值）
    # 深色改黑，浅色改白，越大越浅
    threshold_value = 155
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # 寻找轮廓(这里容易有轮廓的bug)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 使用另一张图寻找轮廓和图像
    global marked_image

    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 存储轮廓中心点
    center_points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < point_area:  # 根据黑点的大小调整阈值
            M = cv2.moments(contour)

            if M['m00'] != 0:
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])

                center = (center_x, center_y)
                center_points.append(center)

                # 绘制轮廓
                cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 1)
                # 在轮廓上标记中心点
                cv2.circle(marked_image, center, 1, (0, 0, 255), -1)
    return center_points


center_points = getCenterPoints(image)


# 将原有点的中心坐标转换为字典格式，并以大写字母命名
def turntoCenterDicts(center_points):
    center_dict = {}  # 创建一个空字典，用于存储点的坐标信息
    # 遍历 center_points 列表并添加到字典，并将这些点设置为terminal
    for i, center in enumerate(center_points):
        point_name = chr(65 + i)  # 使用字母 A、B、C、... 来命名点，可以根据需要修改
        center_dict[point_name] = center
    return center_dict


center_dict = turntoCenterDicts(center_points)


# 给各个点一个索引名称
def getTerminals(center_points):
    terminals = []
    for i, center in enumerate(center_points):
        point_name = chr(65 + i)  # 使用字母 A、B、C、... 来命名点，可以根据需要修改
        terminals.append(point_name)
    return terminals


terminals = getTerminals(center_points)


# %% 算法部分，撒（<n-2）斯坦那点分别求最小生成树，使用遗传算法优化出最佳点位置，并比较不同增加点数的长度

# 获取图像坐标上下限，xy同步统计，奇数位为x，偶数位为y
# 获取图像上限
def lbGenerator(point_num):
    lb = []
    for i in range(point_num * 2):
        lb.append(1)
    return lb


# 获取图像下限
def ubGenerator(point_num, image):
    ub = []
    # 奇数位为x，偶数位为y
    for i in range(point_num * 2):
        x = (i + 1) % 2
        ub.append(image.shape[x])
    return ub


# 将生成的斯坦纳点x和y分离成列表
def pointsDeliver(xy):
    Steiner_points = []
    for i in range(0, len(xy), 2):
        Steiner_point_x = xy[i]
        Steiner_point_y = xy[i + 1]
        Steiner_point = (Steiner_point_x, Steiner_point_y)
        Steiner_points.append(Steiner_point)
    return Steiner_points


# 将生成的斯坦纳点列表转化为字典
def turntoSteinerDicts(Steiner_points):
    Steiner_dict = {}  # 创建一个空字典，用于存储点的坐标信息
    for i, Steiner_point in enumerate(Steiner_points):
        point_name = chr(97 + i)  # 使用字母 A、B、C、... 来命名点，可以根据需要修改
        Steiner_dict[point_name] = Steiner_point
    return Steiner_dict


# 获取点的最小生成树
def minimum_spanning_tree(points):
    # 构建一个带权重的图
    G = nx.Graph()
    for node1, (x1, y1) in points.items():
        for node2, (x2, y2) in points.items():
            if node1 != node2:
                # 计算欧几里得距离作为权重
                weight = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                G.add_edge(node1, node2, weight=weight)

    # 使用最小生成树算法计算最小生成树
    minimum_spanning_tree = nx.minimum_spanning_tree(G)

    # 打印最小生成树的边
    # print("最小生成树的边:", minimum_spanning_tree.edges())

    # print(1)

    # # 绘制图和最小生成树
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, font_weight='bold')
    # nx.draw_networkx_edges(minimum_spanning_tree, pos, edge_color='r', style='dashed')
    # plt.show()

    return minimum_spanning_tree


# 增加点，生成最小生成树
def minimumTree(Steiner_dict):
    points = {**center_dict, **Steiner_dict}
    miniTree = minimum_spanning_tree(points)
    return miniTree


# 带上格式转换
def tree(Steiner_points):
    Steiner_dict = turntoSteinerDicts(Steiner_points)
    tree = minimumTree(Steiner_dict)
    return tree


# 计算树长度
def treeLength(tree):
    # 初始化总长度为0
    total_length = 0
    # 遍历树中的每条边
    for u, v, data in tree.edges(data=True):
        # 获取边的权重（即欧几里得距离）
        weight = data['weight']
        # 累加权重到总长度
        total_length += weight
    return total_length


# 用于优化算法的树长计算函数
def treeFunc(xy):
    # 从sko的输入格式，转换成自己的格式
    xy = xy.astype(int)
    xy = xy.tolist()
    Steiner_points = pointsDeliver(xy)

    steiner_tree = tree(Steiner_points)
    minimumLength = treeLength(steiner_tree)

    # 转个sko看的格式
    minimumLength = np.array(minimumLength, dtype=float)
    return minimumLength


# %% 主函数部分

# 原图点数
num = len(terminals)

# 设定一个巨大的最小长度
miniLength = sys.maxsize

# 存当前最优各点位置
miniXY = []

for i in range(1, num - 2 + 1):
    # 增加点数
    n = i
    print(n)

    # 获取上下限
    lb = lbGenerator(n)
    ub = ubGenerator(n, image)
    # 设定size_pop
    size_pop=num*100
    # 设定最大迭代次数
    # max_iter=int(image.size/size_pop/20)
    max_iter=500
    # 遗传算法进行优化，可以更改迭代次数和初始种群数，以避免陷入局部最优。实测GA比PSO不容易陷入局部最优
    ga = GA(func=treeFunc, n_dim=n * 2, size_pop=size_pop, max_iter=max_iter, prob_mut=0.001, lb=lb, ub=ub, precision=1e-7)
    best_x, best_y = ga.run()

    # print('ga_best_x', best_x, 'ga_best_y', best_y)
    # 将best_x转化回x和y


    # # 绘制迭代图
    # Y_history = pd.DataFrame(ga.all_history_Y)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    # Y_history.min(axis=1).cummin().plot(kind='line')
    # plt.show()

    # 将记录的xy坐标转换回我们设定的格式
    x_y = best_x
    x_y = x_y.astype(int)
    x_y = x_y.tolist()
    x_y = pointsDeliver(x_y)

    # 挑选最优版本进行存储
    if best_y < miniLength:
        miniLength = best_y
        miniXY = x_y



#%% 画图部分
# 把最优点重新生成那个最短的斯坦纳树
steiner_tree=tree(miniXY)
# 重新得到最优点
Steiner_dict = turntoSteinerDicts(miniXY)
points = {**center_dict, **Steiner_dict}
print(points)
print(miniLength)

# 遍历图的每条边，并在图像上绘制
for edge in steiner_tree.edges():
    cv2.line(image, points[edge[0]], points[edge[1]], (0, 0, 255), 5)


# 调整窗口尺寸以适应图像
cv2.namedWindow('Steiner Tree', cv2.WINDOW_NORMAL)
# 显示带标记的图像
cv2.imshow('Steiner Tree', image)
# cv2.waitKey(0)
k = cv2.waitKey(0) & 0xFF  # 64位机器
cv2.imwrite('STEINER_tree.png', image)
cv2.destroyAllWindows()






