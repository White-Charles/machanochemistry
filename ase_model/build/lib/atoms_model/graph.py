
def point_position(line_point1, line_point2, point3):
    # 计算
    # print(line_point1, line_point2, point3)
    if line_point1[0] == line_point2[0]:
        # 处理直线垂直于x轴的情况
        return point3[0] != line_point1[0]
    else:
        # 计算直线的斜率
        slope = (line_point2[1] - line_point1[1]) / (line_point2[0] - line_point1[0])

        # 计算第三个点的纵坐标
        y3 = line_point1[1] + slope * (point3[0] - line_point1[0])

        # return point3[1] > y3
        return point3[1] > y3+1E-3 # 如果有问题需要考虑是否需要更换
