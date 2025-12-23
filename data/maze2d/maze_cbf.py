from typing import List
from create_corridor import Rectangle
import numpy as np
import math

class MazeCBF:
    def __init__(self, 
                 maze, 
                 rect_list: List[Rectangle],
                 alpha: float = 0.5):
        """
        Docstring for __init__
        
        :param self: Description
        :param maze: 迷宫环境
        :param rect_list: 
        :param alpha: 缩放系数 [0, 1]
        """
        assert alpha >= 0 and alpha <= 1

        self.alpha = alpha
        self.rect_list = rect_list
        self.maze = maze

        # CBF form: (x-x_c)^2/a^2 + (y-y_c)^2/b^2 -1 >= 0
        self.ellips_list = self.create_ellips_list()

    def create_ellips_list(self):

        ellips_list = []
        for rect in self.rect_list:
            # 左上角格子中心
            p_min = self.maze.cell_rowcol_to_xy(np.array([rect.r_min, rect.c_min]))
            # 右下角格子中心
            p_max = self.maze.cell_rowcol_to_xy(np.array([rect.r_max, rect.c_max]))

            half_scale = self.maze.maze_size_scaling * 0.5
            x_min, x_max = p_min[0] - half_scale, p_max[0] + half_scale
            y_min, y_max = p_max[1] - half_scale, p_min[1] + half_scale

            x_center = (x_min + x_max) * 0.5
            y_center = (y_min + y_max) * 0.5

            x_length = x_max - x_min
            y_length = y_max - y_min

            # 最大内切椭圆
            a_in = x_length * 0.5
            b_in = y_length * 0.5
            # 最小外接椭圆
            a_out = x_length / math.sqrt(2)
            b_out = y_length / math.sqrt(2)
            # 使用参数alpha控制近似程度
            a = a_in + (a_out - a_in) * self.alpha
            b = b_in + (b_out - b_in) * self.alpha

            ellips_list.append([x_center, y_center, a, b])

        return ellips_list


    
        
        
    
