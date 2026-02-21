# 第二个文档
## Framework for IEEE course final project
## Fan Cheng, 2022

import random
import copy
import math


class Matrix:
    r"""
    自定义的二维矩阵类

    Args:
        data: 一个二维的嵌套列表，表示矩阵的数据。即 data[i][j] 表示矩阵第 i+1 行第 j+1 列处的元素。
              当参数 data 不为 None 时，应根据参数 data 确定矩阵的形状。默认值: None
        dim: 一个元组 (n, m) 表示矩阵是 n 行 m 列, 当参数 data 为 None 时，根据该参数确定矩阵的形状；
             当参数 data 不为 None 时，忽略该参数。如果 data 和 dim 同时为 None, 应抛出异常。默认值: None
        init_value: 当提供的 data 参数为 None 时，使用该 init_value 初始化一个 n 行 m 列的矩阵，
                    即矩阵各元素均为 init_value. 当参数 data 不为 None 时，忽略该参数。 默认值: 0

    Attributes:
        dim: 一个元组 (n, m) 表示矩阵的形状
        data: 一个二维的嵌套列表，表示矩阵的数据"""

    def __init__(self, data=None, dim=None, init_value=0):  # 1
        self.data = data
        self.dim = dim
        self.init_value = init_value
        if self.dim == None and self.data == None:
            return None
        elif self.data == None:
            self.data = [[self.init_value for _ in range(dim[1])] for _ in range(dim[0])]
        elif self.dim == None:
            if type(self.data[0]) != int:  # 防止生成一维矩阵时报错
                self.dim = (len(self.data), len(self.data[0]))
            else:
                self.dim = (len(self.data), self.data[0])
        elif self.dim != (len(self.data), len(self.data[0])):
            if type(self.data[0]) != int:
                self.dim = (len(self.data), len(self.data[0]))
            else:
                self.dim = (len(self.data), self.data[0])

    def shape(self):  # 2
        if self.data != None:
            return self.dim[0], self.dim[1]
        elif self.dim != None:
            return self.dim
        else:
            return "Error!!!"

    def reshape(self, newdim):  # 3
        lst1 = []
        lst2 = [[0 for _ in range(newdim[1])] for _ in range(newdim[0])]

        if (newdim[0] * newdim[1] != self.dim[0] * self.dim[1]):
            return "Error!!!"
        else:
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    lst1.append(self.data[i][j])

            k = 0
            for k in range(newdim[0] * newdim[1]):
                lst2[k // newdim[1]][k % newdim[1]] = lst1[k]

            return Matrix(lst2)

    def dot(self, other):  # 4
        if len(self.data[0]) != len(other.data):
            return "Error!!!"
        else:
            result_data = [[0 for _ in range(len(other.data[0]))] for _ in range(len(self.data))]
            for i in range(len(self.data)):
                for j in range(len(other.data[0])):
                    for k in range(len(other.data)):
                        result_data[i][j] += self.data[i][k] * other.data[k][j]
            return Matrix(data=result_data)

    def T(self):  # 5
        a = [[0 for _ in range(self.dim[0])] for _ in range(self.dim[1])]
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                a[j][i] = self.data[i][j]
        self.data = a
        self.dim = (self.dim[1], self.dim[0])

    def sum(self, axis=None):  # 6
        if axis == 0:
            sum1 = [[0 for _ in range(self.dim[1])]]
            for i in range(self.dim[1]):
                for j in range(self.dim[0]):
                    sum1[0][i] += self.data[j][i]
            return Matrix(sum1)

        elif axis == 1:
            self.T()
            max1 = self.sum(0)
            self.T()
            max1.T()
            return max1

        else:
            max2 = self.sum(1)
            max3 = max2.sum(0)
            return max3

    def copy(self):  # 7 要求：输入一个Matrix，返回一个Matrix
        a = copy.deepcopy(self.data)
        return Matrix(a)

    # 已知上面这个是可行的，但是我们是否可以不用deepcopy就完成呢（因为题干里没有给import copy）
    # def copy(self):  # 7
    #     lst = self.data
    #     lst1 = [[0 for _ in range(self.dim[1])] for _ in range(self.dim[0])]
    #     for i in range(len(lst)):
    #         for j in range(len(lst[0])):
    #             lst1[i][j] = lst[i][j]
    #     return Matrix(lst1)
    # （↑这个也是可行的↑）

    def Kronecker_product(self, other):  # 8
        a = Matrix(None, (self.dim[0] * other.dim[0], self.dim[1] * other.dim[1]), 0)
        for i in range(a.dim[0]):
            for j in range(a.dim[1]):
                b = i // other.dim[0]
                c = j // other.dim[1]
                d = i % other.dim[0]
                e = j % other.dim[1]
                a.data[i][j] = self.data[b][c] * other.data[d][e]
        return a

    def __getitem__(self, key):  # 9
        if type(key) == tuple and len(key) == 2 \
                and type(key[0]) == int and type(key[1]) == int:
            return self.data[key[0]][key[1]]
        else:
            x1, x2 = key
            sa1, so1 = x1.start, x1.stop
            if sa1 == None:
                sa1 = 0
            if so1 == None:
                so1 = self.dim[0]
            x1 = slice(sa1, so1)

            sa2, so2 = x2.start, x2.stop
            if sa2 == None:
                sa2 = 0
            if so2 == None:
                so2 = self.dim[1]
            x2 = slice(sa2, so2)

            return Matrix([[self.data[i][j] for j in range(x2.start, x2.stop)] \
                           for i in range(x1.start, x1.stop)])

    def gauss(self):  # 自编·高斯函数
        a = self.copy().data
        i, p = 0, 0

        while i < min(self.dim[0], self.dim[1]):
            while p < len(a[0]) and a[i][p] == 0:
                for j in range(i + 1, self.dim[0]):
                    if a[j][p] != 0:
                        a[i], a[j] = a[j], a[i]
                        break
                if a[i][p] == 0:
                    p += 1

            for k in range(self.dim[0]):
                if k != i and p < len(a[0]):
                    d = a[k][p]
                    for j in range(len(a[0])):
                        a[k][j] -= a[i][j] * d / a[i][p]  # 这里加了一个/ a[i][p]

            i += 1
            p += 1

        return Matrix(a)

    def __setitem__(self, key, value):  # 10
        if isinstance(key[0], int) and isinstance(key[1], int):
            i, j = key
            self.data[i][j] = value
        elif isinstance(key[0], slice) and isinstance(key[1], slice):
            start_row, stop_row, step_row = key[0].indices(len(self.data))
            start_col, stop_col, step_col = key[1].indices(len(self.data[0]))
            for i in range(start_row, stop_row):
                for j in range(start_col, stop_col):
                    self.data[i][j] = value.data[i - start_row][j - start_col]

    def __pow__(self, n):  # 11
        if self.dim[0] != self.dim[1]:
            return "Error!!!"
        if n == 0:
            return Matrix.I(self.dim[0])
        i = n - 1
        b = self.copy()
        while i > 0:
            b = Matrix.dot(self, b)
            i -= 1
        return b  # 要返回运算结果？？？

    def __add__(self, other):  # 12
        if self.dim[0] != other.dim[0] and self.dim[1] != other.dim[1]:
            return "Error!!!"
        sum = [[0 for _ in range(self.dim[1])] for _ in range(self.dim[0])]
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                x0 = self.data[i][j] + other.data[i][j]
                sum[i][j] = x0
        return Matrix(sum)

    def __sub__(self, other):  # 13
        other1 = Matrix(None, (self.dim[0], self.dim[1]))

        other1.data = [[-x for x in other.data[i]] for i in range(other.dim[0])]
        return self + other1

    def __mul__(self, other):  # 14
        if len(self.data) != len(other.data) or len(self.data[0]) != len(other.data[0]):
            return "Error!!!"
        else:
            result_data = [[0 for _ in range(len(self.data[0]))] for _ in range(len(self.data))]
            for i in range(len(self.data)):
                for j in range(len(self.data[0])):
                    result_data[i][j] = self.data[i][j] * other.data[i][j]
            return Matrix(data=result_data)

    def __len__(self):  # 15
        return self.dim[0] * self.dim[1]

    def __str__(self):  # 16
        if self.data == None and self.dim == None:
            return "Error!!!"
        else:
            result = "["
            first_row = True
            for row in self.data:
                if not first_row:
                    result += " \n "
                else:
                    first_row = False
                result += "["

                for item in range(len(row)):
                    if item < len(row) - 1:
                        result += str(row[item]) + " "
                    else:
                        result += str(row[item])
                result += "]"
            result += "]"
            return result

    def det(self):  # 17 # 不直接使用高斯函数的版本
        if len(self.data) != len(self.data[0]):
            return "Error!!!"
        else:
            if len(self.data) == 1:
                return self.data[0][0]
            else:
                result = 0
                for i in range(len(self.data)):
                    sub_matrix_data = [row[:i] + row[i + 1:] for row in self.data[1:]]
                    sub_matrix = Matrix(data=sub_matrix_data)
                    result += ((-1) ** i) * self.data[0][i] * sub_matrix.det()
                return result

    # def det(self): # 也可以直接调用gauss()来求行列式！
    #     if self.dim[0] != self.dim[1]:
    #         return "Error!"
    #     else:
    #         a = self.gauss().data
    #         pro = 1
    #         for i in range(len(a)):
    #             pro *= a[i][i]
    #         return pro

    def inverse(self):  # 18 # 不直接使用高斯函数的版本
        if self.dim[0] != self.dim[1]:
            return "Error!!!"
        else:
            c = self.copy()
            a = c.data
            b = [0] * self.dim[0]
            for i in range(self.dim[0]):
                b[i] = 1
                a[i].extend(b)
                b[i] = 0
            for i in range(self.dim[0]):
                if a[i][i] == 0:
                    for j in range(i, self.dim[0]):
                        if a[j][i] != 0:
                            a[i], a[j] = a[j], a[i]
                    else:
                        return "Error!!!"
                elif a[i][i] != 1:
                    d = a[i][i]
                    for j in range(2 * self.dim[0]):
                        a[i][j] /= d

                for k in range(self.dim[0]):
                    if k != i:
                        d = a[k][i]
                        for j in range(2 * self.dim[0]):
                            a[k][j] -= a[i][j] * d
        A = Matrix(a)
        return A[:, self.dim[0]:]

    # def inverse(self): # 也可以直接调用gauss()来求解逆矩阵！
    #     if self.dim[0] != self.dim[1]:
    #         return "Error!!!"
    #     elif self.det() == 0:
    #         return "Error!!!"
    #     else:
    #         zr_lst = [[0 for _ in range(self.dim[1])] for _ in range(self.dim[0])]
    #         for i in range(len(zr_lst)):
    #             zr_lst[i][i] = 1
    #         sum = Matrix.concatenate([self,Matrix(zr_lst)],0)
    #         sum1 = sum.gauss()
    #         for i in range(sum1.dim[0]):
    #             for j in range(sum1.dim[1]-1,-1,-1):
    #                 sum1.data[i][j] /= sum1.data[i][i]
    #         return sum1[:,self.dim[1]:]

    def rank(self):  # 19
        lst = self.gauss().data
        zero_row = 0
        flag = True
        for i in range(len(lst) - 1, -1, -1):
            for j in range(len(lst[0])):
                if not math.isclose(lst[i][j], 0):
                    flag = False
                    break
            if not flag:
                return len(lst) - zero_row
            else:
                zero_row += 1

        return len(lst) - zero_row

    def I(n):  # 20
        result = Matrix(None, (n, n))
        for i in range(n):
            result.data[i][i] = 1
        return result

    def narray(dim, init_value=1):  # dim (,,,,,), init为矩阵元素初始值21
        b = init_value  # 21
        lst = list(dim)
        lst.reverse()
        for i in lst:
            a = [b for _ in range(i)]
            b = a
        return Matrix(b, None, init_value)

    def arange(sa, so, se):  # 22
        lst = [[]]
        for i in range(sa, so, se):
            lst[0].append(i)
        return Matrix(lst)

    def zeros(dim):  # 23
        return Matrix.narray(dim, 0)

    def zeros_like(matrix):  # 24
        a = matrix.data
        lst = [len(a)]
        while type(a[0]) == list:
            a = a[0]
            lst.append(len(a))
        dim = tuple(lst)
        return Matrix.narray(dim, 0)

    def ones(dim):  # 25
        return Matrix.narray(dim, 1)

    def ones_like(matrix):  # 26
        a = matrix.data
        lst = [len(a)]
        while type(a[0]) == list:
            a = a[0]
            lst.append(len(a))
        dim = tuple(lst)
        return Matrix.narray(dim, 1)

    def nrandom(dim):  # 27
        if len(dim) == 1:
            return random.randint(1, 100)
        elif len(dim) == 2:
            new = [[random.randint(1, 100) for _ in range(dim[1])] for _ in range(dim[0])]
        else:
            new = [[Matrix.nrandom(dim[1:]).data] for _ in range(dim[0])]
        return Matrix(new)

    def nrandom_like(matrix):  # 28
        a = matrix.data
        lst = [len(a)]
        while type(a[0]) == list:
            a = a[0]
            lst.append(len(a))
        dim = tuple(lst)
        return Matrix.nrandom(dim)

    def concatenate(matrices, axis=0):  # 29
        if not matrices:
            return None
        if axis == 1:
            col_num = matrices[0].dim[1]
            for matrix in matrices[1:]:
                if matrix.dim[1] != col_num:
                    return "Error!!!"
            new_data = []
            for matrix in matrices:
                new_data.extend(matrix.data)
            return Matrix(data=new_data)
        elif axis == 0:
            row_num = matrices[0].dim[0]
            for matrix in matrices[1:]:
                if matrix.dim[0] != row_num:
                    return "Error!!!"
            new_data = [[] for _ in range(row_num)]
            for matrix in matrices:
                for i in range(row_num):
                    new_data[i].extend(matrix.data[i])
            return Matrix(data=new_data)
        else:
            return "Error!!!"

    def vectorize(self, func):  # 30
        a = Matrix.copy(self)
        b = a.data
        for i in range(a.dim[0]):
            for j in range(a.dim[1]):
                b[i][j] = func(b[i][j])
        return Matrix(b)


if __name__ == "__main__":
    print("test here")
    pass