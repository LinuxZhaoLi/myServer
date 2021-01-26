#！/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy

world_alcohol = numpy.genfromtxt("world_alcohol.txt",delimiter= ",",dtype=str)    # delimiter分隔符
print(type(world_alcohol))   # numpy.ndarray  核心 <class 'numpy.ndarray'>
# print(world_alcohol)
# print(help(numpy.genfromtxt))  # 查看帮助文档
third_country = world_alcohol[1,4]  # 取值
print(third_country)
from io import StringIO
import numpy as np
s = StringIO(u"1,1.3,abcde")
print(type(s))  # <class '_io.StringIO'>
data = np.genfromtxt(s, dtype=[('myint', 'i8'), ('myfloat', 'f8'),('mystring', 'S5')], delimiter=",")
vector = numpy.array([1,2,3,4])  # 创建一维数组
print(vector.shape) # (4,)
vector1 = numpy.array([[1,2,3],[1,2,3]]) # (2, 3)
print(vector1.shape)
print(vector1.dtype)    # int32
numbers = numpy.array([[1,2,3],[1,2,3.5]])
print(numbers.dtype)    #float64   传入数据类型保持一致

matrix = numpy.array([[1,2,3,4],[10,20,30,40],[100,200,300,400]])
print(matrix[:,1])      # 取第二列  所有行的第二列

vector2 = numpy.array([1,2,3,4])
print(vector2 == 3)     #[False False  True False]

equal_to_third = (vector2 == 3)     # 获取等于3 的数据
print(vector2[equal_to_third])


matrix = numpy.array([[5,10,15],[20,25,30],[35,40,45]])
second_colum_25 = (matrix[:,1]) == 25  #  获取包含25的列
print(matrix[second_colum_25,:])

vector = numpy.array([1,2,3,4])
a = numpy.arange(15).reshape(3,5)
print(a.ndim)
a = numpy.ones((2,3,4),dtype=np.int32)
print(a)
