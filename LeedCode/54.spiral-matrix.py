#
# @lc app=leetcode id=54 lang=python
#
# [54] Spiral Matrix
#
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        return matrix and list(matrix.pop(0)) + self.spiralOrder(zip(*matrix)[::-1])
        ### if matrix is empty
        '''
        如果 matrix 为空，返回之前的 list
        如果matrix 不为空，返回
       
        or 和 and 都是从左到右运算，
        or：返回第一个为真的值，都为假时返回后一个值
        and： 若所有值均为真，则返回后一个值，有一个假的值，则返回第一个假的值。

        a = [1,2,3] b=[4,5,6]
        c = [[1,2,3],[4,5,6]]
        zip(*c) = [(1,4), (2,5),(3,6)]
        zip(*c)[::-1] = [(3,6),(2,4),(1,4)]
        zip(*c[::-1]) = [(4,1),(5,2), (6,3)]
        '''
