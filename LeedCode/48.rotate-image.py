#
# @lc app=leetcode id=48 lang=python
#
# [48] Rotate Image
#
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        matrix[:] = zip(*matrix[::-1])
        '''
        a = [1,2,3]
        b = [4,5,6]
        c = zip(a,b)  --> c = [(1,4),(2,5),(3,6)]
        d = zip(*c) --> [(1,2,3),(4,5,6)]

        '''
        

