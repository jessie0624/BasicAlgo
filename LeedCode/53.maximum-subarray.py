#
# @lc app=leetcode id=53 lang=python
#
# [53] Maximum Subarray
#
# import math
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ##1) O(n^2)
        # maxRet = float("-inf")
        # maxList = []
        # Len = len(nums)
        # for l in range(1, Len + 1): ###数组长度 1个元素->N个元素
        #     for index in range(0,Len-l+1): ###求和的起始index
        #         maxSum = 0
        #         subNum = nums[index:index+l]
        #         # print(subNum)
        #         for i in subNum:
        #             maxSum += i
        #         if maxSum > maxRet:
        #             maxRet = maxSum
        #             maxList = nums[index-l:index+l-1]
        # return maxRet

        ##2)O(n)
        maxRet = float("-inf")
        curSum = 0
        for i in range(len(nums)):
            curSum += nums[i]
            if curSum > maxRet:
                maxRet = curSum
            if curSum <  0 and i < len(nums) -1:
                curSum = 0
        return maxRet 




        

