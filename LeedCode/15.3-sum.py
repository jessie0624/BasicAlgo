#
# @lc app=leetcode id=15 lang=python
#
# [15] 3Sum
#
# https://leetcode.com/problems/3sum/description/
#
# algorithms
# Medium (24.09%)
# Likes:    3871
# Dislikes: 433
# Total Accepted:    564.4K
# Total Submissions: 2.3M
# Testcase Example:  '[-1,0,1,2,-1,-4]'
#
# Given an array nums of n integers, are there elements a, b, c in nums such
# that a + b + c = 0? Find all unique triplets in the array which gives the sum
# of zero.
# 
# Note:
# 
# The solution set must not contain duplicate triplets.
# 
# Example:
# 
# 
# Given array nums = [-1, 0, 1, 2, -1, -4],
# 
# A solution set is:
# [
# ⁠ [-1, 0, 1],
# ⁠ [-1, -1, 2]
# ]
# 
# 
#
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        if len(nums) < 3:
            return res
            
        nums.sort()

        if nums[0] > 0 or nums[len(nums) - 1] < 0:
            return res
        if nums[0] == 0 and nums[len(nums) - 1] == 0 and len(nums)>=3:
            res.append([0,0,0])
            return res
        for i in range(len(nums) - 2):  ##j k
            if nums[i] > 0:
                break
            if  nums[i] == nums[i -1]:
                continue
            l, r = i + 1, len(nums) -1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l+1]:
                        l += 1

                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1; r -= 1
            
        return res
            
