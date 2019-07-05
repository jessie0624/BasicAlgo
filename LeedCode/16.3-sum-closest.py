#
# @lc app=leetcode id=16 lang=python
#
# [16] 3Sum Closest
#
# https://leetcode.com/problems/3sum-closest/description/
#
# algorithms
# Medium (45.77%)
# Likes:    1133
# Dislikes: 82
# Total Accepted:    352.3K
# Total Submissions: 769.8K
# Testcase Example:  '[-1,2,1,-4]\n1'
#
# Given an array nums of n integers and an integer target, find three integers
# in nums such that the sum is closest to target. Return the sum of the three
# integers. You may assume that each input would have exactly one solution.
# 
# Example:
# 
# 
# Given array nums = [-1, 2, 1, -4], and target = 1.
# 
# The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
# 
# 
#
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # nums = [i + target for i in nums]

        ## target find the minest(abs) -target *3
        if len(nums) <3:
            return

        nums.sort()
        res_sum = 0
        
        diff = float('inf')
        for i in range(len(nums)):
            l,r = i+1, len(nums) - 1
            while l < r:
                temp_sum = nums[i] + nums[l] + nums[r]
                new_diff = abs(temp_sum - target)
                if new_diff < diff:
                    diff = new_diff
                    res_sum = temp_sum
                
                if temp_sum > target:
                    r -= 1
                elif temp_sum < target:
                    l += 1
                else:
                    return temp_sum
        return res_sum

