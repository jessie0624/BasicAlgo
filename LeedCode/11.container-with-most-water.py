#
# @lc app=leetcode id=11 lang=python
#
# [11] Container With Most Water
#
# https://leetcode.com/problems/container-with-most-water/description/
#
# algorithms
# Medium (44.83%)
# Likes:    3339
# Dislikes: 444
# Total Accepted:    381.3K
# Total Submissions: 850K
# Testcase Example:  '[1,8,6,2,5,4,8,3,7]'
#
# Given n non-negative integers a1, a2, ..., an , where each represents a point
# at coordinate (i, ai). n vertical lines are drawn such that the two endpoints
# of line i is at (i, ai) and (i, 0). Find two lines, which together with
# x-axis forms a container, such that the container contains the most water.
# 
# Note: You may not slant the container and n is at least 2.
# 
# 
# 
# 
# 
# The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In
# this case, the max area of water (blue section) the container can contain is
# 49. 
# 
# 
# 
# Example:
# 
# 
# Input: [1,8,6,2,5,4,8,3,7]
# Output: 49
# 
#
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        # maxCon = 0
        # maxI = 0
        # maxJ = 0

        # for i in range(len(height) - 1):
        #     for j in range(i,len(height)):
        #         curCon =  (j-i) * min(height[i], height[j])
        #         if curCon > maxCon:
        #             maxCon = curCon
        #             maxI = i
        #             maxJ = j
        # return maxCon

        # maxCon = 0
        if len(height) < 2:
            return 0
        maxI = 0
        maxJ = len(height) - 1
        maxH = min(height[maxJ], height[maxI])
        maxCon = maxH * (maxJ - maxI)
        while maxI < maxJ:
            if height[maxI] >= height[maxJ]:
                maxJ -= 1
            else:
                maxI += 1
            curH = min(height[maxI], height[maxJ])
            curCon = curH * (maxJ - maxI)
            if curCon > maxCon:
                maxCon = curCon
        return maxCon

        

