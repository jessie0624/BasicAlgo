#
# @lc app=leetcode id=27 lang=python
#
# [27] Remove Element
#
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """

        cur_index = len(nums) - 1
        
        while cur_index >= 0:
            if nums[cur_index] == val:
                del nums[cur_index]
            cur_index -= 1
        return len(nums)
        

