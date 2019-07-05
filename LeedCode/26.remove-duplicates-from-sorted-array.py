#
# @lc app=leetcode id=26 lang=python
#
# [26] Remove Duplicates from Sorted Array
#
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cur_index = 0
        while cur_index < len(nums) - 1:
            if nums[cur_index] == nums[cur_index + 1]:
                del nums[cur_index]
                cur_index -= 1
            cur_index += 1
        return len(nums)
            

