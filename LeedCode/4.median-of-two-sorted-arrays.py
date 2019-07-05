#
# @lc app=leetcode id=4 lang=python
#
# [4] Median of Two Sorted Arrays
#
# https://leetcode.com/problems/median-of-two-sorted-arrays/description/
#
# algorithms
# Hard (26.48%)
# Likes:    4395
# Dislikes: 602
# Total Accepted:    444.7K
# Total Submissions: 1.7M
# Testcase Example:  '[1,3]\n[2]'
#
# There are two sorted arrays nums1 and nums2 of size m and n respectively.
# 
# Find the median of the two sorted arrays. The overall run time complexity
# should be O(log (m+n)).
# 
# You may assume nums1 and nums2 cannot be both empty.
# 
# Example 1:
# 
# 
# nums1 = [1, 3]
# nums2 = [2]
# 
# The median is 2.0
# 
# 
# Example 2:
# 
# 
# nums1 = [1, 2]
# nums2 = [3, 4]
# 
# The median is (2 + 3)/2 = 2.5
# 
# 
#
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        l = len(nums1) + len(nums2)
        if l % 2:
            return self.kth(nums1, nums2, l//2 + 1)
        else:
            return float(self.kth(nums1, nums2, l//2 + 1) + self.kth(nums1,nums2, l//2)) /2
        
   
    def kth(self,A,B,k):
        la, lb = len(A), len(B)
        if la > lb: return self.kth(B,A,k)
        if la == 0: return B[k-1]
        if k == 1: return min(A[0], B[0])
        
        k1 = min(k//2, la)  ## split K into 2 part. one is for A, one is for B
        k2 = k-k1
        
        if A[k1-1] > B[k2-1]: return self.kth(A, B[k2:], k-k2)
        elif A[k1-1] < B[k2-1]: return self.kth(A[k1:],B,k-k1)
        else: return A[k1-1]


#     1）当array1[k/2-1] == array2[k/2-1] 则返回array1[k/2-1]或者array2[k/2-1]
#     2）当array1[k/2-1] > array2[k/2-1]  则array2在[0,k/2-1]范围内的元素一定比array1、array2合并后第k个元素小，可以不用考虑array2在[0,k/2-1]范围内的元素
#     3）当array1[k/2-1] < array2[k/2-1]  则array1在[0,k/2-1]范围内的元素一定比array1、array2合并后第k个元素小，可以不用考虑array1在[0,k/2-1]范围内的元素
#      因此算法可以写成一个递归的形式，递归结束的条件为：
#     1）array1[k/2-1] == array2[k/2-1] return array1[k/2-1]
#     2）array1或者array2为空时，return array1[k-1]或者array2[k-1]
#     3）k==1时，返回min(array1[0],array2[0])

