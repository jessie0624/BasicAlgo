#
# @lc app=leetcode id=66 lang=python3
#
# [66] Plus One
#
# https://leetcode.com/problems/plus-one/description/
#
# algorithms
# Easy (41.31%)
# Likes:    874
# Dislikes: 1554
# Total Accepted:    398.1K
# Total Submissions: 963.6K
# Testcase Example:  '[1,2,3]'
#
# Given a non-empty array of digits representing a non-negative integer, plus
# one to the integer.
# 
# The digits are stored such that the most significant digit is at the head of
# the list, and each element in the array contain a single digit.
# 
# You may assume the integer does not contain any leading zero, except the
# number 0 itself.
# 
# Example 1:
# 
# 
# Input: [1,2,3]
# Output: [1,2,4]
# Explanation: The array represents the integer 123.
# 
# 
# Example 2:
# 
# 
# Input: [4,3,2,1]
# Output: [4,3,2,2]
# Explanation: The array represents the integer 4321.
# 
#
import math
class Solution:
    def plusOne(self, digits):
        # case 1 --36ms
        # l = len(digits)  #36ms
        # number = 0
        # for i in range(0,l):
        #     number += digits[i] * math.pow(10, (l - i - 1))
        # number += 1
        # return [int(i) for i in str(number)]
        
        # case 2: 44ms
        # l = len(digits)
        # digits[l - 1] += 1
        # while digits[l - 1] == 0:
        #     l = l - 1
        #     digits[l - 1] += 1
        # return digits
        
        #case 3: 36ms   tt86.89% m21.36%
        for i in list(range(len(digits) - 1, -1, -1)):
            if digits[i] == 9:
                digits[i] = 0
                if i == 0:
                    digits.insert(0,1)
            else:
                digits[i] = digits[i] + 1
                break
        return digits

        # ### 28ms
        # num = 0
        # for i in range(len(digits)):
    	#     num += digits[i] * pow(10, (len(digits)-1-i))
        # return [int(i) for i in str(num+1)]
            

        
        


