import unittest
from QuickSort import *
from BubbleSort import *
from MergeSort import *
from SelectSort import *
from InsertSort import *

class TestQuickSort(unittest.TestCase):
    def test_quicksort(self):
        data1 = [10,9,-1,0,4,7,8]
        ret = data1
        QuickSort(data1)
        sorted(ret)
        self.assertEqual(ret, data1)
    def test_bubblesort(self):
        data2 = [10,9,-1,0,4,7,8]
        ret = data2
        BubbleSort(data2)
        sorted(ret)
        self.assertEqual(ret, data2)
    
    def test_bubblesort_flag(self):
        data3 = [10,9,-1,0,4,7,8]
        ret = data3
        BubbleSort_Flag(data3)
        sorted(ret)
        self.assertEqual(ret, data3)

    def test_mergesort(self):
        data4 = [10, 9, -1, 0, 4, 7, 8]
        ret = data4
        MergeSort(data4)
        sorted(ret)
        self.assertEqual(ret, data4)

    def test_selectsort(self):
        data5 = [10, 9, -1, 0, 4, 7, 8]
        ret = data5
        SelectSort(data5)
        sorted(ret)
        self.assertEqual(ret, data5)
    
    def test_insertsort(self):
        data6 = [10, 9, -1, 0, 4, 7, 8]
        ret = data6
        InsertSort(data6)
        sorted(ret)
        self.assertEqual(ret, data6)

if __name__ == "__main__":
    unittest.main()