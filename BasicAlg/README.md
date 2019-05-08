# Basic Algo 
## Sort functions
#### QickSort: 
    Partion and sort. 
        Partion return a tuple the left and right for middle part. so when recall sort function we should -1 for left part and  +1 for right part.
            return (less + 1, more - 1) ## return equal range 
            sort(data, left, p[0] - 1)
            sort(data, p[1] + 1, right)
    Time complexity: O(N * logN)
    Space complexity: O(N * logN)
    Stable: No

#### MergeSort:
    Merge and sort. 
        Merge function requires 'data, left, right, mid' parameters. Compare data in left-mid with data in mid+1-right and append to a temp space, then move back to data[left : right + 1]
        Sort function requires 'data, left, right' parameters. When left == right, no need sort/merge, return directly.Otherwise, we need got the mid parameter from left + (right - left) //2 ,and  sort left and right part, then merge left and right by merge function. 
            sort(data, left, mid), sort(data, mid + 1, right),  merge(data, left, right, mid)
    Time complexity: O(N * logN)
    Space complexity: O(N)
    Stable: Yes

#### BubbleSort:
    Time complexity: O(N ^2)
    Space complexity: O(1)
    Stable: Yes

#### HeapSort:

#### InsertSort:
    regards the previous M value are sorted. Then sort the next one(M+1).Compare the data (M+1) with O-M and insert to the right place.
    Time complexity: O(N ^ 2)  常数项特别低
    Space complexity: O(1)
    Stable: Yes

#### selectSort:
    Time complexity: O(N ^ 2)
    Space complexity: O(1)
    Stable: No







