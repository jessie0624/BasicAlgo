def MergeSort(data):
    def merge(data, left, right, mid):
        lp = left
        rp = mid + 1
        tmp = []
        while lp <= mid and rp <= right:
            if data[lp] <= data[rp]:
                tmp.append(data[lp])
                lp += 1
            elif data[lp] > data[rp]:
                tmp.append(data[rp])
                rp += 1
        while lp <= mid:
            tmp.append(data[lp])
            lp += 1
        while rp <= right:
            tmp.append(data[rp])
            rp += 1
        data[left : right + 1] = tmp

    def sort(data, left, right):
        if left == right:
            return
        elif left < right:
            mid = left + (right - left)//2
            sort(data, left, mid)
            sort(data, mid + 1, right)
            merge(data, left, right, mid)
    
    if not data or len(data) < 2:
        return
    sort(data, 0, len(data) - 1)
        