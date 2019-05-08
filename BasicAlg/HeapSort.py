def HeapSort(data):
    ##将数组构造成大根堆
    def heapInsert(data, index):
        
        while (data[index] > data[(index - 1)//2 if index - 1 > 0 else 0]):
            data[index], data[(index - 1)//2] = data[(index-1)//2], data[index]
            index = (index - 1)//2 if index - 1 > 0 else 0 
     
     ##某个位置变化后做调整
    def heapFy(data, index):
        left = index * 2 + 1
        while (left < len(data)):
            if left + 1 < len(data):
                largest = left if data[left] > data[left + 1] else left + 1
            else:
                largest = left
            largest = index if data[index] > data[largest] else largest
            if index == largest:
                break
            data[index], data[largest] = data[largest], data[index]
            index = largest
            left = index * 2 + 1
        return data 

    ##排序
    if not data or len(data) < 2:
        return
    for i in range(len(data)):
        heapInsert(data, i)
    
    j = len(data) - 1
    while j >= 0:
        data[j], data[0] = data[0], data[j]
        data[0:j] = heapFy(data[0:j],0)  ##key point: modify unsorted data in range(0,j)
        j -= 1
        
     


        
        