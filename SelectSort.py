def SelectSort(data):
    if not data or len(data) < 2:
        return
    for i in range(len(data) - 1): ## 0- len(data)-2, 
        for j in range(i+1, len(data)): ##i+1 - len(data)-1
            if data[j] < data[i]:
                data[j], data[i] = data[i], data[j]
    