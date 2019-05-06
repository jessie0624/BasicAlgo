def InsertSort(data):
    if not data or len(data) < 2:
        return
    for i in range(1, len(data)):
        for j in range(0, i-1):
            if data[j + 1] < data[j]:
                data[j + 1], data[j] = data[j], data[j + 1]