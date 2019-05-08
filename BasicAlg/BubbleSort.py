def BubbleSort(data):
    if not data or len(data) < 2:
        return
    for i in range(len(data) - 1, 0, -1):
        for j in range(i):
            if data[j+1] < data[j]:
                data[j+1], data[j] = data[j], data[j+1]

def BubbleSort_Flag(data):
    if not data or len(data) < 2:
        return
    Flag = True
    for i in range(len(data) - 1, 0, -1):
        for j in range(i):
            if data[j+1] < data[j]:
                data[j+1], data[j] = data[j], data[j+1]
                Flag = False
        if Flag:
            break

