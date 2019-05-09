import math
def ShellsSort(data):
    h = math.floor((len(data) -1)/2)
    while h >= 1:
        for i in range(h, len(data)):
            j = i
            while j >=h and data[j]<data[j-h]:
                data[j], data[j-h] = data[j-h], data[j]
                j -= h
            h = math.floor(h//2)
  