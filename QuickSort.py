import random
def QuickSort(data):
    def partition(data, left, right):
        less = left-1
        more = right
        while left < more:
            if data[left] < data[right]:
                data[left], data[less+1] = data[less+1], data[left]
                left += 1
                less += 1
            elif data[left] > data[right]:
                data[left], data[more-1] = data[more-1], data[left]
                more -=1
            else:
                left += 1
        data[more], data[right] = data[right], data[more]
        return (less + 1, more - 1) ## return equal range 

    def sort(data, left, right):
        if left < right:
            ran = random.randint(left, right)
            data[right], data[ran] = data[ran], data[right]
            p = partition(data, left, right)
            sort(data, left, p[0] - 1)
            sort(data, p[1] + 1, right)

    if not data or len(data)<2:
        return
    sort(data, 0, len(data)-1)
