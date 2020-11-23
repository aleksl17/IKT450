
def find_highest(lst):
    highest = 0
    for i in range(len(lst)):
        if lst[i] > highest:
            highest = i
    return highest

def find_order(lst):
    retval = []
    for i in range(len(lst)):
        highest = find_highest(lst)
        lst[highest] = -1
        retval.append(highest)
    return retval

def compare(pred, y):
    ordered_1 = find_order(pred)
    ordered_2 = find_order(y)

    return ordered_1 == ordered_2


