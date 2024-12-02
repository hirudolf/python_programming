LIST_A = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LIST_B = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

LIST_C = []
count = 1
index_A = 0
index_B = 0

while len(LIST_C) < 20:
    if len(LIST_C) + count <= 20:
        LIST_C.extend(LIST_A[index_A:index_A + count])
        index_A += count
    if len(LIST_C) + count <= 20:
        LIST_C.extend(LIST_B[index_B:index_B + count])
        index_B += count
    count += 1

print(LIST_C)
