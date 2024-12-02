print("별 찍기\n1.☆\n2.★\n3.#")
A = int(input("모드를 선택 하시오 : "))
N = int(input("길이를 입력 하시오 : "))
if A == 1:
    for i in range(N):
        if i == 0:
            print(" " * (N * 2 + 2), "☆", sep='')
        else:
            print(" " * (N * 2 + 2 - i), "☆", " " * (i * 2 - 1), "☆", sep='')

    print("☆" * N, " " * (N * 2 - 1), "☆" * N, sep='')

    for i in range(N):
        if i <= int(N / 3):
            print(" " * (N - 3 + i * 2), "☆", " " * (N * 4 - 1 - 4 * i), "☆", sep='')
        elif i == 2:
            print(" " * (N + 2 - i), "☆", " " * (i + 3), "☆", " " * (N + 2 - i), "☆", sep='')

    for i in range(N):
        if i <= int(N / 2):
            print(" " * (N - i - 1), "☆", " " * (N - 2 * i - 2), "☆", " " * (N + 4 * i + i), "☆", " " * (N - 3 - i),
                  "☆", sep='')

    print(" ", "☆", " " * (N * 4), "☆", sep='')
elif A == 2:
    for i in range(N):
        if i == 0:
            print(" " * (N * 2 + 2), "★", sep='')
        else:
            print(" " * (N * 2 + 2 - i), "★", "★" * (i * 2 - 1), "★", sep='')

    print("★" * N, "★" * (N * 2 - 1), "★" * N, sep='')

    for i in range(N):
        if i <= int(N / 3):
            print(" " * (N - 3 + i * 2), "★", "★" * (N * 3 - 1 - 4 * i), "★", sep='')
        elif i == 2:
            print(" " * (N + 2 - i), "★", "★" * (i + 3), "★", "★" * (N + 2 - i), sep='')

    for i in range(N):
        if i <= int(N / 2):
            print(" " * (N - i - 1), "★", "★" * (N - 2 * i - 2), "★", " " * (N + 5 * i + i), "★", "★" * (N - 3 - i + 2),
                  sep='')

    print(" ", "★", " " * (N * 4), "★", sep='')

elif A == 3:
    for i in range(N):
        if i == 0:
            print(" " * (N - 2 * i), "#", " " * (N - 2), "#", sep='')
        elif i == N - 1:
            print("#", " " * (2 * i - N), "#", sep='')
        elif i == 1 or i == N - 2:
            print("#" * N * 2, sep='')
        else:
            print(" " * (N - i), "#", " " * (N - 2), "#", sep='')
