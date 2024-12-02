A = int(input("과제 : "))
B = int(input("중간 : "))
C = int(input("기말 : "))
D = (A + B + C) / 3
print("평균 :", round((A + B + C) / 3, 6))
if D >= 90:
    print("A학점 입니다.")
elif D >= 80:
    print("B학점 입니다.")
elif D >= 70:
    print("C학점 입니다.")
elif D >= 60:
    print("D학점 입니다.")
else:
    print("F학점 입니다.")
