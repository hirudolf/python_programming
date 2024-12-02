#문제 1: 무한 Generator=
def infi_num():
    num = 0
    while True:
        yield num
        num += 1

gen = infi_num()

for i in range(11):
    print(next(gen))



#문제 2: 제너레이터와 코루틴 결합
def infi_num():
    num = 0
    while True:
        yield num
        num += 1

def corutine():
    while True:
        value = yield
        print(value * 2)


gen = infi_num()
co = corutine()
next(co)

for i in range(11):
    num = next(gen)
    co.send(num)




#문제 3: 텍스트 처리 시스템
def read_lines(file_name):
    with open(file_name) as f:
        for line in f:
            yield line.strip()

def corutine():
    while True:
        line = yield
        print(line.upper())

gen = read_lines('hihi.txt')
co = corutine()
next(co)

for line in read_lines('hihi.txt'):
    print(line)
    co.send(line)



