class Book:
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def display_info(self):
        print("책 제목: ", self.title)
        print("저자: ", self.author)
        print("가격: ", self.price)

    def __eq__(self, other):
        if isinstance(other, Book):
            return self.price == other.price
        return False


book1 = Book("어린왕자", "앙투안 드 생텍쥐페리", 13000)
book2 = Book("모모", "미하엘 엔데 저", 13000)

print(book1 == book2)