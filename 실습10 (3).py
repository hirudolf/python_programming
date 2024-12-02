class Movie:
    def __init__(self, title, schedule):
        self.title = title
        self.schedule = schedule
        self.seats = {time: [False] * 10 for time in schedule}

    def reserve_seat(self, time, seat_number):
        if self.seats[time][seat_number] == True:
            print("이미 예약된 좌석입니다.")
        else:
            self.seats[time][seat_number] = True
            print("예약 완료")

    def get_available_seats(self, time):
        return self.seats[time].count(False)

class Theater:
    def __init__(self):
        self.movies = {}

    def add_movie(self, title, schedule):
        if title not in self.movies:
            self.movies[title] = Movie(title, schedule)
            print(f"'{title}' 영화가 추가되었습니다.")

    def reserve_movie_seat(self, title, time, seat_number):
        if title not in self.movies:
            print(f"오류: '{title}' 영화 이름 오류.")
            return
        if time not in self.movies[title].schedule:
            print(f"오류: '{title}' 해당 시간이 아님.")
            return
        self.movies[title].reserve_seat(time, seat_number)

    def get_movie_schedule(self, title):
        print(f"'{title}' 영화의 상영 시간표:")
        for time in self.movies[title].schedule:
            available_seats = self.movies[title].get_available_seats(time)
            print(f"  시간: {time} - 예약 가능한 좌석 수: {available_seats}")


# 영화관 시스템 테스트
theater = Theater()

# 영화 추가
theater.add_movie("인셉션", ["14:00", "17:00", "20:00"])
theater.add_movie("인터스텔라", ["13:00", "16:00", "19:00"])

# 좌석 예약
theater.reserve_movie_seat("인셉션", "14:00", 3)
theater.reserve_movie_seat("인셉션", "14:00", 3)  # 이미 예약된 좌석
theater.reserve_movie_seat("인터스텔라", "19:00", 5)

# 영화의 상영 시간표와 예약 가능한 좌석 수 확인
theater.get_movie_schedule("인셉션")
theater.get_movie_schedule("인터스텔라")
