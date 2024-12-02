import statistics

class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def calculate_average(self):
        return statistics.mean(self.scores)

class GradeBook:
    def __init__(self):
        self.students = []
        self.student_ids = set()

    def add_student(self, name, student_id):
        if student_id not in self.student_ids:
            new_student = Student(name, student_id)
            self.students.append(new_student)
            self.student_ids.add(student_id)

    def add_student_score(self, student_id, score):
        for student in self.students:
            if student.student_id == student_id:
                student.add_score(score)
                return

    def get_top_students(self, n):
        students_avg = []
        for student in self.students:
            name = student.name
            average = student.calculate_average()
            students_avg.append((average, name))
        students_avg.sort(reverse=True)
        return students_avg[:n]

# 성적부 테스트
grade_book = GradeBook()

# 학생 추가
grade_book.add_student("김철수", 101)
grade_book.add_student("이영희", 102)
grade_book.add_student("박민수", 103)

# 학생 점수 추가
grade_book.add_student_score(101, 88)
grade_book.add_student_score(101, 92)
grade_book.add_student_score(102, 75)
grade_book.add_student_score(102, 80)
grade_book.add_student_score(103, 95)
grade_book.add_student_score(103, 90)

# 상위 2명의 학생 출력
top_students = grade_book.get_top_students(2)
print("상위 2명 학생의 성적:", top_students)
