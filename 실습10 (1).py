class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.Attendance = False

    def mark_attendance(self):
        self.Attendance = True


class AttendanceBook:
    def __init__(self):
        self.student_list = []
        self.student_ids = set()

    def add_student(self, name, student_id):
        if student_id not in self.student_ids:
            new_student = Student(name, student_id)
            self.student_list.append(new_student)
            self.student_ids.add(student_id)

    def mark_student_attendance(self, student_id):
        for student in self.student_list:
            if student.student_id == student_id:
                student.mark_attendance()
                return

    def get_attendance_summary(self):
        summary = {"출석": 0, "결석": 0}
        for student in self.student_list:
            if student.Attendance == True:
                summary["출석"] = summary["출석"] + 1
            else:
                summary["결석"] = summary["결석"] + 1
        return summary

    def get_student_list(self):
        Attendance_students = []
        for student in self.student_list:
            if student.Attendance:
                Attendance_students.append(student.name)
        return Attendance_students


# 출석부 테스트
attendance_book = AttendanceBook()

# 학생 추가
attendance_book.add_student("김철수", 101)
attendance_book.add_student("이영희", 102)
attendance_book.add_student("박민수", 103)
attendance_book.add_student("김철수", 101)  # 중복 학번

# 출석 체크
attendance_book.mark_student_attendance(101)
attendance_book.mark_student_attendance(103)

# 출석 요약 및 출석한 학생 목록 출력
print("출석 요약:", attendance_book.get_attendance_summary())
print("출석한 학생 목록:", attendance_book.get_student_list())
