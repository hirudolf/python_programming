class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def display_info(self):
        print("이름:", self.name, "| 급여:", self.salary)


class Manager:
    def __init__(self, team_members):
        self.team_members = team_members

    def add_team_member(self, employee):
        self.team_members.append(employee)

    def display_team(self):
        print("팀원 목록")
        for member in self.team_members:
            member.display_info()


team = []
manager = Manager(team)

employee1 = Employee("홍길동", 50000)
employee2 = Employee("김영희", 60000)

manager.add_team_member(employee1)
manager.add_team_member(employee2)

manager.display_team()
