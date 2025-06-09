# 定义类
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old."

# 继承
class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major

    def introduce(self):
        return f"I am {self.name}, a {self.major} student."

# 使用
student = Student("Alice", 20)
grad = GradStudent("Bob", 22, "CS")
print(student.introduce())  # I am Alice, 20 years old.
print(grad.introduce())     # I am Bob, a CS student.
