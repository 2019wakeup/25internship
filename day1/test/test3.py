# 条件语句
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

# 循环语句
for i in range(5):
    if i == 3:
        continue
    print(i)

# 异常处理
try:
    num = int(input("Enter a number: "))
    print(100 / num)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Execution completed.")
