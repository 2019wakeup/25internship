# Python与Git基础课程

## 1. 环境准备

```bash
# 检查Python版本
python --version

# 创建并激活虚拟环境
python -m venv myenv
# Windows下激活
myenv\Scripts\activate
# Linux/Mac下激活
source myenv/bin/activate

# 安装第三方库
pip install requests
```

---

## 2. 变量、变量类型、作用域

- **基本类型**：int, float, str, bool, list, tuple, dict, set
- **作用域**：全局变量、局部变量，关键字 global、nonlocal
- **类型转换**：int()、float()、str() 等

```python
name = "Alice"           # str
age = 20                 # int
grades = [90, 85, 88]    # list
info = {"name": "Alice", "age": 20}   # dict

# 类型转换
age_str = str(age)
number = int("123")

# 作用域示例
x = 10  # 全局变量
def my_function():
    global x
    y = 5  # 局部变量
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")
```

---

## 3. 运算符及表达式

- 算术：+  -  *  /  //  %  **
- 比较：==  !=  >  <  >=  <=
- 逻辑：and  or  not
- 位运算：&  |  ^  <<  >>

```python
a = 10
b = 3
print(a + b)   # 13
print(a // b)  # 3
print(a ** b)  # 1000
x = True
y = False
print(x and y)  # False
print(a > b)    # True
```

---

## 4. 语句：条件、循环、异常

- 条件：if, elif, else
- 循环：for, while, break, continue
- 异常处理：try, except, finally

```python
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

for i in range(5):
    if i == 3:
        continue
    print(i)

try:
    num = int(input("Enter a number: "))
    print(100 / num)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Execution completed.")
```

---

## 5. 函数：定义、参数、匿名函数、高阶函数

- 定义函数：def
- 默认参数、可变参数：*args, **kwargs
- 匿名函数：lambda
- 高阶函数：函数作为参数/返回值

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))
print(greet("Bob", "Hi"))

def sum_numbers(*args):
    return sum(args)
print(sum_numbers(1, 2, 3))  # 6

double = lambda x: x * 2
print(double(5))  # 10

def apply_func(func, value):
    return func(value)
print(apply_func(lambda x: x ** 2, 4))  # 16
```

---

## 6. 包和模块

- 创建模块：写一个.py文件
- 导入模块：import、from ... import ...
- 创建包：含__init__.py的文件夹
- 安装第三方模块：pip install

```python
# mymodule.py
def say_hello():
    return "Hello from module!"

# main.py
import mymodule
print(mymodule.say_hello())

import requests
r = requests.get("https://api.github.com")
print(r.status_code)
```

---

## 7. 类和对象

- 定义类、属性、方法
- 继承：class 子类(父类)
- 实例化对象

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old."

class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major

    def introduce(self):
        return f"I am {self.name}, a {self.major} student."

student = Student("Alice", 20)
grad = GradStudent("Bob", 22, "CS")
print(student.introduce())
print(grad.introduce())
```

---

## 8. 装饰器

- 高阶函数与@语法
- 带参数装饰器

```python
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hi, {name}!")

greet("Alice")
```

---

## 9. 文件操作

- 读写文本文件：open/read/write
- with上下文管理器
- 处理CSV、JSON文件

```python
# 写文件
with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")

# 读文件
with open("example.txt", "r") as f:
    content = f.read()
    print(content)

# CSV
import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])
```



# Git 常见命令操作详解

## 1. 配置Git

```bash
git config --global user.name "你的名字"     # 设置全局用户名
git config --global user.email "你的邮箱"     # 设置全局邮箱
git config --global core.editor "vim"        # 设置默认编辑器(可选)
git config --list                            # 查看当前配置
```

---

## 2. 创建/克隆仓库

- **初始化本地仓库**
    ```bash
    git init
    ```

- **克隆远程仓库**
    ```bash
    git clone [仓库地址]
    # 示例
    git clone https://github.com/user/repo.git
    ```

---

## 3. 添加与提交

- **查看当前状态**
    ```bash
    git status
    ```
- **添加文件到暂存区**
    ```bash
    git add 文件名      # 添加单个文件
    git add .           # 添加当前目录所有更改
    ```
- **撤销暂存（从暂存区移回工作区）**
    ```bash
    git reset 文件名
    ```
- **提交到本地仓库**
    ```bash
    git commit -m "本次提交的说明"
    ```

---

## 4. 查看历史与版本回退

- **查看提交历史**
    ```bash
    git log
    git log --oneline          # 简洁模式
    git log --graph --all      # 图形视图
    ```

- **查看文件变化**
    ```bash
    git diff                   # 查看未暂存的变化
    git diff --cached          # 查看已暂存但未提交的变化
    ```

- **版本回退**
    ```bash
    git reset --hard HEAD^         # 回退到上一个版本
    git reset --hard commit_id     # 回退到指定版本
    git reflog                     # 查看所有历史移动记录
    ```

---

## 5. 分支管理

- **查看当前分支**
    ```bash
    git branch
    ```
- **创建新分支**
    ```bash
    git branch 新分支名
    ```
- **切换分支**
    ```bash
    git checkout 分支名
    ```
- **创建并切换新分支**
    ```bash
    git checkout -b 新分支名
    ```
- **合并分支（如将 new-feature 合并到当前分支）**
    ```bash
    git merge new-feature
    ```
- **删除分支**
    ```bash
    git branch -d 分支名         # 合并后删除
    git branch -D 分支名         # 强制删除
    ```

---

## 6. 远程仓库操作

- **添加远程地址**
    ```bash
    git remote add origin [仓库地址]
    # 查看远程仓库
    git remote -v
    ```

- **推送到远程仓库**
    ```bash
    git push origin main           # 推送 main 分支
    git push -u origin 分支名      # 首次推送并建立关联
    ```

- **拉取远程仓库内容**
    ```bash
    git pull origin main
    ```
    > 加 `--rebase` 可避免多余的合并提交:
    > `git pull --rebase origin main`

- **克隆时选择分支**
    ```bash
    git clone -b develop [仓库地址]
    ```

---

## 7. 恢复与撤销操作

- **撤销对工作区的修改**
    ```bash
    git checkout -- 文件名     # 恢复某文件到上一次提交的状态
    ```

- **删除文件（并在Git中记录移除）**
    ```bash
    git rm 文件名
    git commit -m "删除文件"
    ```

---

## 8. tag 标签管理（常用于发布版本）

- **创建标签**
    ```bash
    git tag v1.0
    ```
- **查看标签**
    ```bash
    git tag
    ```
- **推送标签到远程**
    ```bash
    git push origin v1.0
    git push origin --tags      # 推送所有标签
    ```

---

## 9. 忽略文件

- **使用 `.gitignore` 文件，按行写入要忽略的文件或目录名**
    ```
    # 忽略所有 .pyc 文件
    *.pyc
    # 忽略某个文件夹
    log/
    # 忽略某个具体文件
    secret.txt
    ```

---

## 10. 常见快捷总结

| 操作            | 命令例子                    |
| --------------- | --------------------------- |
| 初始化仓库      | `git init`                  |
| 克隆仓库        | `git clone url`             |
| 添加到暂存区    | `git add .`                 |
| 提交            | `git commit -m "注释"`      |
| 查看状态        | `git status`                |
| 查看历史        | `git log --oneline`         |
| 创建&切换新分支 | `git checkout -b newbranch` |
| 合并分支        | `git merge other-branch`    |
| 推送到远程      | `git push origin 分支名`    |
| 拉取远程内容    | `git pull origin main`      |

