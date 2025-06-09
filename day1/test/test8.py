# 写文件
with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")

# 读文件
with open("example.txt", "r") as f:
    content = f.read()
    print(content)

# 处理CSV
import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])
