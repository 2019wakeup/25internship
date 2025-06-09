# 创建模块 mymodule.py
# mymodule.py
def say_hello():
    return "Hello from module!"

# 主程序
import mymodule
print(mymodule.say_hello())

# 导入第三方模块
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200

# 包使用示例
from mypackage import mymodule
