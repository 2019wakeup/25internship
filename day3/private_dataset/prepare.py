### prepare.py

import os

# 创建保存路径的函数
def create_txt_file(root_dir, txt_filename):
    # 打开并写入文件
    with open(txt_filename, 'w') as f:
        # 遍历每个类别文件夹
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                # 遍历该类别文件夹中的所有图片
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")

create_txt_file(r'D:\Desktop\tcl\dataset\image2\train', 'train.txt')
create_txt_file(r'D:\Desktop\tcl\dataset\image2\val', "val.txt")
