path = r".\waveguide.png"
import base64


# 转化成64*64的图片
import PIL
from PIL import Image
img = Image.open(path)
img = img.resize((64, 64))
img.save(r".\waveguide64.png")


with open(r".\waveguide64.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
print(encoded_string)

# 将原始的图片保存为ico
img = Image.open(path)
img.save(r".\waveguide.ico")