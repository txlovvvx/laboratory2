from datetime import datetime
import os
import cv2
import numpy as np

input_folder = "data/input"
output_file = "data/output/result.jpg"
log_file = "logs/history.txt"
template_file = "template.jpg"

def write_log(filename, status):
  now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  line = f"[{now}] файл: {filename} | фильтр: Unsharp Masking + цветокоррекция по шаблону | статус: {status}\n"
  with open(log_file, "a", encoding="utf-8") as f:
    f.write(line)

filename = input("введите имя файла (например, image1.jpg или image2.png): ")
filepath = os.path.join(input_folder, filename)

try:
  f = open(filepath, "rb")
  f.close()
except FileNotFoundError:
  print("ошибка: файл не найден.")
  write_log(filename, "ошибка (файл не найден)")
  exit()
except PermissionError:
  print(
      f"ошибка: нет прав на чтение файла {filename}. обратитесь к"
      " администратору."
  )
  write_log(filename, "ошибка доступа")
  exit()

img = cv2.imread(filepath)
template = cv2.imread(template_file)

if img is None:
  print("ошибка: не удалось открыть изображение.")
  write_log(filename, "ошибка чтения")
  exit()

if template is None:
  print("ошибка: не найден файл шаблона template.jpg.")
  write_log(filename, "ошибка (нет шаблона)")
  exit()

img_float = img.astype(np.float32)
blurred = cv2.GaussianBlur(img_float, (0, 0), 2)
mask = img_float - blurred
sharpened = np.clip(img_float + 0.8 * mask, 0, 255).astype(np.uint8)

src_lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB).astype(np.float32)
ref_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB).astype(np.float32)

for i in range(3):
  src_mean = src_lab[:, :, i].mean()
  src_std = src_lab[:, :, i].std()
  ref_mean = ref_lab[:, :, i].mean()
  ref_std = ref_lab[:, :, i].std()

  if src_std != 0:
    src_lab[:, :, i] = (src_lab[:, :, i] - src_mean) * (
        ref_std / src_std
    ) + ref_mean
  else:
    src_lab[:, :, i] = ref_mean

result = np.clip(src_lab, 0, 255).astype(np.uint8)
result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

cv2.imwrite(output_file, result)
print("результат сохранен в Lab2_Media/data/output/result.jpg")

write_log(filename, "успешно")