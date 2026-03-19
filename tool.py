import json
import os
import random
from datetime import datetime
import numpy as np
from PIL import Image

def parse_specimens_jsonl_file2dict(FILE_PATH: str, KEY: str, VALUE: str) -> dict:
    specimens_dict = {}
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return specimens_dict

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:  # 确保行不为空
                    try:
                        data = json.loads(stripped_line)
                        dict_key = data.get(KEY)
                        dict_value = data.get(VALUE)
                        if dict_key is not None and dict_value is not None:
                            specimens_dict[dict_key] = dict_value
                        else:
                            print(f"Warning: Missing 'dict_key' or 'dict_value' in line: {stripped_line}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line '{stripped_line}'. Error: {e}")
    except IOError as e:
        print(f"Error reading file {FILE_PATH}: {e}")

    return specimens_dict

def get_random(MIN_NUM: int, MAX_NUM: int) -> int:
    return random.randint(MIN_NUM, MAX_NUM)

def get_positive_random(MAX_NUM: int) -> int:
    return random.randint(1, MAX_NUM)

def get_int_random(MAX_NUM: int) -> int:
    return random.randint(0, MAX_NUM)

def get_current_time_str() -> str:
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H-%M-%S")
    return time_str

def get_file_list(DIR_PATH = "./") -> list:
    file_list = []
    if not os.path.isdir(DIR_PATH):
        print(f"错误：目录 {DIR_PATH} 不存在！")
        return file_list
    file_list = os.listdir(DIR_PATH)

    return file_list

def get_save_path(DIR_PATH: str, FILE_NAME: str) -> str:
    return os.path.join(DIR_PATH, FILE_NAME)

def get_numpy_image(COMPOSITE_NP_ARRAY: np.ndarray) -> Image:
    return Image.fromarray(COMPOSITE_NP_ARRAY).convert("RGB")

def remove_file_path(FILE_PATH: str) -> None:
    os.remove(FILE_PATH)

def create_data_dir() -> None:
    directories = [
        "data/diffusion_models",
        "data/loras",
        "output/image2image",
        "output/text2image"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(dir_path):
            print(f"目录已存在/创建成功：{dir_path}")
        else:
            print(f"目录创建失败：{dir_path}")
