import os

import model
import tool

DEMO_BASE_MODEL_PATH = "data/diffusion_models/stable-diffusion-v1-5"
DEMO_LORA_WEIGHTS_PATH = "data/loras/SwiftSketch_main.safetensors"
OUTPUT_PATH = "output/test/test4img"
# OUTPUT_FILENAME_TEXT = "text2image.jpg"
OUTPUT_FILENAME_IMAGE = "image2image.jpg"
TEXT_TEST = "dog"
MASK_INPUT_TEST = "star"
# INPUT_IMAGE = "output/c.jpg"
# INPUT_IMAGE_MASK = "output/c.jpg"
INPUT_NEGATIVE_PROMPT = "blurry, pixelated, text"
INPUT_NUM_INFERENCE_STEP = 32
INPUT_GUIDANCE_SCALE = 7.5
# INPUT_SEED = 42
INPUT_STRENGTH = 0.8
LORA_PROMPT = model.get_lora_prompt("SwiftSketch_main.safetensors")
GENERATE_MODEL = model.get_GenerateModel(BASE_MODEL_PATH = DEMO_BASE_MODEL_PATH, LORA_WEIGHTS_PATH = DEMO_LORA_WEIGHTS_PATH)
SEED_LIST = [42, 114, 758, 2556, 2200, 4396, 1557, 8884844, 28275, 9999]
TRAIN_SPECIMEN_NAMES = [
    "angel",
    "ant",
    "apple",
    "astronaut",
    "backpack",
    "banana",
    "bear",
    "bed",
    "beer",
    "bee",
    "bicycle",
    "boat",
    "book",
    "broccoli",
    "bus",
    "butterfly",
    "cabin",
    "cake",
    "camel",
    "camera",
    "candle",
    "carrot",
    "car",
    "castle",
    "cat",
    "chair",
    "child",
    "clock",
    "cow",
    "crab",
    "cup",
    "deer",
    "dog",
    "dolphin",
    "dragon",
    "drill",
    "duck",
    "elephant",
    "fish",
    "flamingo",
    "floor lamp",
    "flower",
    "fork",
    "giraffe",
    "hammer",
    "hat",
    "helicopter",
    "horse",
    "ice cream",
    "jacket",
    "kangaroo",
    "kimono",
    "laptop",
    "lion",
    "lobster",
    "man",
    "margarita",
    "moon",
    "motorcycle",
    "mountain",
    "octopus",
    "parrot",
    "pen",
    "phone",
    "pig",
    "pizza",
    "purse",
    "quiche",
    "rabbit",
    "robot",
    "sandwich",
    "scissors",
    "sculpture",
    "shark",
    "sheep",
    "spider",
    "squirrel",
    "star",
    "strawberry",
    "submarine",
    "sword",
    "t-shirt",
    "table",
    "teapot",
    "television",
    "The Eiffel Tower",
    "tiger",
    "tomato",
    "train",
    "truck",
    "vase",
    "waffle",
    "watch",
    "whale",
    "windmill",
    "wine bottle",
    "woman",
    "yoga",
    "zebra",
    "mermaid"
]
TEST_SPECIMEN_NAMES = [
    "airplane",
    "alligator",
    "anchor",
    "antler",
    "avocado",
    "axe",
    "baby",
    "ball",
    "balloon",
    "bat",
    "beach",
    "bell",
    "belt",
    "bird",
    "blender",
    "bow",
    "bowl",
    "bracelet",
    "bread",
    "brush",
    "bubble",
    "bucket",
    "butter",
    "cactus",
    "calendar",
    "camera",
    "canoe",
    "cap",
    "card",
    "castle",
    "cat",
    "chain",
    "chair",
    "chameleon",
    "cherry",
    "cheese",
    "chicken",
    "circle",
    "clip",
    "cloud",
    "coat",
    "coin",
    "compass",
    "cookie",
    "corn",
    "couch",
    "crayon",
    "crown",
    "cupcake",
    "daisy",
    "diamond",
    "dinner",
    "dinosaur",
    "dog",
    "door",
    "drawer",
    "dress",
    "drum",
    "ear",
    "egg",
    "electricity",
    "engine",
    "eraser",
    "eye",
    "fan",
    "feather",
    "fence",
    "fire",
    "fish",
    "flag",
    "frog",
    "fruit",
    "game",
    "garage",
    "garden",
    "ghost",
    "gift",
    "glasses",
    "glove",
    "goat",
    "grape",
    "grass",
    "guitar",
    "gun",
    "hair",
    "hammer",
    "hand",
    "heart",
    "hill",
    "home",
    "horn",
    "horse",
    "hospital",
    "hot",
    "house",
    "ice",
    "island",
    "jelly",
    "juice",
    "key",
    "kitchen",
    "knife",
    "leaf",
    "leg",
    "light",
    "lock",
    "lunch",
    "magnet",
    "map",
    "mask",
    "match",
    "milk",
    "monkey",
    "mouse",
    "mouth",
    "nail",
    "neck",
    "needle",
    "nest",
    "notebook",
    "nut",
    "ocean",
    "oil",
    "orange",
    "paint",
    "palm",
    "panda",
    "paper",
    "park",
    "peach",
    "pear",
    "pen",
    "pencil",
    "phone",
    "piano",
    "pie",
    "pig",
    "pillow",
    "pine",
    "pipe",
    "pizza",
    "plane",
    "plant",
    "plate",
    "potato",
    "present",
    "prince",
    "princess",
    "pumpkin",
    "pyramid",
    "queen",
    "quilt",
    "rabbit",
    "rain",
    "rainbow",
    "rat",
    "ring",
    "river",
    "rock",
    "rocket",
    "roof",
    "rose",
    "ruler",
    "salad",
    "salt",
    "sandal",
    "sauce",
    "saw",
    "sailboat",
    "school",
    "scooter",
    "screen",
    "screw",
    "sea",
    "seed",
    "shoe",
    "shop",
    "silver",
    "sing",
    "sink",
    "skirt",
    "sky",
    "sleep",
    "smile",
    "snow",
    "soap",
    "sock",
    "song",
    "sound",
    "spoon",
    "spring",
    "square",
    "stamp",
    "star",
    "station",
    "steam",
    "steel",
    "stick",
    "stomach",
    "stone",
    "store",
    "storm",
    "story",
    "stove",
    "straight",
    "string",
    "sun",
    "swim",
    "swing",
    "table",
    "tail",
    "talk",
    "tea",
    "teacher",
    "tent",
    "text",
    "tree",
    "triangle",
    "trousers",
    "truck",
    "tube",
    "turtle",
    "tv",
    "umbrella",
    "underwear",
    "valley",
    "van",
    "vegetable",
    "village",
    "violin",
    "volcano",
    "wagon",
    "wall",
    "war",
    "wash",
    "waste",
    "watch",
    "water",
    "wave",
    "wax",
    "way",
    "weapon",
    "weather",
    "web",
    "wedding",
    "wheel",
    "whistle",
    "window",
    "wine",
    "wing",
    "wire",
    "wolf",
    "wood",
    "wool",
    "word",
    "work",
    "world",
    "worm",
    "wound",
    "wrap",
    "wreath",
    "wrench",
    "write",
    "xylophone",
    "yacht",
    "yard",
    "yellow",
    "zoo"
]

# import tool
# RANDOM_SEED = tool.get_positive_random(2147483647)
# SPECIMENS =  tool.parse_specimens_jsonl_file_to_dict("data/specimens/SwiftSketch_specimens.jsonl")
# RANDOM_INPUT_PROMPT = SPECIMENS[f"{tool.get_positive_random(100)}"]

def test_text2image(file_name,seed):
    text2image_config = model.get_text2image_config(
        OUTPUT_DIR = OUTPUT_PATH,
        OUTPUT_FILENAME = file_name,
        LORA_PROMPT = LORA_PROMPT,
        PROMPT = TEXT_TEST,
        NEGATIVE_PROMPT = INPUT_NEGATIVE_PROMPT,
        NUM_INFERENCE_STEP = INPUT_NUM_INFERENCE_STEP,
        GUIDANCE_SCALE = INPUT_GUIDANCE_SCALE,
        SEED = seed)
    GENERATE_MODEL.text2image(text2image_config)

# def test_image2image(generate_model):
#     image2image_config = model.get_image2image_config(
#         OUTPUT_DIR = OUTPUT_PATH,
#         OUTPUT_FILENAME = OUTPUT_FILENAME_IMAGE,
#         IMAGE = INPUT_IMAGE,
#         MASK_IMAGE = INPUT_IMAGE_MASK,
#         LORA_PROMPT = LORA_PROMPT,
#         PROMPT = MASK_INPUT_TEST,
#         NEGATIVE_PROMPT = INPUT_NEGATIVE_PROMPT,
#         NUM_INFERENCE_STEP = INPUT_NUM_INFERENCE_STEP,
#         GUIDANCE_SCALE = INPUT_GUIDANCE_SCALE,
#         SEED = INPUT_SEED,
#         STRENGTH = INPUT_STRENGTH)
#     generate_model.image2image(image2image_config)

def test_new_text2image(file_name,seed):
    text2image_config = model.get_text2image_config(
        OUTPUT_DIR = OUTPUT_PATH,
        OUTPUT_FILENAME = file_name,
        LORA_PROMPT = LORA_PROMPT,
        PROMPT = TEXT_TEST,
        NEGATIVE_PROMPT = INPUT_NEGATIVE_PROMPT,
        NUM_INFERENCE_STEP = INPUT_NUM_INFERENCE_STEP,
        GUIDANCE_SCALE = INPUT_GUIDANCE_SCALE,
        SEED = seed)
    GENERATE_MODEL.text2image(text2image_config)

def test_loop_text2image(max_batch):
    for i in range(max_batch):
        file_name = f"test_loop_text2image_{i}.jpg"
        seed = tool.get_positive_random(2147483647)
        test_new_text2image(file_name,seed)


def test_loop_LoRA_adapter_weight():
    for i in range(11):
        LoRA_adapter_weight_num = float(i) / 10
        GENERATE_MODEL.set_LoRA_adapter_weights(LoRA_adapter_weight_num)
        for j in range(10):
            test_new_text2image(f"test_{LoRA_adapter_weight_num}_{SEED_LIST[j]}.jpg",SEED_LIST[j])

def test_loop_CFG_scale():
    def test_run(file_name, seed,cfg_scale):
        text2image_config = model.get_text2image_config(
            OUTPUT_DIR=OUTPUT_PATH,
            OUTPUT_FILENAME=file_name,
            LORA_PROMPT=LORA_PROMPT,
            PROMPT=TEXT_TEST,
            NEGATIVE_PROMPT=INPUT_NEGATIVE_PROMPT,
            NUM_INFERENCE_STEP=INPUT_NUM_INFERENCE_STEP,
            GUIDANCE_SCALE=cfg_scale,
            SEED=seed)
        GENERATE_MODEL.text2image(text2image_config)
    for i in range(11):
        cfg_scale_num = 2.0 + float(i)
        for j in range(10):
            test_run(f"test_{cfg_scale_num}_{SEED_LIST[j]}.jpg",SEED_LIST[j],cfg_scale_num)

def test_loop_step():
    def test_run(file_name, seed,input_step):
        text2image_config = model.get_text2image_config(
            OUTPUT_DIR=OUTPUT_PATH,
            OUTPUT_FILENAME=file_name,
            LORA_PROMPT=LORA_PROMPT,
            PROMPT=TEXT_TEST,
            NEGATIVE_PROMPT=INPUT_NEGATIVE_PROMPT,
            NUM_INFERENCE_STEP=input_step,
            GUIDANCE_SCALE=INPUT_GUIDANCE_SCALE,
            SEED=seed)
        GENERATE_MODEL.text2image(text2image_config)
    for i in range(11):
        step = 22 + 2 * i
        for j in range(10):
            test_run(f"test_{step}_{SEED_LIST[j]}.jpg",SEED_LIST[j],step)

def test_train_loop_text():
    def test_run(file_name, seed,input_prompt):
        text2image_config = model.get_text2image_config(
            OUTPUT_DIR=OUTPUT_PATH,
            OUTPUT_FILENAME=file_name,
            LORA_PROMPT=LORA_PROMPT,
            PROMPT=input_prompt,
            NEGATIVE_PROMPT=INPUT_NEGATIVE_PROMPT,
            NUM_INFERENCE_STEP=INPUT_NUM_INFERENCE_STEP,
            GUIDANCE_SCALE=INPUT_GUIDANCE_SCALE,
            SEED=seed)
        GENERATE_MODEL.text2image(text2image_config)
    for i in range(100):
        prompt = TRAIN_SPECIMEN_NAMES[i]
        for j in range(10):
            test_run(f"test_{prompt}_{SEED_LIST[j]}.jpg",SEED_LIST[j],prompt)

def test_test_loop_text():
    def test_run(file_name, seed,input_prompt):
        text2image_config = model.get_text2image_config(
            OUTPUT_DIR=OUTPUT_PATH,
            OUTPUT_FILENAME=file_name,
            LORA_PROMPT=LORA_PROMPT,
            PROMPT=input_prompt,
            NEGATIVE_PROMPT=INPUT_NEGATIVE_PROMPT,
            NUM_INFERENCE_STEP=INPUT_NUM_INFERENCE_STEP,
            GUIDANCE_SCALE=INPUT_GUIDANCE_SCALE,
            SEED=seed)
        GENERATE_MODEL.text2image(text2image_config)
    for i in range(100):
        prompt = TEST_SPECIMEN_NAMES[i]
        for j in range(10):
            test_run(f"test_{prompt}_{SEED_LIST[j]}.jpg",SEED_LIST[j],prompt)

def test_loop_img():
    def test_image2image(file_name,output_path,input_image_path,seed,input_prompt):
        image2image_config = model.get_image2image_config(
            OUTPUT_DIR=output_path,
            OUTPUT_FILENAME=file_name,
            IMAGE=input_image_path,
            MASK_IMAGE=input_image_path,
            LORA_PROMPT=LORA_PROMPT,
            PROMPT=input_prompt,
            NEGATIVE_PROMPT=INPUT_NEGATIVE_PROMPT,
            NUM_INFERENCE_STEP=INPUT_NUM_INFERENCE_STEP,
            GUIDANCE_SCALE=INPUT_GUIDANCE_SCALE,
            SEED=seed,
            STRENGTH=INPUT_STRENGTH)
        GENERATE_MODEL.image2image(image2image_config)
    def get_bad_image_path(class_name, num):
        base_dir = "output/test/test4img/input"
        num_str = f"{num:03d}"
        filename = f"{class_name}_{num_str}_bad.jpg"
        full_path = os.path.join(base_dir, class_name, filename)
        return full_path
    for prompt_text in ["cake","cow","flower","phone","tiger"]:
        for i in range(1,101):
            image_path = get_bad_image_path(prompt_text,i)
            save_name = f"{prompt_text}_{i}.jpg"
            save_path = f"output/test/test4img/save/{prompt_text}"
            test_image2image(save_name,save_path,image_path,42,prompt_text)
    pass

def test_img2():
    def test_image2image(s,name,seed):
        image2image_config = model.get_image2image_config(
            OUTPUT_DIR=OUTPUT_PATH,
            OUTPUT_FILENAME=name,
            IMAGE="output/test/test4img/save/tiger/tiger_85.jpg",
            MASK_IMAGE="output/test/test4img/save/tiger/tiger_85.jpg",
            LORA_PROMPT=LORA_PROMPT,
            PROMPT="tiger",
            NEGATIVE_PROMPT=INPUT_NEGATIVE_PROMPT,
            NUM_INFERENCE_STEP=INPUT_NUM_INFERENCE_STEP,
            GUIDANCE_SCALE=INPUT_GUIDANCE_SCALE,
            SEED=seed,
            STRENGTH=s)
        GENERATE_MODEL.image2image(image2image_config)
    for i in range(1,11):
        strength = float(i) / 10
        for j in range(10):
            file_name = f"test_{strength}_{SEED_LIST[j]}_.jpg"
            test_image2image(strength,file_name,SEED_LIST[j])

def main():
    # generate_model = model.get_GenerateModel(BASE_MODEL_PATH = DEMO_BASE_MODEL_PATH,LORA_WEIGHTS_PATH = DEMO_LORA_WEIGHTS_PATH)
    # print("-" * 50)
    # print(f"随机种子: {RANDOM_SEED} 随机输入: {RANDOM_INPUT_PROMPT}")
    # print("-" * 50)
    # print(f"LORA_PROMPT: {LORA_PROMPT}")
    # print("-" * 50)
    # test_text2image(generate_model)
    # print("-" * 50)
    # test_image2image(generate_model)
    # print("-" * 50)
    # test_new_text2image(generate_model)
    # print("-" * 50)
    # test_loop_text2image(generate_model,50)
    # test_loop_LoRA_adapter_weight()
    # test_loop_CFG_scale()
    # test_loop_step()
    # test_train_loop_text()
    # test_test_loop_text()
    # test_loop_img()
    test_img2()
    pass

if __name__ == "__main__":
    main()
