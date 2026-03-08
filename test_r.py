import model

DEMO_BASE_MODEL_PATH = "data/diffusion_models/stable-diffusion-v1-5"
DEMO_LORA_WEIGHTS_PATH = "data/loras/SwiftSketch_main.safetensors"
OUTPUT_PATH = "output"
OUTPUT_FILENAME_TEXT = "text2image.jpg"
OUTPUT_FILENAME_IMAGE = "image2image.jpg"
TEXT_TEST = "star"
MASK_INPUT_TEST = "star"
INPUT_IMAGE = "output/c.jpg"
INPUT_IMAGE_MASK = "output/c.jpg"
INPUT_NEGATIVE_PROMPT = "blurry, pixelated, text"
INPUT_NUM_INFERENCE_STEP = 30
INPUT_GUIDANCE_SCALE = 7.5
INPUT_SEED = 42
INPUT_STRENGTH = 0.8
LORA_PROMPT = model.get_lora_prompt("SwiftSketch_main.safetensors")

# import tool
# RANDOM_SEED = tool.get_positive_random(2147483647)
# SPECIMENS =  tool.parse_specimens_jsonl_file_to_dict("data/specimens/SwiftSketch_specimens.jsonl")
# RANDOM_INPUT_PROMPT = SPECIMENS[f"{tool.get_positive_random(100)}"]

def test_text2image(generate_model):
    text2image_config = model.get_text2image_config(
        OUTPUT_DIR = OUTPUT_PATH,
        OUTPUT_FILENAME = OUTPUT_FILENAME_TEXT,
        LORA_PROMPT = LORA_PROMPT,
        PROMPT = TEXT_TEST,
        NEGATIVE_PROMPT = INPUT_NEGATIVE_PROMPT,
        NUM_INFERENCE_STEP = INPUT_NUM_INFERENCE_STEP,
        GUIDANCE_SCALE = INPUT_GUIDANCE_SCALE,
        SEED = INPUT_SEED)
    generate_model.text2image(text2image_config)

def test_image2image(generate_model):
    image2image_config = model.get_image2image_config(
        OUTPUT_DIR = OUTPUT_PATH,
        OUTPUT_FILENAME = OUTPUT_FILENAME_IMAGE,
        IMAGE = INPUT_IMAGE,
        MASK_IMAGE = INPUT_IMAGE_MASK,
        LORA_PROMPT = LORA_PROMPT,
        PROMPT = MASK_INPUT_TEST,
        NEGATIVE_PROMPT = INPUT_NEGATIVE_PROMPT,
        NUM_INFERENCE_STEP = INPUT_NUM_INFERENCE_STEP,
        GUIDANCE_SCALE = INPUT_GUIDANCE_SCALE,
        SEED = INPUT_SEED,
        STRENGTH = INPUT_STRENGTH)
    generate_model.image2image(image2image_config)

def main():
    generate_model = model.get_GenerateModel(BASE_MODEL_PATH = DEMO_BASE_MODEL_PATH,LORA_WEIGHTS_PATH = DEMO_LORA_WEIGHTS_PATH)
    # print("-" * 50)
    # print(f"随机种子: {RANDOM_SEED} 随机输入: {RANDOM_INPUT_PROMPT}")
    print("-" * 50)
    print(f"LORA_PROMPT: {LORA_PROMPT}")
    print("-" * 50)
    test_text2image(generate_model)
    print("-" * 50)
    test_image2image(generate_model)

if __name__ == "__main__":
    main()
