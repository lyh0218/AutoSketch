# model.py
import os
import time
import torch
from tool import parse_specimens_jsonl_file2dict
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline,DiffusionPipeline

class _GenerateModel:
    def __init__(self,
                 base_model_path: str,
                 lora_weights_path: str):
        # self._pipe: DiffusionPipeline
        self._inpaint_pipe: StableDiffusionInpaintPipeline
        self._base_model_path: str = base_model_path
        self._lora_weights_path: str = lora_weights_path
        self._load_model_spend_time: float
        self._load_model()

    def _load_model(self):
        print(f"加载基础模型: {self._base_model_path}")
        print(f"加载LORA模型: {self._lora_weights_path}")
        start_time = time.time()
        try:
            # self._pipe = DiffusionPipeline.from_pretrained(
            #     self._base_model_path,
            #     torch_dtype=torch.float16,
            #     safety_checker=None
            # )
            self._inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self._base_model_path,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        except Exception as err:
            print(f"基础模型加载失败 '{self._base_model_path}': {err}")
            self._pipe = None
            return
        # self._pipe = self._pipe.to("cuda")
        self._inpaint_pipe = self._inpaint_pipe.to("cuda")
        try:
            # self._pipe.load_lora_weights(self._lora_weights_path)
            self._inpaint_pipe.load_lora_weights(self._lora_weights_path,adapter_name = "LoRA-adapter")
            print("成功加载LORA模型")
        except Exception as err:
            print(f"LORA模型加载失败:'{self._lora_weights_path}': {err}")
            self._pipe = None
            return
        end_time = time.time()
        spend_time = end_time - start_time
        self._load_model_spend_time = spend_time
        print(f"模型加载完成 花费{spend_time:.3f}s")

    def get_load_model_spend_time(self) -> float:
        return self._load_model_spend_time

    # def text2image(self, CONFIG: dict) -> list:
    #     start_time = time.time()
    #     generator = torch.Generator(device="cuda").manual_seed(CONFIG["seed"])
    #     save_path = CONFIG["save_path"]
    #     with torch.no_grad():
    #         image = self._pipe(
    #             prompt = CONFIG["prompt"],
    #             negative_prompt = CONFIG["negative_prompt"],
    #             guidance_scale = CONFIG["guidance_scale"],
    #             num_inference_steps = CONFIG["num_inference_steps"],
    #             generator = generator,
    #             height = 512,
    #             width = 512
    #         ).images[0]
    #     image.save(save_path)
    #     end_time = time.time()
    #     spend_time = end_time - start_time
    #     print(f"已生成图片保存于{save_path} 花费{spend_time:.3f}s")
    #     return [save_path,spend_time]

    def image2image(self, CONFIG: dict) -> list:
        start_time = time.time()
        generator = torch.Generator(device="cuda").manual_seed(CONFIG["seed"])
        save_path = CONFIG["save_path"]
        input_image = Image.open(CONFIG["image"]).convert("RGB")
        input_mask_image = Image.open(CONFIG["mask_image"]).convert("L")
        with torch.no_grad():
            image = self._inpaint_pipe(
                image = input_image,
                mask_image = input_mask_image,
                strength = CONFIG["strength"],
                prompt = CONFIG["prompt"],
                negative_prompt = CONFIG["negative_prompt"],
                guidance_scale = CONFIG["guidance_scale"],
                num_inference_steps = CONFIG["num_inference_steps"],
                generator = generator,
                height = 512,
                width = 512
            ).images[0]
        image.save(save_path)
        end_time = time.time()
        spend_time = end_time - start_time
        print(f"已生成图片保存于{save_path} 花费{spend_time:.3f}s")
        return [save_path,spend_time]

    def text2image(self, CONFIG: dict) -> list:
        CONFIG["image"] = "data/full_white.jpg"
        CONFIG["mask_image"] = "data/full_white.jpg"
        CONFIG["strength"] = 1
        return self.image2image(CONFIG)

    def set_LoRA_adapter_weights(self, adapter_weights: float):
        self._inpaint_pipe.set_adapters(["LoRA-adapter"],adapter_weights = [adapter_weights])


def get_GenerateModel(BASE_MODEL_PATH: str, LORA_WEIGHTS_PATH: str) -> _GenerateModel:
    return _GenerateModel(BASE_MODEL_PATH, LORA_WEIGHTS_PATH)

def get_text2image_config(
        OUTPUT_DIR: str,
        OUTPUT_FILENAME: str,
        LORA_PROMPT: str,
        PROMPT: str,
        NEGATIVE_PROMPT: str,
        NUM_INFERENCE_STEP: int,
        GUIDANCE_SCALE: float,
        SEED: int,
        ) -> dict:
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    input_config = {"save_path": save_path,
                    "prompt": PROMPT + LORA_PROMPT,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "num_inference_steps": NUM_INFERENCE_STEP,
                    "guidance_scale": GUIDANCE_SCALE,
                    "seed": SEED,
                    }
    return input_config

def get_image2image_config(
        OUTPUT_DIR: str,
        OUTPUT_FILENAME: str,
        IMAGE: str,
        MASK_IMAGE: str,
        LORA_PROMPT: str,
        PROMPT: str,
        NEGATIVE_PROMPT: str,
        NUM_INFERENCE_STEP: int,
        GUIDANCE_SCALE: float,
        SEED: int,
        STRENGTH: float
        ) -> dict:
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    input_config = {"save_path": save_path,
                    "image": IMAGE,
                    "mask_image": MASK_IMAGE,
                    "prompt": PROMPT + LORA_PROMPT,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "num_inference_steps": NUM_INFERENCE_STEP,
                    "guidance_scale": GUIDANCE_SCALE,
                    "seed": SEED,
                    "strength": STRENGTH
                    }
    return input_config

def get_lora_prompt(LORA_NAME: str) -> str:
    prompt_dict = parse_specimens_jsonl_file2dict(FILE_PATH ="data/specimens/lora_specimens.jsonl", KEY ="LORA", VALUE ="PROMPT")
    try:
        return prompt_dict[LORA_NAME]
    except KeyError:
        return ""
