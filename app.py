# -*- coding: utf-8 -*-
import gradio as gr
import model
import tool

GENERATE_MODEL: model

def init() -> list:
    item_list = []
    base_model_list = tool.get_file_list("data/diffusion_models")
    lora_weights_list = tool.get_file_list("data/loras")
    item_list.append(base_model_list)
    item_list.append(lora_weights_list)
    return item_list

def mian():
    INIT_LIST = init()
    BASE_PATH_MODEL: str
    LORA_PATH_MODEL: str
    LORA_PROMPT: str
    try:
        global GENERATE_MODEL
        BASE_PATH_MODEL = tool.get_save_path("data/diffusion_models",INIT_LIST[0][0])
        LORA_PATH_MODEL = tool.get_save_path("data/loras",INIT_LIST[1][0])
        LORA_PROMPT = model.get_lora_prompt(INIT_LIST[1][0])
        GENERATE_MODEL = model.get_GenerateModel(BASE_PATH_MODEL, LORA_PATH_MODEL)
        gr.Info(f"模型加载完成 花费{GENERATE_MODEL.get_load_model_spend_time()}")
    except IndexError:
        LORA_PROMPT = ""
        gr.Warning(label = "模型文件缺失，请检查文件路径")


    with gr.Blocks(title = "AutoSketch") as main_page:
        loading_model = gr.Textbox(label = "已加载模型",value = f"{INIT_LIST[0][0]} + {INIT_LIST[1][0]}",interactive = False)
        with gr.Tab(label = "文本生成"):
            with gr.Column(visible = True) as text2image_page:
                with gr.Row():
                    text2image_prompt_input = gr.Textbox(label = "正向提示词",lines = 2)
                    text2image_negative_prompt_input = gr.Textbox(label = "反向提示词",lines = 2)
                with gr.Row(equal_height = True):
                    with gr.Column():
                        def text2image_sumit_fn(text2image_prompt,text2image_negative_prompt,text2image_num_inference_steps,text2image_guidance_scale,text2image_seed):
                            global GENERATE_MODEL
                            print("LORA_PROMPT: " + LORA_PROMPT)
                            text2image_config = model.get_text2image_config(
                                OUTPUT_DIR = "output/image2image",
                                OUTPUT_FILENAME = tool.get_current_time_str() + ".jpg",
                                LORA_PROMPT = LORA_PROMPT,
                                PROMPT = text2image_prompt,
                                NEGATIVE_PROMPT = text2image_negative_prompt,
                                NUM_INFERENCE_STEP = text2image_num_inference_steps,
                                GUIDANCE_SCALE = text2image_guidance_scale,
                                SEED = text2image_seed
                            )
                            text2image_save_path = GENERATE_MODEL.text2image(text2image_config)
                            gr.Info(f"生成成功，保存路径为：{text2image_save_path[0]}")
                            return text2image_save_path[0]
                        text2image_num_inference_steps_input = gr.Number(label = "推理步数",interactive = True,value = 30)
                        text2image_guidance_scale_input = gr.Number(label = "CFG",interactive = True,value = 7.5)
                        text2image_seed_input = gr.Number(label = "随机种子",interactive = True,value = 42)
                        text2image_sumit = gr.Button(value = "生成图像",variant = "primary")

                    with gr.Column():
                        text2image_output = gr.Image(label = "输出图像",interactive = False)
                    text2image_sumit.click(fn = text2image_sumit_fn,inputs = [
                        text2image_prompt_input,
                        text2image_negative_prompt_input,
                        text2image_num_inference_steps_input,
                        text2image_guidance_scale_input,
                        text2image_seed_input
                    ],outputs = [text2image_output],queue = True)
        with gr.Tab(label = "绘画生成"):
            with gr.Column(visible = True) as image2image_paint_page:
                def image2image_paint_sumit_fn(image2image_paint_keep_paint,image2image_paint_prompt,image2image_paint_negative_prompt,image2image_paint,image2image_paint_num_inference_step,image2image_paint_guidance_scale,image2image_paint_strength,image2image_paint_seed):
                    global GENERATE_MODEL
                    save_path_temp = "output/temp_paint.jpg"
                    tool.get_numpy_image(image2image_paint["composite"]).save(save_path_temp)
                    if image2image_paint_keep_paint:
                        mask_image_path = save_path_temp
                    else:
                        mask_image_path = "data/full_white.jpg"
                    image2image_paint_config = model.get_image2image_config(
                        OUTPUT_DIR = "output/image2image",
                        OUTPUT_FILENAME = tool.get_current_time_str() + ".jpg",
                        IMAGE = save_path_temp,
                        MASK_IMAGE = mask_image_path,
                        LORA_PROMPT = LORA_PROMPT,
                        PROMPT = image2image_paint_prompt,
                        NEGATIVE_PROMPT = image2image_paint_negative_prompt,
                        NUM_INFERENCE_STEP = image2image_paint_num_inference_step,
                        GUIDANCE_SCALE = image2image_paint_guidance_scale,
                        SEED = image2image_paint_seed,
                        STRENGTH = image2image_paint_strength
                    )
                    image2image_save_path = GENERATE_MODEL.image2image(image2image_paint_config)
                    tool.remove_file_path(save_path_temp)
                    gr.Info(f"生成成功，保存路径为：{image2image_save_path[0]}")
                    return image2image_save_path[0]
                with gr.Row():
                    image2image_paint_prompt_input = gr.Textbox(label = "正向提示词",lines = 2)
                    image2image_paint_negative_prompt_input = gr.Textbox(label = "反向提示词",lines = 2)
                with gr.Row(equal_height = True):
                    image2image_paint_input = gr.Sketchpad(label = "输入图像",interactive = True,width = 512,height = 512)
                    image2image_paint_output = gr.Image(label = "输出图像",interactive = False)
                with gr.Row(equal_height = True):
                    image2image_paint_keep_paint_input = gr.Checkbox(label = "保留绘画",value = True)
                    image2image_paint_num_inference_steps_input = gr.Number(label="推理步数", interactive=True, value=30)
                    image2image_paint_guidance_scale_input = gr.Number(label="CFG", interactive=True, value=7.5)
                    image2image_paint_strength_input = gr.Number(label="生成强度", interactive=True, value=0.8)
                    image2image_paint_seed_input = gr.Number(label="随机种子", interactive=True, value=42)
                    image2image_paint_sumit = gr.Button(value="生成图像", variant="primary")
                image2image_paint_sumit.click(fn = image2image_paint_sumit_fn,inputs = [
                    image2image_paint_keep_paint_input,
                    image2image_paint_prompt_input,
                    image2image_paint_negative_prompt_input,
                    image2image_paint_input,
                    image2image_paint_num_inference_steps_input,
                    image2image_paint_guidance_scale_input,
                    image2image_paint_strength_input,
                    image2image_paint_seed_input
                ],outputs = [image2image_paint_output],queue = True)
        with gr.Tab(label = "图像补全"):
            with gr.Column(visible = True) as image2image_complete_page:
                def image2image_complete_sumit_fn(image2image_complete_keep_paint,image2image_complete_prompt,image2image_complete_negative_prompt,image2image_complete,image2image_complete_num_inference_step,image2image_complete_guidance_scale,image2image_complete_strength,image2image_complete_seed):
                    global GENERATE_MODEL
                    if image2image_complete_keep_paint:
                        mask_image_path = image2image_complete
                    else:
                        mask_image_path = "data/full_white.jpg"
                    image2image_paint_config = model.get_image2image_config(
                        OUTPUT_DIR = "output/image2image",
                        OUTPUT_FILENAME = tool.get_current_time_str() + ".jpg",
                        IMAGE = image2image_complete,
                        MASK_IMAGE = mask_image_path,
                        LORA_PROMPT = LORA_PROMPT,
                        PROMPT = image2image_complete_prompt,
                        NEGATIVE_PROMPT = image2image_complete_negative_prompt,
                        NUM_INFERENCE_STEP = image2image_complete_num_inference_step,
                        GUIDANCE_SCALE = image2image_complete_guidance_scale,
                        SEED = image2image_complete_seed,
                        STRENGTH = image2image_complete_strength
                    )
                    image2image_save_path = GENERATE_MODEL.image2image(image2image_paint_config)
                    gr.Info(f"生成成功，保存路径为：{image2image_save_path[0]}")
                    return image2image_save_path[0]
                with gr.Row():
                    image2image_complete_prompt_input = gr.Textbox(label = "正向提示词",lines = 2)
                    image2image_complete_negative_prompt_input = gr.Textbox(label = "反向提示词",lines = 2)
                with gr.Row(equal_height = True):
                    image2image_complete_input = gr.Image(label = "输入图像",interactive = True,width = 512,height = 512,type="filepath")
                    image2image_complete_output = gr.Image(label = "输出图像",interactive = False)
                with gr.Row(equal_height = True):
                    image2image_complete_keep_paint_input = gr.Checkbox(label = "保留绘画",value = True)
                    image2image_complete_num_inference_steps_input = gr.Number(label="推理步数", interactive=True, value=30)
                    image2image_complete_guidance_scale_input = gr.Number(label="CFG", interactive=True, value=7.5)
                    image2image_complete_strength_input = gr.Number(label="生成强度", interactive=True, value=0.8)
                    image2image_complete_seed_input = gr.Number(label="随机种子", interactive=True, value=42)
                    image2image_complete_sumit = gr.Button(value="生成图像", variant="primary")
                image2image_complete_sumit.click(fn = image2image_complete_sumit_fn,inputs = [
                    image2image_complete_keep_paint_input,
                    image2image_complete_prompt_input,
                    image2image_complete_negative_prompt_input,
                    image2image_complete_input,
                    image2image_complete_num_inference_steps_input,
                    image2image_complete_guidance_scale_input,
                    image2image_complete_strength_input,
                    image2image_complete_seed_input
                ],outputs = [image2image_complete_output],queue = True)
        with gr.Tab(label = "系统设置"):
            with gr.Column(visible = True) as setting_page:
                with gr.Row(equal_height = True):
                    def loading_model_fn(base_model_input, lora_model_input):
                        global GENERATE_MODEL
                        base_model_path = tool.get_save_path("data/diffusion_models", base_model_input)
                        lora_model_path = tool.get_save_path("data/loras", lora_model_input)
                        GENERATE_MODEL = model.get_GenerateModel(base_model_path,lora_model_path)
                        gr.Info(f"{base_model_input} + {lora_model_input} 加载成功")
                        return gr.Textbox(value = f"{base_model_input} + {lora_model_input}")
                    base_model_setting = gr.Dropdown(label = "基础模型",choices = INIT_LIST[0],value = INIT_LIST[0][0],interactive = True)
                    lora_setting = gr.Dropdown(label = "LORA模型",choices = INIT_LIST[1],value = INIT_LIST[1][0],interactive = True)
                    loading_model_sumit = gr.Button(value = "加载模型",variant = "primary")
                    loading_model_sumit.click(fn = loading_model_fn,inputs = [base_model_setting,lora_setting],outputs = [loading_model],show_progress = "full",queue = True)
    main_page.launch(theme = gr.themes.Soft(),server_port = 7860)

if __name__ == '__main__':
    mian()
