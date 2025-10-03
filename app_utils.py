import os
import tempfile
import logging
import torch
from PIL import Image
import numpy as np
# import spaces # Optional, keeping if needed for HF Spaces but commenting out for local if not installed, or keeping if it is. 
# Assuming spaces is installed or handled. If not, I'll remove the decorator.
# For standard Streamlit, spaces decorator might not be needed unless deploying to ZeroGPU.
# I'll keep it imported if available, else mock it.

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

from torchvision.transforms.functional import to_pil_image
from utils_model import get_processor_model, move_to_device, process_image

from utils_attn import (
    handle_attentions_i2t, plot_attention_analysis, handle_relevancy, handle_text_relevancy, reset_tokens,
    plot_text_to_image_analysis, handle_box_reset, boxes_click_handler, attn_update_slider
)

from utils_relevancy import construct_relevancy_map


logger = logging.getLogger(__name__)

N_LAYERS = 32 
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROLE0 = "USER"
ROLE1 = "ASSISTANT"

processor = None
model = None

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class State:
    def __init__(self):
        self.messages = []
        self.prompt = ""
        self.prompt_len = 0
        self.image = None
        self.recovered_image = None
        self.input_text_tokenized = []
        self.output_ids_decoded = []
        self.attention_key = ""
        self.image_idx = 0
        self.enc_attn_weights = []
        self.enc_attn_weights_vit = []

def initialize_model(args):
    global model, processor, system_prompt, ROLE0, ROLE1
    if model is None:
        processor, model = get_processor_model(args)
    
    if 'gemma' in args.model_name_or_path:
        system_prompt = ''
        ROLE0 = 'user'
        ROLE1 = 'model'

def clear_history():
    state = State()
    return state

def add_text(state, text, image, image_process_mode):
    global processor, ROLE0, ROLE1, system_prompt
    
    if state is None:
        state = State()
        
    if isinstance(image, dict):
        # Handle composite image if passed as dict (unlikely in Streamlit unless using specific component)
        image = image.get('composite', image)
    
    if isinstance(image, Image.Image):
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
    
    # Check if image is blank/white (simple check)
    if image is not None:
        np_img = np.array(image)
        if (np_img == 255).all():
            image = None

    text = text[:1536]  # Hard cut-off
    logger.info(text)

    prompt_len = 0
    
    if processor.tokenizer.chat_template is not None:
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<image>\n" + text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len += len(prompt)
    else:
        prompt = system_prompt
        prompt_len += len(prompt)
        if image is not None:
            msg = f"\n{ROLE0}: <image>\n{text}\n{ROLE1}:" 
        else:
            msg = f"\n{ROLE0}: {text}\n{ROLE1}: "
        prompt += msg
        prompt_len += len(msg)

    state.messages.append({"role": ROLE0, "content": text, "image": image})
    # state.messages.append({"role": ROLE1, "content": None}) # Placeholder for assistant response

    state.prompt_len = prompt_len
    state.prompt = prompt
    state.image = process_image(image, image_process_mode, return_pil=True)

    return state

@spaces.GPU
def lvlm_bot(state, temperature, top_p, max_new_tokens):   
    global model, processor
    prompt = state.prompt
    image = state.image
    
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Find image token index
    if hasattr(model.config, 'image_token_index'):
        img_token_idx = model.config.image_token_index
    else:
        # Fallback or check processor
        img_token_idx = 32000 # Default for some models, but safer to look up
        # Actually llava-gemma might be different. 
        # The original code used model.config.image_token_index
        pass
        
    try:
        img_idx = torch.where(input_ids == model.config.image_token_index)[1][0].item()
    except:
        # If image token not found or config missing
        img_idx = 0 

    do_sample = True if temperature > 0.001 else False
    
    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []

    if hasattr(model.language_model.config, 'model_type') and model.language_model.config.model_type == "gemma":
        eos_token_id = processor.tokenizer('<end_of_turn>', add_special_tokens=False).input_ids[0]
    else:
        eos_token_id = processor.tokenizer.eos_token_id

    outputs = model.generate(
            **inputs, 
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=eos_token_id
        )

    input_ids_list = input_ids.reshape(-1).tolist()
    if img_idx < len(input_ids_list):
        input_ids_list[img_idx] = 0 # Replace image token with 0 for decoding
    
    input_text = processor.tokenizer.decode(input_ids_list) 
    if input_text.startswith("<s> "):
        input_text = '<s>' + input_text[4:] 
    
    input_text_tokenized = processor.tokenizer.tokenize(input_text)
    if img_idx < len(input_text_tokenized):
        input_text_tokenized[img_idx] = "average_image"
    
    output_ids = outputs.sequences.reshape(-1)[input_ids.shape[-1]:].tolist()  

    generated_text = processor.tokenizer.decode(output_ids)
    output_ids_decoded = [processor.tokenizer.decode(oid).strip() for oid in output_ids] 
    generated_text_tokenized = processor.tokenizer.tokenize(generated_text)

    logger.info(f"Generated response: {generated_text}")

    # Update state messages
    # state.messages[-1]["content"] = generated_text
    state.messages.append({"role": ROLE1, "content": generated_text})

    tempdir = os.getenv('TMPDIR', '/tmp/')
    tempfilename = tempfile.NamedTemporaryFile(dir=tempdir, delete=False)
    tempfilename.close()

    # Save input_ids and attentions
    fn_input_ids = f'{tempfilename.name}_input_ids.pt'
    torch.save(move_to_device(input_ids, device='cpu'), fn_input_ids)
    fn_attention = f'{tempfilename.name}_attn.pt'
    torch.save(move_to_device(outputs.attentions, device='cpu'), fn_attention)
    logger.info(f"Saved attention to {fn_attention}")

    # Handle relevancy map
    word_rel_map = construct_relevancy_map(
        tokenizer=processor.tokenizer, 
        model=model,
        input_ids=inputs.input_ids,
        tokens=generated_text_tokenized, 
        outputs=outputs, 
        output_ids=output_ids,
        img_idx=img_idx
    )
    fn_relevancy = f'{tempfilename.name}_relevancy.pt'
    torch.save(move_to_device(word_rel_map, device='cpu'), fn_relevancy)
    logger.info(f"Saved relevancy map to {fn_relevancy}")
    
    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []

    # Reconstruct processed image
    img_std = torch.tensor(processor.image_processor.image_std).view(3,1,1)
    img_mean = torch.tensor(processor.image_processor.image_mean).view(3,1,1)
    img_recover = inputs.pixel_values[0].cpu() * img_std + img_mean
    img_recover = to_pil_image(img_recover)

    state.recovered_image = img_recover
    state.input_text_tokenized = input_text_tokenized
    state.output_ids_decoded = output_ids_decoded 
    state.attention_key = tempfilename.name
    state.image_idx = img_idx

    return state
