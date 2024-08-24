from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw
import torch
import os
import json
import matplotlib.pyplot as plt
import random
import numpy as np
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import requests

# Workaround for unnecessary flash_attn requirement
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    original_get_imports = get_imports
    if not str(filename).endswith("modeling_florence2.py"):
        return original_get_imports(filename)
    imports = original_get_imports(filename)
    imports = [imp for imp in imports if imp != "flash_attn"]
    return imports


def modify_config_for_davit(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Ensure that the vision_config model_type is set to 'davit'
    if 'vision_config' in config and config['vision_config'].get('model_type', '') != 'davit':
        config['vision_config']['model_type'] = 'davit'

    # Save the modified config back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def download_and_save_model(save_directory):
    model_id = 'microsoft/Florence-2-large'

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Download and save the model with the workaround
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Save the model, processor, and config
    processor.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    config_path = os.path.join(save_directory, 'config.json')
    modify_config_for_davit(config_path)

    return model, processor


def load_model(save_directory):
    config_path = os.path.join(save_directory, 'config.json')
    modify_config_for_davit(config_path)

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        # Load the model and processor from the saved directory with the workaround
        model = AutoModelForCausalLM.from_pretrained(save_directory, trust_remote_code=True,
                                                     torch_dtype='auto').eval().cuda()
        processor = AutoProcessor.from_pretrained(save_directory, trust_remote_code=True)
    return model, processor


# Helper function to run the model inference
def run_example(model, processor, image, task_prompt, text_input=None):
    prompt = task_prompt + (text_input or "")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


# Function to draw polygons on the image for segmentation
def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
                'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

    # Safely access the 'labels' key, defaulting to empty labels if not present
    labels = prediction.get('labels', [''] * len(prediction['polygons']))

    # Correctly access the 'polygons' key from the prediction
    for polygons, label in zip(prediction['polygons'], labels):  # Use lowercase 'polygons'
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                continue

            _polygon = _polygon.reshape(-1).tolist()

            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Only draw the label if it's not empty
            if label:
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image


# Function to plot bounding boxes on the image
def plot_bbox(image, data):
    draw = ImageDraw.Draw(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")
    return image


# Function to convert results to object detection format
def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])

    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results


# Main function to perform the tasks on the image
def process_image(image, actions=None, text_input=None):
    save_directory = 'ckpt/microsoft_Florence-2-large'

    if not os.path.exists(save_directory):
        model, processor = download_and_save_model(save_directory)
    else:
        model, processor = load_model(save_directory)

    if actions is None:
        actions = ['referring_expression_segmentation', 'od', 'dense_region_caption', 'region_proposal',
                   'caption_to_phrase_grounding', 'open_vocabulary_detection', 'caption']

    results = {}

    for action in actions:
        task_prompt = f'<{action.upper()}>'

        if action in ['caption_to_phrase_grounding', 'open_vocabulary_detection'] and text_input:
            action_results = run_example(model, processor, image, task_prompt, text_input)
        else:
            action_results = run_example(model, processor, image, task_prompt)

        output_image = image.copy()

        if action == 'referring_expression_segmentation':
            task_key = '<REFERRING_EXPRESSION_SEGMENTATION>'
            print(f"Output for {task_key}: {action_results[task_key]}")  # Debugging line
            output_image = draw_polygons(output_image, action_results[task_key], fill_mask=True)
        elif action in ['od', 'dense_region_caption', 'region_proposal', 'caption_to_phrase_grounding']:
            task_key = f'<{action.upper()}>'
            output_image = plot_bbox(output_image, action_results[task_key])
        elif action == 'open_vocabulary_detection':
            task_key = '<OPEN_VOCABULARY_DETECTION>'
            if 'polygons' in action_results[task_key]:  # Handle segmentation results
                output_image = draw_polygons(output_image, action_results[task_key], fill_mask=True)
            else:  # Handle bounding box results
                bbox_results = convert_to_od_format(action_results[task_key])
                output_image = plot_bbox(output_image, bbox_results)
        elif action == 'caption':
            task_key = '<CAPTION>'
            text_input = action_results[task_key]
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            action_results = run_example(model, processor, image, task_prompt, text_input)
            action_results['<CAPTION>'] = text_input
            output_image = plot_bbox(output_image, action_results['<CAPTION_TO_PHRASE_GROUNDING>'])

        results[action] = {'result': action_results, 'output_image': output_image}

    return results, image


# Function to display all results together in a montage
def display_montage(results, montage_size=(10, 10)):
    actions = list(results.keys())
    num_actions = len(actions)
    montage_cols = 2
    montage_rows = (num_actions + 1) // montage_cols

    fig, axes = plt.subplots(montage_rows, montage_cols, figsize=montage_size)
    axes = axes.flatten()

    for i, action in enumerate(actions):
        axes[i].imshow(results[action]['output_image'])
        axes[i].set_title(action.replace('_', ' ').title())
        axes[i].axis('off')

    # Remove unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

image_path = "test_buildings.jpg"  # Use the provided image path
image = Image.open(image_path)

# Example usage:
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

# Perform all actions on the image, including the text input for relevant tasks
results, original_img = process_image(image, text_input='buildings')

# Display the results in a montage
display_montage(results)
