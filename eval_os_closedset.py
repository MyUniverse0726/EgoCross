# open-source MLLMs closed evaluation
import json
import os
import re
import time
import torch
import argparse
from tqdm import tqdm
import preference
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor


def setup_model(model_name='qwen2vl', use_8bit=False, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
    """Load the model and corresponding processor/tokenizer."""
    model_config = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "load_in_8bit": use_8bit,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    processor_config = {
        "trust_remote_code": True,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "use_fast": True
    }
    if model_name in ["qwen25vl"]:
        used_VLM_Model = preference.VLM_MODEL_NAME_QWEN25VL
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(used_VLM_Model, **model_config)
        processor = AutoProcessor.from_pretrained(used_VLM_Model, **processor_config)
    elif model_name in ["qwen25vl_3b"]:
        used_VLM_Model = preference.VLM_MODEL_NAME_QWEN25VL_3B
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(used_VLM_Model, **model_config)
        processor = AutoProcessor.from_pretrained(used_VLM_Model, **processor_config)
    elif model_name in ["qwen2vl"]:
        used_VLM_Model = preference.VLM_MODEL_NAME_QWEN2VL
        vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(used_VLM_Model, **model_config)
        processor = AutoProcessor.from_pretrained(used_VLM_Model, **processor_config)
    elif model_name in ["videollama3"]:
        used_VLM_Model = preference.VLM_MODEL_NAME_VIDEOLLAMA3
        vlm_model = AutoModelForCausalLM.from_pretrained(used_VLM_Model, **model_config)
        processor = AutoProcessor.from_pretrained(used_VLM_Model, **processor_config)
    elif model_name in ["internvl25"]:
        used_VLM_Model = preference.VLM_MODEL_NAME_INTERNVL25
        vlm_model = AutoModel.from_pretrained(used_VLM_Model, **model_config).eval()
        tokenizer = AutoTokenizer.from_pretrained(used_VLM_Model, trust_remote_code=True, use_fast=False)
        return vlm_model, tokenizer
    elif model_name in ["internvl3"]:
        used_VLM_Model = preference.VLM_MODEL_NAME_INTERNVL3
        vlm_model = AutoModel.from_pretrained(used_VLM_Model, **model_config).eval()
        tokenizer = AutoTokenizer.from_pretrained(used_VLM_Model, trust_remote_code=True, use_fast=False)
        return vlm_model, tokenizer
    else:
        raise ValueError(f"Unsupported model: {model_name}")
            
    return vlm_model, processor


def inference_qwen(vlm_model, processor, video_path, question, fps=1, max_pixels=360 * 480):
    """Run Qwen-based inference for a video QA sample."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": max_pixels,
                    "fps": fps,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Prepare prompt and inputs for generation.
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Generate the model response.
    generated_ids = vlm_model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False
    )

    # Decode the generated tokens.
    input_length = inputs.input_ids.shape[1]
    generated_ids_trimmed = [
        out_ids[input_length:] for out_ids in generated_ids
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text


def inference_videollama(vlm_model, processor, video_path, question, fps=1, max_pixels=360 * 480):
    # VideoLLaMA3 uses slightly different message schemas for images and videos.
    content_items = []

    if isinstance(video_path, list):
        for img_path in video_path:
            content_items.append({"type": "image", "image":{"image_path": img_path}})
    else:
        content_items.append({
            "type": "video",
            "video": {
                "video_path": video_path,
                "fps": fps,
                "max_frames": 180
            }
        })
    
    content_items.append({"type": "text", "text": question})
    
    messages_videollama3 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": content_items
        }
    ]

    inputs = processor(
        conversation=messages_videollama3,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        output_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return response


import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Utility methods for multi-turn video conversations.
def get_index(bound, original_fps, max_frame, first_idx=0, num_segments=32, target_fps=None):
    # Determine the temporal window.
    if bound:
        start, end = bound
    else:
        start, end = -100000, 100000  # cover the entire video by default
    
    # Convert time range to frame indices.
    start_idx = max(first_idx, round(start * original_fps))
    end_idx = min(round(end * original_fps), max_frame)
    
    # Adjust the number of segments based on the target FPS if provided.
    if target_fps is not None and target_fps > 0:
        duration_seconds = (end_idx - start_idx) / original_fps
        num_segments = int(duration_seconds * target_fps)
        num_segments = max(num_segments, 1)
    
    # Sample frame indices uniformly.
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    
    frame_indices = np.clip(frame_indices, 0, max_frame)
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, target_fps=None):
    # Handle either a list of images or a video file path.
    if isinstance(video_path, list):
        img_paths = video_path
        max_frame = len(img_paths) - 1
        original_fps = 1.0 
        vr = None 
        frame_indices = get_index(bound, original_fps, max_frame, first_idx=0, num_segments=num_segments, target_fps=original_fps)

    else:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        original_fps = float(vr.get_avg_fps())
    
        frame_indices = get_index(
            bound, 
            original_fps,
            max_frame, 
            first_idx=0, 
            num_segments=num_segments, 
            target_fps=target_fps
        )
        
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for frame_index in frame_indices:
        if isinstance(video_path, list):
            img_path = video_path[frame_index]
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# InternVL2.5 / InternVL3 inference.
def inference_internvl(vlm_model, tokenizer, video_path, question, max_tokens=1024, fps=1, max_pixels=360 * 480):
    messages = {
        "video": video_path,
        "max_pixels": max_pixels,
        "fps": fps,
        "text": question
    }
    pixel_values, num_patches_list = load_video(messages["video"], input_size=448, target_fps=messages["fps"])
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    generation_config = dict(
        max_new_tokens=max_tokens, 
        do_sample=False
    )
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + messages["text"]
    response, history = vlm_model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)

    return response

 

def parse_model_response(response: str, ground_truth: str):
    """
    Parse the raw model response and extract prediction and reason fields.

    Returns a dictionary containing the extracted fields, parsing status, and
    the raw output when parsing fails.
    """
    prediction = ""
    reason = ""
    success = False
    parsed_status = "unknown"
    raw_response = response  # keep original response for debugging

    try:
        response_json = json.loads(response)
        prediction = response_json.get("prediction", "")
        reason = response_json.get("reason", "")
        success = (prediction.lower() == ground_truth.lower())
        parsed_status = "primary_success"

    except Exception as pe:
        # Fall back to regex-based JSON extraction.
        parsed_status = f"primary_error: {str(pe)}"

        try:
            # Extract the outermost JSON block.
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                prediction = response_json.get("prediction", "")
                reason = response_json.get("reason", "")
                success = (prediction.lower() == ground_truth.lower())
                parsed_status = "secondary_success"

        except json.JSONDecodeError as je:
            parsed_status = f"{parsed_status}, secondary_error: {str(je)}"
        except Exception as e:
            parsed_status = f"{parsed_status}, secondary_error: {str(e)}"

    result = {
        "prediction": prediction,
        "reason": reason,
        "success": success,
        "parsed_status": parsed_status
    }

    # Preserve the raw response whenever parsing fails.
    if not success or "error" in parsed_status:
        result["raw_response"] = raw_response

    return result


def run(args):
    """Run evaluation over the dataset."""
    print(f"Loading model {args.model_name}")
    
    model, processor = setup_model(model_name=args.model_name, use_8bit=args.use_8bit, min_pixels=args.min_pixels, max_pixels=args.max_model_pixels)

    print(f"Reading dataset: {args.dataset_path}")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        ego_data = json.load(f)

    # Apply limit if requested.
    if args.limit > 0:
        ego_data = ego_data[:args.limit]

    # Prepare output directory and file.
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"ego_{args.model_name}_results_{timestamp}.json")

    results = []

    # Iterate through dataset items.
    for idx, item in enumerate(tqdm(ego_data, desc="Inference progress")):
        torch.cuda.empty_cache()
        # Gather metadata.
        qa_id = item["id"]
        dataset = item["dataset"]
        primary_category = item["primary_category"]
        question_type = item["question_type"]
        question_id = item["question_id"]
        original_video_fps = item["original_video_fps"]

        video_path = item["video_path"]      
        question = item["question_text"]
        options_text = item["options"]
        ground_truth = item["correct_option_letter"]
        answer_text = item["answer_text"]
        
        # Determine the sampling rate.
        img_suffix = os.path.splitext(video_path[0])[1].lower()
        default_sampling = args.fps if args.fps > 0 else 0.5 # videos were sampled at 0.5 fps in egocross_testbed to be frames list
        sampling_fps = 1.0 if dataset == "CholecTrack20" and img_suffix == ".png" else default_sampling # part of CholecTrack20 frames and whole EgoSurgery were sampled at 1.0 fps by original authors

        # Construct the question prompt.
        options_str = "\n\nOptions:\n" + "\n".join(options_text) if options_text else ""
        enhanced_question = (
            "Please carefully read the question and its options, then select the most appropriate answer. "
            f"Question: {question}{options_str}"
            f"The original FPS of the video is {original_video_fps}. This image set is obtained by sampling at {sampling_fps} fps."
            "Respond in JSON format with two fields: 'prediction' (the correct option letter: A, B, C, or D) and 'reason' (a brief explanation of your choice). "
            "Do not include any other content.\n\n"
            "Example response:\n"
            "{\n"
            "    \"prediction\": \"B\",\n"
            "    \"reason\": \"Paris is the capital city of France.\"\n"
            "}\n"
        )
  
        try:
            # Run inference.
            start_time = time.time()
            if args.model_name in ['qwen2vl', 'qwen25vl', 'qwen25vl_3b']:
                response = inference_qwen(vlm_model=model, processor=processor, video_path=video_path, question=enhanced_question, fps=args.fps, max_pixels=args.max_inference_pixels)
            elif args.model_name in ['videollama3']:
                response = inference_videollama(vlm_model=model, processor=processor, video_path=video_path, question=enhanced_question, fps=args.fps, max_pixels=args.max_inference_pixels)
            elif args.model_name in ['internvl25', 'internvl3']:
                response = inference_internvl(vlm_model=model, tokenizer=processor, video_path=video_path, question=enhanced_question, fps=args.fps, max_pixels=args.max_inference_pixels)
            else:
                raise ValueError(f"Unsupported model: {args.model_name}")
            end_time = time.time()
 
            # Parse model output.
            result = parse_model_response(response, ground_truth=ground_truth)
            # Build result entry.
            result_entry = {
                "id": qa_id,
                "dataset": dataset,
                "primary_category": primary_category,
                "question_type": question_type,
                "question_id": question_id,
                "video_path": video_path,
                "question": question,
                "options": options_text,
                "correct_option_letter": ground_truth,
                "answer_text": answer_text,
                "prediction": result["prediction"],
                "reason": result["reason"],
                "inference_time": end_time - start_time,
                "success": result["success"]
            }
            if "raw_response" in result:
                result_entry["raw_response"] = result["raw_response"]
            
            results.append(result_entry)

            # Save intermediate results at the requested interval.
            if args.save_interval > 0 and (idx + 1) % args.save_interval == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append({
                "id": qa_id,
                "dataset": dataset,
                "primary_category": primary_category,
                "question_type": question_type,
                "question_id": question_id,
                "video_path": video_path,
                "question": question,
                "options": options_text,
                "correct_option_letter": ground_truth,
                "answer_text": answer_text,
                "prediction": f"[FATAL_ERROR] {str(e)}",
                "raw_response": response if 'response' in locals() else None,
                "success": False
            })
            # Persist results after failure to avoid data loss.
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    correct_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    summary = {
        "total": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "model": args.model_name,
        "timestamp": timestamp
    }

    print("\nSummary:")
    print(f"Total questions: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=4)

    print(f"Results saved to: {output_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VLMs on EgoCross QA benchmarks.")

    # Model configuration.
    parser.add_argument("--model_name", type=str, default="qwen2vl", choices=["qwen25vl_3b", "qwen25vl", "internvl25", "internvl3", "videollama3"], help="Model identifier.") 
    parser.add_argument("--use_8bit", action="store_true", default=False, help="Enable 8-bit quantization when supported.")
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28, help="Minimum pixel budget for preprocessing.")
    parser.add_argument("--max_model_pixels", type=int, default=1280 * 28 * 28, help="Maximum pixel budget for model inputs.")
    parser.add_argument("--max_inference_pixels", type=int, default=480 * 360, help="Cap for inference pixel count.")
    # Dataset parameters.
    parser.add_argument("--dataset_path", type=str, default="datasets/egocross_testbed_imgs.json", help="Path to the evaluation JSON file.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N samples when > 0.")
    # Inference parameters.
    parser.add_argument("--fps", type=float, default=0.5, help="Frame sampling rate for video inputs.")
    # Output parameters.
    parser.add_argument("--output_dir", type=str, default="results-egocross-testbed/closedset-Qwen25VL", help="Directory for evaluation outputs.")
    parser.add_argument("--save_interval", type=int, default=5, help="Save partial results every N samples when > 0.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
