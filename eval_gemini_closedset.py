import json
import os
import re
import time
from tqdm import tqdm
import google.generativeai as genai
import preference


MODEL_NAME = preference.GEMINI_MODEL_NAME
genai.configure(api_key=preference.GEMINI_API_KEY)
model = genai.GenerativeModel(model_name=MODEL_NAME)

GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
}


def upload_images(image_paths):
    """Upload image files to Gemini and return the uploaded handles."""
    uploaded_files = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        uploaded_files.append(
            genai.upload_file(path=img_path, display_name=os.path.basename(img_path))
        )
    return uploaded_files


def parse_model_response(response: str, ground_truth: str):
    """Parse the model response and extract prediction/reason fields."""
    prediction = ""
    reason = ""
    success = False
    raw_response = response

    try:
        response_json = json.loads(response)
        prediction = response_json.get("prediction", "")
        reason = response_json.get("reason", "")
        success = (prediction.lower() == ground_truth.lower())

    except Exception:
        try:
            json_match = re.search(r"\{[\s\S]*?\}", response)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                prediction = response_json.get("prediction", "")
                reason = response_json.get("reason", "")
                success = (prediction.lower() == ground_truth.lower())
        except json.JSONDecodeError:
            pass
        except Exception:
            pass

    result = {
        "prediction": prediction,
        "reason": reason,
        "success": success
    }

    if not success:
        result["raw_response"] = raw_response

    return result


def run(json_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"ego_gemini2.5pro_results_{timestamp}.json")

    print(f"Loading dataset: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        ego_data = json.load(f)

    results = []
    for idx, item in enumerate(tqdm(ego_data, desc="Inference")):
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

        options_str = "\n\nOptions:\n" + "\n".join(options_text) if options_text else ""
        img_suffix = os.path.splitext(video_path[0])[1].lower()
        sampling_fps = 1.0 if dataset == "CholecTrack20" and img_suffix == ".png" else 0.5

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

        start_time = time.time()
        for attempt_id in range(5):
            try:
                uploaded_files = upload_images(video_path)
                response = model.generate_content([
                    *uploaded_files,
                    enhanced_question
                ], generation_config=GENERATION_CONFIG)
                response_text = response.text
                result = parse_model_response(response_text, ground_truth)
                break
            except Exception as e:
                print(f"Attempt {attempt_id+1} failed: {e}")
                wait_time = 5 * attempt_id
                if wait_time > 0:
                    time.sleep(wait_time)
                result = {"prediction": "[ERROR]", "reason": str(e), "success": False}

        end_time = time.time()

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

        if (idx + 1) % 5 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    correct_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    summary = {
        "total": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "model": "Gemini2.5pro",
        "timestamp": timestamp
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=4)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    json_file = "datasets/egocross_testbed_imgs.json"
    output_dir = "results-egocross-testbed/closedset-Gemini2.5pro"
    run(json_file, output_dir)
