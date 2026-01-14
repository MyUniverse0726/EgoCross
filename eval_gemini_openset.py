import json
import os
import re
import time
from tqdm import tqdm
import google.generativeai as genai
import chat_vlms
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
    """Upload image files to Gemini and return uploaded handles."""
    uploaded_files = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        uploaded_files.append(
            genai.upload_file(path=img_path, display_name=os.path.basename(img_path))
        )
    return uploaded_files


def evaluate_answer(question, standard_answer, prediction):
    """Use an LLM judge to score the prediction between 0 and 5."""

    if isinstance(standard_answer, int) and (prediction == standard_answer or str(standard_answer) in str(prediction)):
        score = 5
        reason = "Prediction matches the ground truth exactly"
        return score, None, ""
    if isinstance(standard_answer, str) and standard_answer.lower() in prediction.lower():
        score = 5
        return score, None, ""

    prompt = """
        You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task: 
        ##INSTRUCTIONS: 
        - Focus on the factual accuracy and semantic equivalence of the predicted answer with the ground-truth. 
        - Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that.
        Please evaluate the following video-based question-answer pair: 
        Question: {question} 
        Ground truth correct Answer: {answer} 
        Predicted Answer: {pred} 
        Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness. Please generate the response in the form of a json object with the following fields:
        ```
        {
            "mark": 4,
            "reason": "",
        }
        ```
    """

    formatted_prompt = prompt.replace("{question}", question).replace("{answer}", str(standard_answer)).replace("{pred}", str(prediction))

    try:
        response = chat_vlms.basic_llm_chat(
            model=preference.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.01,
            top_p=0.9,
            max_tokens=512
        )

        try:
            json_match = re.search(r"\{[\s\S]*?\}", response)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                score = data.get("mark", 0)
                reason = data.get("reason", "")
                return score, reason, response
        except Exception as parse_error:
            print(f"Failed to parse score: {parse_error}")

    except Exception as err:
        print(f"Evaluation error: {err}")
        return 0, None, ""

    return 0, None, ""


def evaluate_answer_with_detail(question, standard_answer, detailed_answer, prediction):
    """Use an LLM judge with detailed context to score the prediction."""

    if isinstance(standard_answer, int) and (prediction == standard_answer or str(standard_answer) in str(prediction)):
        score = 5
        reason = "Prediction matches the ground truth exactly"
        return score, None, ""
    if isinstance(standard_answer, str) and standard_answer.lower() in prediction.lower():
        score = 5
        return score, None, ""

    prompt = """
        You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. Your task is to compare the predicted answer with the ground-truth and determine if the predicted answer is correct or not. Here's how you can accomplish the task: 
        ##INSTRUCTIONS: 
        - Focus on the factual accuracy and semantic equivalence of the predicted answer with the ground-truth. 
        - Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that.
        Please evaluate the following video-based question-answer pair: 
        Question: {question} 
        Correct ground truth: {answer} 
        Detailed ground truth: {detailed_answer}
        Predicted Answer: {pred} 
        Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness. Please generate the response in the form of a json object with the following fields:
        ```
        {
            "mark": 4,
            "reason": "",
        }
        ```
    """

    formatted_prompt = (
        prompt.replace("{question}", question)
        .replace("{answer}", str(standard_answer))
        .replace("{pred}", str(prediction))
        .replace("{detailed_answer}", str(detailed_answer))
    )

    try:
        response = chat_vlms.basic_llm_chat(
            model=preference.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.01,
            top_p=0.9,
            max_tokens=512
        )

        try:
            json_match = re.search(r"\{[\s\S]*?\}", response)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                score = data.get("mark", 0)
                reason = data.get("reason", "")
                return score, reason, response
        except Exception as parse_error:
            print(f"Failed to parse score: {parse_error}")

    except Exception as err:
        print(f"Evaluation error: {err}")
        return 0, None, ""

    return 0, None, ""


def parse_model_response(response: str):
    """Extract prediction and reason fields from the raw model response."""
    prediction = ""
    reason = ""

    try:
        response_json = json.loads(response)
        prediction = response_json.get("prediction", "")
        reason = response_json.get("reason", "")
    except Exception:
        try:
            json_match = re.search(r"\{[\s\S]*?\}", response)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                prediction = response_json.get("prediction", "")
                reason = response_json.get("reason", "")
        except Exception:
            pass

    return prediction, reason


def run(json_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"ego_gemini2.5pro_results_{timestamp}.json")

    print(f"Loading dataset: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        ego_data = json.load(f)

    results = []
    for idx, item in enumerate(tqdm(ego_data, desc="Inference Progress")):
        qa_id = item["id"]
        dataset = item["dataset"]
        primary_category = item["primary_category"]
        question_type = item["question_type"]
        question_id = item["question_id"]
        original_video_fps = item["original_video_fps"]

        video_path = item["video_path"]
        question = item["question_text"]
        answer_text = item["answer_text"]
        detailed_answer = item["detailed_answer"]

        img_suffix = os.path.splitext(video_path[0])[1].lower()
        sampling_fps = 1.0 if dataset == "CholecTrack20" and img_suffix == ".png" else 0.5
        enhanced_question = (
            "Please carefully read the question and the provided context, then provide a clear and concise answer based on your understanding. "
            f"Question: {question}\n"
            f"The original FPS of the video is {original_video_fps}. This image set is obtained by sampling at {sampling_fps} fps."
            "Your answer should be reasoned and directly address the question. "
            "Respond in JSON format with two fields: 'prediction' (your answer as text) and 'reason' (a brief explanation of your reasoning). "
            "Do not include any other content.\n\n"
            "Example response:\n"
            "{\n"
            "    \"prediction\": \"The time duration of the video is 10 seconds.\",\n"
            "    \"reason\": \"The total number of frames is 240 and the frame rate is 24 FPS, so 240 / 24 = 10 seconds.\"\n"
            "}\n"
        )

        start_time = time.time()
        score = 0
        prediction = ""
        reason = ""
        for attempt_id in range(5):
            try:
                uploaded_files = upload_images(video_path)
                response = model.generate_content([
                    *uploaded_files,
                    enhanced_question
                ], generation_config=GENERATION_CONFIG)
                response_text = response.text
                prediction, _ = parse_model_response(response_text)
                score, reason, _ = evaluate_answer_with_detail(
                    question=question,
                    standard_answer=answer_text,
                    detailed_answer=detailed_answer,
                    prediction=prediction
                )
                break
            except Exception as e:
                print(f"Attempt {attempt_id+1} failed: {e}")
                prediction = "[ERROR]"
                reason = str(e)
                wait_time = 5 * attempt_id
                if wait_time > 0:
                    time.sleep(wait_time)

        end_time = time.time()
        success = score >= 4

        result_entry = {
            "id": qa_id,
            "dataset": dataset,
            "primary_category": primary_category,
            "question_type": question_type,
            "question_id": question_id,
            "video_path": video_path,
            "question": question,
            "answer_text": answer_text,
            "detailed_answer": detailed_answer,
            "prediction": prediction,
            "reason": reason,
            "inference_time": end_time - start_time,
            "success": success
        }

        results.append(result_entry)

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
    output_dir = "results-egocross-testbed/openset-Gemini2.5pro"
    run(json_file, output_dir)
