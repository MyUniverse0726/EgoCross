import json
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI
import base64
import chat_vlms
import preference


MODEL_NAME = preference.GPT_MODEL_NAME
client = OpenAI(api_key=preference.GPT_API_KEY)

def encode_image(image_path):
    """Encode an image file as a Base64 data URI."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime_type};base64,{base64_str}"



def evaluate_answer(question, standard_answer, prediction):
    """Use an LLM to score the answer between 0 and 5 and provide a rationale."""

    if isinstance(standard_answer, int) and (prediction == standard_answer or str(standard_answer) in str(prediction)):
        score = 5
        reason = "Prediction matches the ground truth exactly"
        return score, None, ""
    elif isinstance(standard_answer, str) and standard_answer.lower() in prediction.lower():
        score = 5
        return score, None, ""

    # Fall back to LLM scoring when simple matching fails.
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
            "mark": 4,  # The score of the prediction
            "reason": "",  # The reason for the score
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
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                score = data.get("mark", 0)
                reason = data.get("reason", "")
                return score, reason, response
        except Exception as e:
            print(f"Failed to parse score: {e}")

    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0, None, ""



def evaluate_answer_with_detail(question, standard_answer, detailed_answer, prediction):
    """Use an LLM to score the answer with detailed reference between 0 and 5."""

    if isinstance(standard_answer, int) and (prediction == standard_answer or str(standard_answer) in str(prediction)):
        score = 5
        reason = "Prediction matches the ground truth exactly"
        return score, None, ""
    elif isinstance(standard_answer, str) and standard_answer.lower() in prediction.lower():
        score = 5
        return score, None, ""

    # Fall back to LLM scoring when simple matching fails.
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
            "mark": 4,  # The score of the prediction
            "reason": "",  # The reason for the score
        }
        ```
    """

    formatted_prompt = prompt.replace("{question}", question).replace("{answer}", str(standard_answer)).replace("{pred}", str(prediction)).replace("{detailed_answer}", str(detailed_answer))

    try:
        response = chat_vlms.basic_llm_chat(
            model=preference.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.01,
            top_p=0.9,
            max_tokens=512
        )

        try:
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                score = data.get("mark", 0)
                reason = data.get("reason", "")
                return score, reason, response
        except Exception as e:
            print(f"Failed to parse score: {e}")

    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0, None, ""



def parse_model_response(response: str):
    """
    Attempt to parse the model response and extract prediction and reason.
    """
    prediction = ""
    reason = ""

    try:
        response_json = json.loads(response)
        prediction = response_json.get("prediction", "")
        reason = response_json.get("reason", "")
    except Exception as pe:
        try:
            # Match the first JSON object inside the response.
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                prediction = response_json.get("prediction", "")
                reason = response_json.get("reason", "")
        except Exception as e:
            pass

    return prediction, reason



def run(json_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"ego_gpt41_results_{timestamp}.json")
    
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

        content = []
        for image_path in video_path:
            if not os.path.exists(image_path):
                continue
            data_uri = encode_image(image_path)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        content.append({"type": "text", "text": enhanced_question})
        messages = [{"role": "user", "content": content}]

        start_time = time.time()
        score = 0
        prediction = ""
        reason = ""
        for attempt_id in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    stream=False
                )
                response_text = response.choices[0].message.content
                prediction, _ = parse_model_response(response_text)
                score, reason, _ = evaluate_answer_with_detail(
                    question=question,
                    standard_answer=answer_text,
                    detailed_answer=detailed_answer,
                    prediction=prediction
                )
                break  # Stop retry loop on success
            except Exception as e:
                print(f"Attempt {attempt_id+1} failed: {e}")
                prediction = "[ERROR]"
                reason = str(e)
                wait_time = 5 * attempt_id
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

        if (idx + 1) % 1 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)


    correct_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    summary = {
        "total": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "model": "GPT-4.1",
        "timestamp": timestamp
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=4)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    json_file = "datasets/egocross_testbed_imgs.json"
    output_dir = "results-egocross-testbed/openset-GPT41"
    run(json_file, output_dir)
