import json
import ollama
from evaluation.prompts import create_judge_prompt


INPUT_FILE ="C:/Users/sanoj/local_SLM_App_with_Ollama/src/data/final.json"
OUTPUT_FILE = "C:/Users/sanoj/local_SLM_App_with_Ollama/src/data/evaluated_result.json"
GROUND_TRUTH = "C:/Users/sanoj/local_SLM_App_with_Ollama/src/data/ground_truth.json"

#modell

JUDGE_MODEL ="qwen2.5:3b"

##load benchmark final results

with open(INPUT_FILE,"r",encoding="utf-8") as file:
    results = json.load(file)


##load ground truth 
with open(GROUND_TRUTH,"r",encoding="utf-8") as file:
    ground_truth_data = json.load(file)



#llm as a judge function 
# LLM as a judge function
def evaluate_respone(prompt, response, ground_truth):

    try:
        ##3Prompt
        judge_prompt = create_judge_prompt(
             prompt,
            response,
            ground_truth
        )

        # Call Ollama judge
        judge_response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": judge_prompt
                }
            ]
        )

        judge_output = judge_response["message"]["content"]

        # Print raw judge output for debugging
        print("\nJUDGE OUTPUT")
        print(judge_output)
        print("===========\n")

        # Remove Markdown if the model adds it
        judge_output = (
            judge_output
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        # Convert JSON  to dictionary
        scores = json.loads(judge_output)

        # Calculate quality score
        quality_score = round(
            (
                scores["correctness"] +
                scores["relevance"] +
                scores["conciseness"] +
                scores["instruction_following"]
            ) / 8,
            2
        )

        # Add quality score
        scores["quality_score"] = quality_score

        return scores

    except Exception as error:

        print("Judge error:", error)

        return {
            "correctness": None,
            "relevance": None,
            "conciseness": None,
            "instruction_following": None,
            "quality_score": None,
            "reason": "Judge failed"
        }


# Evaluate all results
evaluated_results = []

for number, item in enumerate(results, start=1):

    print(item["model_name"])

    # Get prompt
    prompt = item["prompt"]

    # Get model response
    response = item["response"]

    # Find ground truth
    ground_truth = None

    for gt_item in ground_truth_data:

        if gt_item["prompt"] == prompt:
            ground_truth = gt_item["ground_truth"]
            break

    if ground_truth is None:

        print("Ground truth not found for prompt:", prompt)
        continue

    # Evaluate response
    scores = evaluate_respone(
        prompt,
        response,
        ground_truth
    )

    # Copy original benchmark data
    evaluate_data = item.copy()

    # Add ground truth
    evaluate_data["ground_truth"] = ground_truth

    # Add evaluation scores
    evaluate_data["correctness"] = scores["correctness"]
    evaluate_data["relevance"] = scores["relevance"]
    evaluate_data["conciseness"] = scores["conciseness"]
    evaluate_data["instruction_following"] = scores["instruction_following"]
    evaluate_data["quality_score"] = scores["quality_score"]
    evaluate_data["judge_reason"] = scores["reason"]

    evaluated_results.append(evaluate_data)


# Save final evaluation
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        evaluated_results,
        file,
        indent=4,
        ensure_ascii=False
    )

print("Evaluation file saved successfully")
print("Total evaluated results:", len(evaluated_results))
