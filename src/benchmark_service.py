import json
import traceback
from pathlib import Path


from metrics_services import (
    calculate_latency,
    count_tokens,
    calculate_tokens_per_second,
    memory_usage
)

from ollama_models import ask_all_model


# Load prompt.json
def load_prompt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        traceback.print_exc()
        return []



def benchmark_run(prompt):

    results = []

    try:

        

        for model_result in ask_all_model(prompt):


            memory_before = memory_usage()


            model_name = model_result["model_name"]

            response = model_result["response"]

            start_time = model_result["start_time"]

            end_time = model_result["end_time"]

            memory_used = model_result["memory_usage"]
            


            # Calculate metrics

            latency = calculate_latency(
                start_time,
                end_time
            )


            tokens = count_tokens(
                response
            )


            tokens_per_second = calculate_tokens_per_second(
                response,
                latency
            )


            


            results.append({

                "model_name": model_name,

                "prompt": prompt,

                "response": response,

                "latency": latency,

                "tokens": tokens,

                "tokens_per_second": tokens_per_second,

                "memory_usage": memory_used

            })


        return results


    except Exception:

        traceback.print_exc()

        return []


##3 Load prompt file
BASE_DIR = Path(__file__).parent

prompt_file = BASE_DIR / "prompt.json"


prompts = load_prompt(prompt_file)


#run and save the benchmark results
benchmark_results = []


for item in prompts:

    model_results = benchmark_run(item)

    benchmark_results.extend(model_results)
    print(benchmark_results)
    
    print("Benchmark completed:", item)



# Save benchmark results

output_file = BASE_DIR / "benchmark_results.json"


with open(output_file,"w",encoding="utf-8") as f:

    json.dump(benchmark_results,f,indent=4)


print("All benchmark results saved")