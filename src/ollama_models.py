
import ollama
import time

from metrics_services import memory_usage

def ollama_model(model_name, prompt, temperature=0.3,num_predict =128):

    try:



        memory_before = memory_usage()
        

        start_time = time.perf_counter()

         


        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": temperature,
                "num_predict":num_predict
            }
        )


        end_time = time.perf_counter()

        memory_after = memory_usage()

        memory_used = memory_after - memory_before


        text_response = response.message.content


        return {

            "model_name": model_name,

            "response": text_response,

            "start_time": start_time,

            

            "end_time": end_time,
            "memory_usage": round(memory_used, 2)
        }


    except Exception as e:

        return {
            "model_name": model_name,
            "error": str(e)
        }

def ask_all_model(prompt):
    try:
        models = [
            "llama3.2:3b",
            "phi3:mini",
            "qwen2.5:3b"
        ]

        results = []

        for model in models:
            results.append(ollama_model(model, prompt))


        return results
        print(f"check models:",results)

    except Exception as e:
        print(f"Error in ask_all_model: {e}")
        return []




##retry mechanism

def retry_model(model_name,prompt,temperature=0.3):

    retry_prompt = f"""
    
    your previous response did not satisfy the required format


    please answer the following prompt again

    prompt:{prompt}

    return only the vaild json format answer
    
    """

    return ollama_model(
        model_name = model_name,
        prompt =retry_prompt,
        temperature=temperature
    )
    