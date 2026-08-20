import json

from validator_services import validate_json

from ollama_models import retry_model


MAX_RETRIES = 2
def final_record(record):

    try:
        #  Validate json record
        result = validate_json(record)

        
        if result is not None:
           
            print("Valid record successfully")
           
            return result.model_dump()

        # Retry if validation failed

        for attempt in range(MAX_RETRIES):

            print("Validation failed")

            retry_result = retry_model(
               
                record["model_name"],
               
                record["prompt"]
            )

            
            result = validate_json(retry_result)

            
            if result is not None:
               
                print("Retry successfully completed")
                
                return result.model_dump()

        #  All retries failed
        print("Retry failed")
        
        return None

    except KeyError as e:

        print(f"Missing field in record: {e}")

        return None

    except Exception as e:

        print(f"error: {e}")

        return None


#open json file for validation

with open("data/benchmark_results.json", "r") as file:

        records = json.load(file)


final_results = []
errors = []


for record in records:


    try:
       result = final_record(record)

       if result is not None:

          final_results.append(result)
       
       else:

          errors.append({

                "model_name": record.get("model_name"),

                "prompt": record.get("prompt"),

                "error": "Validation failed after retry"
           })
    
    except Exception as e:

        print(f"retry error:{e}")
        



##save final records
with open("data/final.json", "w") as file:

        json.dump(final_results,file,indent=4)



    # Save failed records

with open("data/error.json", "w") as file:

        json.dump(errors,file,indent=4)

print("processing completed")