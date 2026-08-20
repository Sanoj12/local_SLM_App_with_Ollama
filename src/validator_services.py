from pydantic import BaseModel,Field,ValidationError





class BenchmarkResult(BaseModel):
    model_name :str
    prompt:str
    response:str
    latency:float = Field(ge=0)
    tokens:int= Field(ge=0)
    tokens_per_second:float = Field(ge=0)
    memory_usage:float = Field(ge=0)




###valid json -> required fields -> correct datatypes

def validate_json(output):

    try:
    
        result = BenchmarkResult(**output)

        return result
    
    except ValidationError as e:
        print(f"validation failed:{e}")
        return None
        