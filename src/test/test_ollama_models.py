from ollama_models import ask_all_model

try:

    prompt="""

      Explain Generative AI in simple words

     """
    response = ask_all_model(prompt)


    for result in response: 

        print("===============")
        print("model:")
        print(result["model"])

        print("\n response:")
        print(result["response"])
        
        print("\n latency:")
        print(result["latency"],"seconds")



except Exception as e:

    print(f"error occurence :{e}")