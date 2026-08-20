import psutil

import traceback
#calculate total response latency in seconds
def calculate_latency(start_time,end_time):
       try:
          return end_time-start_time
        
       except Exception:
          
          traceback.print_exc()
          return 0





##count the number of tokens

def count_tokens(text):

    try:

        return len(text.split())
    

    except Exception:
        traceback.print_exc()
        return 0


##calculate token generation speed

def calculate_tokens_per_second(text,latency):

    try:

        tokens= count_tokens(text)

        if latency <=0:
            return 0 

        return round(tokens/latency,2)
    
    except Exception:

        traceback.print_exc()
        
        return 0



####check memory usage

def memory_usage():
    total_memory = 0

    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            name = process.info["name"]

            if name and "ollama" in name.lower():
                total_memory += process.info["memory_info"].rss

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return total_memory / (1024 * 1024)  # MB