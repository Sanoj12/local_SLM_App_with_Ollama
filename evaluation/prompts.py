


def create_judge_prompt(prompt,response,ground_truth):

    judge_prompt = f"""
       
       you are an llm evaluator.

       Evaluate the ai generated answer using the question,expected answer,and the following criteria.


       evaluate these four criteria:

       1.correctness
       2.relevance
       3.conciseness
       4.instruction following

    
    use only these scores:

    0 = Fails
    1 = partially satisfies
    2 = fully satisifies


    CORRECTNESS

       2 = Fully correct.
       
       1 = Partially correct, incomplete, or has a small factual error.
       
       0 = Incorrect.

    RELEVANCE

           2 = Directly answers the question.
           
           1=Partially answers the question.
           
           0 = Does not answer the question.

    Conciseness

          2 =Short and contains only necessary information.
          1 =Somewhat longer than necessary.
          0 =Very long and contains mostly unnecessary information.

    Instruction following

        2 =Follows all explicit instructions.
        
        1 = Partially follows the instructions.
         0 =Does not follow the instructions.

If the question has no explicit instructions, give 2.

      
      
       rules:
      
         - do not require exact wording.
         -evalaute meaning,not wording
         -Extra correct information does not make an answer incorrect.
         -Do not require the AI answer to exactly match the expected answer.
         -do not penalize correctness because of extra correct information.
         - Scores must agree with the written reason.
         - Use ONLY 0, 1, or 2.

         return only valid json 



         {{
    "correctness": 0,
    "relevance": 0,
    "conciseness": 0,
    "instruction_following": 0,
    "reason": "brief explain of the evalaution."
}}


   QUESTION:
            {prompt}

   EXPECTED ANSWER:
           {ground_truth}

   AI ANSWER:
            {response}
 
     
"""
    return judge_prompt