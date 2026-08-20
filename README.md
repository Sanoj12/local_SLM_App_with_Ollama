# Offline SLM Benchmarking with Ollama

## Overview

This project benchmarks local Small Language Models (SLMs) using Ollama.

The benchmark compares models based on both:

- Performance
- Response quality

## Models

The benchmark evaluates:

- Llama 3.2:3B
- Phi-3 Mini
- Qwen 2.5:3B

## Performance Metrics

The following performance metrics are collected:

- Latency
- Token count
- Tokens per second
- Memory usage

## Quality Evaluation

The project uses **LLM-as-a-Judge** to automatically evaluate
the quality of generated responses.

The judge evaluates each response using:

- Correctness
- Relevance
- Conciseness
- Instruction Following

An overall `quality_score` is calculated from these evaluation scores.

## Why LLM-as-a-Judge?

The benchmark contains natural-language responses from multiple
local SLMs. Rule-based evaluation can work well for simple
tasks such as exact answers or classification, but it is limited
when different responses use different wording while expressing
the same meaning.

Therefore, this project uses LLM-as-a-Judge to evaluate the
quality of natural-language responses using a consistent rubric.

The judge receives the question and the model response and
returns structured evaluation scores.

Ollama is used for the judge so that the evaluation can run
locally without depending on an external API.

## Evaluation Workflow

final.json
    ↓
LLM-as-a-Judge
    ↓
Correctness
Relevance
Conciseness
Instruction Following
    ↓
Quality Score
    ↓
evaluated_results.json
    ↓
EDA
    ↓
Streamlit Dashboard

## Project Structure

local_SLM_App_with_Ollama/
│
├── data/
│   ├── final.json
│   └── evaluated_results.json
│
├── evaluation/
│   └── llm_judge.py
│
├── eda/
│   └── eda.ipynb
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── benchmark_service.py
│   └── ollama_models.py
│
├── requirements.txt
├── README.md
└── .gitignore

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Make sure Ollama is running

ollama list

### 3. Run the benchmark

python src/benchmark_service.py

### 4. Run LLM-as-a-Judge

python evaluation/llm_judge.py

### 5. Analyze the results

Open:

eda/eda.ipynb

### 6. Run the dashboard

streamlit run dashboard/app.py

## Output

The benchmark produces:

- `final.json` — raw benchmark results
- `evaluated_results.json` — benchmark results with LLM-as-a-Judge scores

## Limitations

LLM-as-a-Judge is not guaranteed to be perfectly objective.
The judge model can have biases or inconsistencies.

To improve reliability, this project uses:

- A fixed evaluation rubric
- Consistent scoring criteria
- Temperature 0
- Structured JSON output

For a production-grade evaluation system, human evaluation
or comparison with a stronger independent judge can be added.