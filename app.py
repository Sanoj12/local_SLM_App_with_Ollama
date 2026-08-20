import streamlit as st
import pandas as pd
import json
import plotly.express as px

##page 

st.set_page_config(
    page_title="SLM Benchmark Dashboard",
    page_icon="🤖",
    layout="wide"
)

#title

st.title("⚡SLM Benchmark & Model Selection Dashboard")

st.caption("Performance • Quality • Efficiency • Final Model Recommendation")


#load data

with open("C:/Users/sanoj/local_SLM_App_with_Ollama/src/data/evaluated_result.json", "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)



##sidebar

st.sidebar.title("⚙️ Dashboard Controls")

selected_models = st.sidebar.multiselect(
    "Select Models",
    options=df["model_name"].unique(),
    default=df["model_name"].unique()
)


filtered_df = df[
    df["model_name"].isin(selected_models)
]

if filtered_df.empty:
    st.warning("Please select at least one model.")
    st.stop()



###summary



summary = filtered_df.groupby("model_name").agg(
    avg_latency=("latency", "mean"),
    median_latency=("latency", "median"),
    avg_tokens_per_second=("tokens_per_second", "mean"),
    avg_memory=("memory_usage", "mean"),
    avg_correctness=("correctness", "mean"),
    avg_relevance=("relevance", "mean"),
    avg_conciseness=("conciseness", "mean"),
    avg_instruction_following=("instruction_following", "mean"),
    avg_quality=("quality_score", "mean")
).reset_index()

summary = summary.round(3)


##3find best model


##quality score highest
best_quality_model = summary.loc[
    summary["avg_quality"].idxmax(),
    "model_name"
]

##latency lower is better
fastest_model = summary.loc[
    summary["avg_latency"].idxmin(),
    "model_name"
]


##token per second is higher is better
fastest_generation_model = summary.loc[
    summary["avg_tokens_per_second"].idxmax(),
    "model_name"
]

##memory usage lower is better
lowest_memory_model = summary.loc[
    summary["avg_memory"].idxmin(),
    "model_name"
]

##data overview

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Responses",
        len(df)
    )

with col2:
    st.metric(
        "Models",
        df["model_name"].nunique()
    )

with col3:
    st.metric(
        "Prompts",
        df["prompt"].nunique()
    )



st.divider()
with st.expander("🔍 View Raw Benchmark Data"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )




#cards

st.subheader("Key Results")

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "🏆 Best Quality",
        best_quality_model
    )

with col2:
    st.metric(
         "⚡ Fastest Latency",
        fastest_model
    )


with col3:
    st.metric(
        "🚀 Best Generation Speed",
        fastest_generation_model
    )
    

with col4:

    st.metric(
         "💾 Lowest Memory",
        lowest_memory_model
    )






st.divider()



st.warning(
    """
    ⚠️ Latency Warning

    The models showed response times of around 24–40 seconds.
    This may feel slow for real-time applications.

    Latency depends on the hardware, model size, and
    other processes running on the computer.
    """
)

#####Model comparison#########


st.subheader("📊 Model Comparison")

##change column name
display_summary = summary.rename(
    columns={
        "model_name": "Model",
        "avg_latency": "Latency (sec)",
        "median_latency": "Median Latency",
        "avg_tokens_per_second": "Tokens/sec",
        "avg_memory": "Memory",
        "avg_correctness": "Correctness",
        "avg_relevance": "Relevance",
        "avg_conciseness": "Conciseness",
        "avg_instruction_following": "Instruction Following",
        "avg_quality": "Quality"
    }
)


st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()


#### QUALITY VS PERFORMANCE

st.subheader("🎯 Quality vs Performance")

fig = px.scatter(
    summary,
    x="avg_latency",
    y="avg_quality",
    text="model_name",
    size="avg_quality",
    hover_data=[
        "avg_latency",
        "avg_quality",
        "avg_memory",
        "avg_tokens_per_second"
    ],
    labels={
        "avg_latency": "Average Latency (seconds)",
        "avg_quality": "Average Quality Score"
    },
    title="Lower Latency + Higher Quality = Better"
)


fig.update_traces(
    textposition="top center"
)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "The ideal model is toward the lower-left for latency "
    "and higher position for quality."
)


####PERFORMANCE SECTION
st.subheader("⚡ Performance Analysis")

col1, col2 = st.columns(2)

##Latency
with col1:
    
    ##Latency vs model
    latency = px.bar(
        summary,
        x="model_name",
        y="avg_latency",
        text="avg_latency",
        title = "Average Latency",
        labels={
            "model_name":"Model",
            "avg_latency":"Latency (seconds)"
        }
    )
    
    latency.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    latency.update_layout(
        height = 400
    )

    st.plotly_chart(
        latency,
        use_container_width=True
    )


with col2:

    speed = px.bar(
        summary,
        x="model_name",
        y="avg_tokens_per_second",
        text="avg_tokens_per_second",
        title="Average token per second",
        labels={
             "model_name": "Model",
            "avg_tokens_per_second": "Tokens / Second"
        }
    )

    speed.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    speed.update_layout(
        height=400
    )

    st.plotly_chart(
        speed,
        use_container_width=True
    )




#####m  MEMORY VS MODEL NAME

st.subheader("💾 Memory Usage")

memory = px.bar(
    summary,
    x="model_name",
    y="avg_memory",
    text="avg_memory",
    title="Average Memory Usage",
    labels={
        "model_name": "Model",
        "avg_memory": "Memory Usage"
    }
)

memory.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

memory.update_layout(
    height=450
)

st.plotly_chart(
    memory,
    use_container_width=True
)


## QUALITY ANALYSIS

st.subheader("⭐ Quality Analysis")

quality =  summary.melt(
    id_vars="model_name",
    value_vars=[
        "avg_correctness",
        "avg_relevance",
        "avg_conciseness",
        "avg_instruction_following"
    ],
    var_name = "Metric",
    value_name = "Score"
)

##change name

quality["Metric"] = quality["Metric"].replace(
    {
        "avg_correctness": "Correctness",
        "avg_relevance": "Relevance",
        "avg_conciseness": "Conciseness",
        "avg_instruction_following": "Instruction Following"
    }
    
)

quality = px.bar(
    quality,
    x="model_name",
    y="Score",
    color="Metric",
    barmode="group",
    title="Quality Metrics by Model",
    labels={
        "model_name": "Model",
        "Score": "Score"
    }
)

quality.update_layout(
    height=500
)

st.plotly_chart(
    quality,
    use_container_width=True
)


###qulity score
st.subheader("🏅 Overall Quality Score")

quality_score = px.bar(
    summary,
    x="model_name",
    y="avg_quality",
    text="avg_quality",
    title="Average Quality Score",
    labels={
        "model_name": "Model",
        "avg_quality": "Quality Score"
    }
)

quality_score.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)
quality_score.update_layout(
    height=400
)

st.plotly_chart(
    quality_score,
    use_container_width=True
)


##final decision

st.divider()

st.header("🏆 Final Model Recommendation")

##selct model

final_model = best_quality_model


final_row = summary[
    summary["model_name"] == final_model
].iloc[0]

st.success(f"Recommended Model: {final_model}")



col1,col2,col3 = st.columns(3)

with col1:

    st.metric(
        "Quality",
        f"{final_row['avg_quality']:.3f}"
    )


with col2:
    st.metric(
        "Latency",
        f"{final_row['avg_latency']:.2f} sec"
    )


with col3:
    st.metric(
        "Memory",
        f"{final_row['avg_memory']:.2f}"
    )


st.markdown(
    f"""
    Why {final_model} ?



    
    -  Highest average quality:{final_row['avg_quality']:.3f}
    -  Average latency:{final_row['avg_latency']:.2f} seconds
    -  Tokens per second:{final_row['avg_tokens_per_second']:.2f}
    -  Memory usage: {final_row['avg_memory']:.2f}

Final Decision: {final_model} provides a good overall balance
between response quality and performance for this benchmark.
"""
)
    
    
    


#footer

st.divider()

st.caption(
    "SLM Benchmark Project • Ollama • EDA • LLM-as-a-Judge Evaluation"
)    
    