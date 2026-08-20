import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Offline SLM Benchmark",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    with open("C:/Users/sanoj/local_SLM_App_with_Ollama/src/data/evaluated_result.json", "r") as file:
        data = json.load(file)

    return pd.DataFrame(data)


df = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Offline SLM Benchmark Dashboard")

st.write(
    "Comparison of local Small Language Models based on "
    "performance and response quality."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Filter")

models = df["model_name"].unique().tolist()

selected_models = st.sidebar.multiselect(
    "Select Models",
    models,
    default=models
)

filtered_df = df[
    df["model_name"].isin(selected_models)
]


# ============================================================
# BENCHMARK ENVIRONMENT
# ============================================================

st.subheader("🖥️ Benchmark Environment")

st.info(
    """
    Performance metrics are hardware-dependent.

    Latency and memory usage can be affected by CPU,
    available RAM, background processes, model size,
    and whether GPU acceleration is available.

    Therefore, these results represent performance on
    the current benchmark hardware and should not be
    treated as universal model performance.
    """
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

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


# ============================================================
# MODEL SUMMARY
# ============================================================

model_summary = filtered_df.groupby("model_name").agg({

    "latency": "mean",

    "tokens": "mean",

    "tokens_per_second": "mean",

    "memory_usage": "mean",

    "correctness": "mean",

    "relevance": "mean",

    "conciseness": "mean",

    "instruction_following": "mean",

    "quality_score": "mean"

}).reset_index()


# ============================================================
# OVERALL SCORE
# ============================================================

# Quality normalization
model_summary["quality_normalized"] = (
    model_summary["quality_score"] / 5
)


# Speed normalization
model_summary["speed_normalized"] = (
    model_summary["tokens_per_second"]
    / model_summary["tokens_per_second"].max()
)


# Overall score
#
# Quality = 70%
# Speed   = 30%
#
# Quality is given higher weight because
# performance is strongly dependent on hardware.

model_summary["overall_score"] = (
    0.70 * model_summary["quality_normalized"]
    +
    0.30 * model_summary["speed_normalized"]
)


# ============================================================
# FIND BEST MODELS
# ============================================================

best_model = model_summary.loc[
    model_summary["overall_score"].idxmax()
]

fastest_model = model_summary.loc[
    model_summary["latency"].idxmin()
]

best_quality_model = model_summary.loc[
    model_summary["quality_score"].idxmax()
]

memory_efficient_model = model_summary.loc[
    model_summary["memory_usage"].idxmin()
]


# ============================================================
# OVERALL BEST MODEL
# ============================================================

st.subheader("🏆 Overall Benchmark Result")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🏆 Best on This Benchmark",
        best_model["model_name"]
    )

with col2:

    st.metric(
        "⚡ Lowest Latency",
        fastest_model["model_name"]
    )

with col3:

    st.metric(
        "⭐ Best Quality",
        best_quality_model["model_name"]
    )

with col4:

    st.metric(
        "💾 Lowest Memory",
        memory_efficient_model["model_name"]
    )


st.success(
    f"""
    {best_model["model_name"]} achieved the highest overall
    benchmark score based on response quality and generation speed.

    Overall Score:
    {best_model["overall_score"] * 100:.2f}%
    """
)


# ============================================================
# EXPLAIN OVERALL SCORE
# ============================================================

with st.expander("How is the Overall Score calculated?"):

    st.write(
        """
        The overall score combines response quality and
        generation speed.

        Quality Score = 70%

        Speed Score = 30%

        Overall Score =
        (0.70 × Quality Score) +
        (0.30 × Speed Score)

        Quality receives a higher weight because latency
        and memory usage are strongly influenced by the
        benchmark hardware.
        """
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.subheader("⚡ Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Latency",
        f"{filtered_df['latency'].mean():.2f}"
    )

with col2:

    st.metric(
        "Average Tokens",
        f"{filtered_df['tokens'].mean():.2f}"
    )

with col3:

    st.metric(
        "Average Tokens/sec",
        f"{filtered_df['tokens_per_second'].mean():.2f}"
    )

with col4:

    st.metric(
        "Average Memory",
        f"{filtered_df['memory_usage'].mean():.2f}"
    )


# ============================================================
# HARDWARE WARNING
# ============================================================

st.warning(
    """
    ⚠️ Hardware Consideration

    The benchmark was conducted on local hardware.
    Higher latency does not necessarily mean that a model
    is inherently slow.

    CPU performance, RAM availability, background processes,
    model size, quantization, and GPU availability can all
    affect inference latency.

    Therefore, latency should primarily be used to compare
    models within this benchmark environment.
    """
)


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

st.subheader("📈 Performance Comparison")

performance_metric = st.selectbox(
    "Select Performance Metric",
    [
        "latency",
        "tokens",
        "tokens_per_second",
        "memory_usage"
    ]
)


performance_data = (
    filtered_df
    .groupby("model_name")[performance_metric]
    .mean()
)


fig, ax = plt.subplots()

performance_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Model")
ax.set_ylabel(performance_metric)

ax.set_title(
    f"Average {performance_metric} by Model"
)

plt.xticks(rotation=0)

st.pyplot(fig)


# ============================================================
# QUALITY METRICS
# ============================================================

st.subheader("⭐ Response Quality")

quality_metric = st.selectbox(
    "Select Quality Metric",
    [
        "correctness",
        "relevance",
        "conciseness",
        "instruction_following",
        "quality_score"
    ]
)


quality_data = (
    filtered_df
    .groupby("model_name")[quality_metric]
    .mean()
)


fig, ax = plt.subplots()

quality_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Model")
ax.set_ylabel("Score")

ax.set_title(
    f"Average {quality_metric}"
)

ax.set_ylim(0, 5)

plt.xticks(rotation=0)

st.pyplot(fig)


# ============================================================
# PERFORMANCE VS QUALITY
# ============================================================

st.subheader("🎯 Performance vs Quality")

fig, ax = plt.subplots()

for model in filtered_df["model_name"].unique():

    model_data = filtered_df[
        filtered_df["model_name"] == model
    ]

    ax.scatter(
        model_data["tokens_per_second"],
        model_data["quality_score"],
        label=model
    )


ax.set_xlabel("Tokens Per Second")

ax.set_ylabel("Quality Score")

ax.set_title(
    "Generation Speed vs Response Quality"
)

ax.set_ylim(0, 5)

ax.legend()

st.pyplot(fig)


st.caption(
    """
    A model positioned toward the upper-right provides
    a stronger combination of generation speed and
    response quality.
    """
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

st.subheader("📋 Model Comparison")

display_summary = model_summary.copy()

display_summary["overall_score"] = (
    display_summary["overall_score"] * 100
).round(2)

display_summary = display_summary.rename(
    columns={
        "model_name": "Model",
        "latency": "Avg Latency",
        "tokens": "Avg Tokens",
        "tokens_per_second": "Avg Tokens/sec",
        "memory_usage": "Avg Memory",
        "correctness": "Correctness",
        "relevance": "Relevance",
        "conciseness": "Conciseness",
        "instruction_following": "Instruction Following",
        "quality_score": "Quality Score",
        "overall_score": "Overall Score (%)"
    }
)


columns_to_show = [
    "Model",
    "Avg Latency",
    "Avg Tokens",
    "Avg Tokens/sec",
    "Avg Memory",
    "Correctness",
    "Relevance",
    "Conciseness",
    "Instruction Following",
    "Quality Score",
    "Overall Score (%)"
]


display_summary = display_summary[
    columns_to_show
].round(2)


st.dataframe(
    display_summary,
    use_container_width=True
)


# ============================================================
# DETAILED RESULTS
# ============================================================

st.subheader("🔍 Detailed Benchmark Results")

st.dataframe(
    filtered_df,
    use_container_width=True
)


# ============================================================
# CONCLUSION
# ============================================================

st.subheader("📝 Benchmark Conclusion")

st.write(
    f"""
    Based on the current benchmark results, 
    {best_model["model_name"]} achieved the highest overall
    score.

    {fastest_model["model_name"]} achieved the lowest average
    latency, while {best_quality_model["model_name"]} achieved
    the highest response quality.

    These results are specific to the current benchmark
    prompts and hardware environment. Therefore, they should
    be interpreted as a comparative local benchmark rather
    than a universal ranking of the models.
    """
)