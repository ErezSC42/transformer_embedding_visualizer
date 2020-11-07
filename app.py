import os
import pickle
import warnings
import altair as alt
import streamlit as st
import streamlit_theme as stt

stt.set_theme({
    'primary':'#00cc99',
})

SAMPLES_TO_DISPALY = 20
X_LIM_MIN = -20
X_LIM_MAX = 20
Y_LIM_MIN = -20
Y_LIM_MAX = 22
seed = 42
warnings.filterwarnings(action="ignore")

LABELS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]


def _set_block_container_style(
        max_width: int = 1200,
        max_width_100_percent: bool = False,
        padding_top: int = 0,
        padding_right: int = 1,
        padding_left: int = 1,
        padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}

</style>
""",
        unsafe_allow_html=True,
    )


#@st.cache
def load_data():
    temp_bert = {}
    temp_roberta = {}
    for i in range(1,13):
        bert_results_path = os.path.join("umap_pickles", f"bert_base_embedding_layer_{i}.pkl")
        roberta_results_path = os.path.join("umap_pickles", f"roberta_base_embedding_layer_{i}.pkl")
        with open(bert_results_path, "rb") as fp:
            temp_bert[i] = pickle.load(fp)
        with open(roberta_results_path, "rb") as fp:
            temp_roberta[i] = pickle.load(fp)
    return temp_bert, temp_roberta


bert_embeddings_dataframes, roberta_embeddings_dataframes = load_data()
#print(type(bert_embeddings_dataframes[1][0]))



st.title("Transformer Embedding Explorer")

chosen_bert_layer = st.sidebar.slider(
    min_value=1,
    max_value=12,
    label="Layer",
    step=1,
    value=4
)


labels_to_use = st.sidebar.multiselect(
    "Select POS labels:",
    options=LABELS,
    default=LABELS
)


query_df_bert, lower_dim_data_bert = bert_embeddings_dataframes[chosen_bert_layer]
query_df_roberta, lower_dim_data_roberta = roberta_embeddings_dataframes[chosen_bert_layer]
query_df_roberta.to_csv("roberta_data.csv")
query_df_bert.to_csv("bert_data.csv")
query_df_roberta["word"] = query_df_roberta["word"].str.replace(" ","")
query_df_roberta["word"] = query_df_roberta["word"].str.lower()
query_df_bert["index"] =  query_df_bert.index
query_df_roberta["index"] =  query_df_roberta.index


selected_input_type = st.sidebar.radio(
    "Select Input Mode",
    ("word", "all", "row")
)


input_display = "Word" if selected_input_type == "word" else "Row Index"

if selected_input_type == "row":
    query_text = st.sidebar.number_input(
        label=f"Select {input_display}",
        value=seed,
        max_value=42100,
        min_value=1
    )
    row_text_bert = query_df_bert["text"].loc[query_text]
    row_text_roberta = query_df_roberta["text"].loc[query_text]
    matching_idx_bert = (query_df_bert["text"] == row_text_bert) & (query_df_bert["label"].isin(labels_to_use))
    matching_idx_roberta = (query_df_roberta["text"] == row_text_roberta) & (query_df_roberta["label"].isin(labels_to_use))
    st.write(f"{row_text_bert}")

    viz_df_bert = query_df_bert[matching_idx_bert]
    viz_df_roberta = query_df_roberta[matching_idx_roberta]

elif selected_input_type == "word":
    query_text = st.sidebar.text_input(
        label=f"Select {input_display}",
        max_chars=15,
        value="",
    )
    if query_text != "":
        matching_idx_bert = (query_df_bert["word"] == query_text) & (query_df_bert["label"].isin(labels_to_use))
        matching_idx_roberta = (query_df_roberta["word"] == query_text) & (query_df_roberta["label"].isin(labels_to_use))
        st.write(query_text)
        viz_df_bert = query_df_bert[matching_idx_bert]
        viz_df_roberta = query_df_roberta[matching_idx_roberta]

    else:
        matching_idx_bert = (query_df_bert["word"] == query_text) & (query_df_bert["label"].isin(labels_to_use))
        matching_idx_roberta = (query_df_roberta["word"] == query_text) & (query_df_roberta["label"].isin(labels_to_use))
        viz_df_bert = query_df_bert[matching_idx_bert]
        viz_df_roberta = query_df_roberta[matching_idx_roberta]

else:  # all
    matching_idx_bert = query_df_bert["label"].isin(labels_to_use)
    matching_idx_roberta = query_df_roberta["label"].isin(labels_to_use)

    viz_df_bert = query_df_bert[matching_idx_bert]
    viz_df_roberta = query_df_roberta[matching_idx_roberta]



viz_df_bert["X"] = lower_dim_data_bert[matching_idx_bert,0]
viz_df_bert["Y"] = lower_dim_data_bert[matching_idx_bert,1]


viz_df_roberta["X"] = lower_dim_data_roberta[matching_idx_roberta,0]
viz_df_roberta["Y"] = lower_dim_data_roberta[matching_idx_roberta,1]



#st.write(f"Results: {len(matching_idx_bert)}")
#st.write(f"Results: {len(matching_idx_roberta)}")

#print(matching_idx_roberta)



st.write(f"BERT Results: {len(viz_df_bert)}")

if len(viz_df_bert) > 0:
    chart_bert = alt.Chart(viz_df_bert).mark_point(size=50, filled=True).encode(
        alt.X("X", scale=alt.Scale(domain=(X_LIM_MIN, X_LIM_MAX))),
        alt.Y("Y", scale=alt.Scale(domain=(Y_LIM_MIN, Y_LIM_MAX))),
        color="label",
        tooltip=['index','word_offset','word', 'label', 'text']
    ).interactive()
    st.altair_chart(chart_bert, use_container_width=True)


st.write(f"RoBERTa Results: {len(viz_df_roberta)}")
if len(viz_df_roberta) > 0:
    chart_roberta = alt.Chart(viz_df_roberta).mark_point(size=50, filled=True).encode(
        alt.X("X", scale=alt.Scale(domain=(X_LIM_MIN, X_LIM_MAX))),
        alt.Y("Y", scale=alt.Scale(domain=(Y_LIM_MIN, Y_LIM_MAX))),
        color="label",
        tooltip=['index','word_offset', 'word', 'label', 'text']
    ).interactive()
    st.altair_chart(chart_roberta, use_container_width=True)

#st.write(viz_df_bert.head(SAMPLES_TO_DISPALY))
#st.write(viz_df_roberta.head(SAMPLES_TO_DISPALY))

_set_block_container_style(max_width_100_percent=True)
