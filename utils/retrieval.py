
import streamlit as st
from pinecone import Pinecone


pc = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"]
)

index = pc.Index(
    st.secrets["PINECONE_INDEX_NAME"]
)


def retrieve_context(
    embedding,
    selected_file="All Files",
    top_k=3
):
    if not embedding:
        raise ValueError("Question embedding is empty.")

    filter_param = (
        None
        if selected_file == "All Files"
        else {"source": selected_file}
    )

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        **(
            {"filter": filter_param}
            if filter_param is not None
            else {}
        )
    )

    return [
        match["metadata"]["text"]
        for match in results.get("matches", [])
        if match.get("metadata", {}).get("text")
    ]

   
