from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
##############################################################

##################################################################

def retrieve_context(embedding, selected_file="All Files", top_k=3):
    # If "All Files" is selected, don't apply a filter
    filter_param = None if selected_file == "All Files" else {"source": selected_file}

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_param
    )

    return [match["metadata"]["text"] for match in results["matches"]]

   