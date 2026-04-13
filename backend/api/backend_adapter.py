class BackendAdapter:
    """
    Mock RAG adapter. Returns fake chunks and a placeholder response.
    Replace generate() internals when a real RAG system is connected.
    """

    def generate(self, prompt: str, top_k: int = 3) -> dict:
        retrieved_chunks = [
            {
                "id": f"chunk_{i + 1}",
                "source": f"doc_{i + 1}.txt",
                "text": f"[Mock chunk {i + 1}] Relevant excerpt for: '{prompt}'",
            }
            for i in range(top_k)
        ]

        response = f"[Mock response] This is a placeholder answer for: '{prompt}'"
        final_prompt = prompt

        return {
            "retrieved_chunks": retrieved_chunks,
            "response": response,
            "final_prompt": final_prompt,
        }
