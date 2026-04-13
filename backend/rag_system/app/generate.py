"""
Text generation module.

Classes:
- Generator: Handles prompt building and text generation
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Generator:
    """
    Generator for building prompts and generating answers.
    
    Methods:
    - build_prompt(query, retrieved_chunks): Create prompt from query and context
    - generate(prompt, max_new_tokens): Generate text from prompt
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the Generator.
        
        Args:
            model_name (str): Name of the HF model.
                             Defaults to gpt2 as a lightweight placeholder
        """
        if model_name is None:
            model_name = "gpt2"
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def build_prompt(self, query, retrieved_chunks):
        """
        Build a prompt from query and retrieved context chunks.
        
        Instructs the model to:
        - Answer using only retrieved context
        - Say so if evidence is insufficient
        
        Args:
            query (str): User query
            retrieved_chunks (list): List of retrieved chunk strings
            
        Returns:
            str: Formatted prompt
        """
        # Format context
        context = "\n\n".join([f"[Document {i+1}]:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Build prompt with instruction
        prompt = f"""Context documents:

{context}

Question: {query}

Instructions:
- Answer only using the context provided above.
- If the context doesn't contain enough information to answer, say "I don't know based on the provided context."
- Keep your answer concise and relevant.

Answer:"""
        
        return prompt
    
    def generate(self, prompt, max_new_tokens=256):
        """
        Generate text from a prompt.
        
        Args:
            prompt (str): Input prompt
            max_new_tokens (int): Maximum new tokens to generate
            
        Returns:
            str: Generated text (excluding the prompt)
        """
        # Tokenize prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Generate with temperature and top-p sampling for better quality
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract generated text (remove prompt)
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()
        
        return generated_text
