import os
from openai import OpenAI
from .rag_constants import LLAMA_API_KEY, LLAMA_ENDPOINT


class LLAMAClient:
    
    
    def __init__(self):
       self.client = OpenAI(
            api_key = LLAMA_API_KEY,
            base_url = LLAMA_ENDPOINT
        )

    def generate_content(self, prompt):
        """
        Sends input to llama3.1-70b using the chat API and returns the response.
        """
        response = self.client.chat.completions.create(
            model="llama3.1-70b",
            messages=[
                {"role": "system", "content": "You are an expert assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()