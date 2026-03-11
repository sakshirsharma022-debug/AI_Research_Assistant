from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")

        )

    def generate_answer(self,question, context):

        prompt = f"""
        You are an experienced research mentor helping a student understand research papers.

        Your job is to:
        - Explain research ideas clearly
        - Compare different research approaches if multiple papers are referenced
        - Guide the user like a senior researcher
        - Be conversational but professional
        - Break down complex ideas in simple language

        If research papers are available, base your answers on the provided context.

        If the answer cannot be found in the context, say so and provide general guidance instead.
        Make sure you remember the previous conversations. 
        Context from research papers:
        {context}

        User Question:
        {question}

        Answer like a helpful research mentor:
        """
        
        completion = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"user","content":prompt}
            ],

        )

        return completion.choices[0].message.content
    
