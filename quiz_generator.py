import openai
import json
import random
from typing import List, Dict, Any
import streamlit as st
import os
from datetime import datetime

class QuizGenerator:
    def __init__(self):
        self.api_key = self._get_api_key()
        if self.api_key:
            openai.api_key = self.api_key
        else:
            st.error("OpenAI API key not found! Please set OPENAI_API_KEY in environment variables.")
    
    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment or Streamlit secrets"""
        # Try environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Try Streamlit secrets
        if not api_key and hasattr(st, 'secrets'):
            try:
                api_key = st.secrets['OPENAI_API_KEY']
            except:
                pass
        
        return api_key
    
    def generate_quiz(self, chunks: List[Dict[str, Any]], difficulty: str = 'medium', num_questions: int = 10) -> List[Dict[str, Any]]:
        """Generate quiz questions from document chunks"""
        if not self.api_key:
            st.error("Cannot generate quiz without OpenAI API key!")
            return []
        
        questions = []
        total_chunks = len(chunks)
        questions_per_chunk = max(1, num_questions // total_chunks)
        
        # Distribute questions across chunks
        chunk_question_count = self._distribute_questions(total_chunks, num_questions)
        
        with st.progress(0) as progress_bar:
            for i, chunk in enumerate(chunks):
                num_q_for_chunk = chunk_question_count[i]
                if num_q_for_chunk > 0:
                    chunk_questions = self._generate_questions_for_chunk(
                        chunk, difficulty, num_q_for_chunk
                    )
                    questions.extend(chunk_questions)
                
                # Update progress
                progress_bar.progress((i + 1) / total_chunks)
        
        # Shuffle questions for variety
        random.shuffle(questions)
        
        # Limit to requested number
        questions = questions[:num_questions]
        
        # Add question indices
        for i, question in enumerate(questions):
            question['question_id'] = i + 1
        
        return questions
    
    def _distribute_questions(self, num_chunks: int, total_questions: int) -> List[int]:
        """Distribute questions evenly across chunks"""
        base_questions = total_questions // num_chunks
        extra_questions = total_questions % num_chunks
        
        distribution = [base_questions] * num_chunks
        
        # Distribute extra questions randomly
        for i in range(extra_questions):
            distribution[i] += 1
        
        return distribution
    
    def _generate_questions_for_chunk(self, chunk: Dict[str, Any], difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate questions for a specific chunk"""
        prompt = self._build_prompt(chunk['text'], difficulty, num_questions)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Using more cost-effective model
                messages=[
                    {"role": "system", "content": self._get_system_prompt(difficulty)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            questions_text = response.choices[0].message.content
            questions = self._parse_questions(questions_text, chunk)
            
            return questions
            
        except Exception as e:
            st.error(f"Error generating questions for chunk {chunk['index']}: {str(e)}")
            return []
    
    def _get_system_prompt(self, difficulty: str) -> str:
        """Get system prompt based on difficulty"""
        base_prompt = """You are an expert quiz generator. Create high-quality multiple choice questions based on the provided text. 

Follow these guidelines:
- Create questions that test understanding, not just memorization
- Provide 4 answer options (A, B, C, D)
- Make wrong answers plausible but clearly incorrect
- Include brief explanations for correct answers
- Focus on key concepts and important information"""

        difficulty_prompts = {
            'easy': base_prompt + """
- Use simple, straightforwar
