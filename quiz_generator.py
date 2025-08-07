iimport openai
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
- Use simple, straightforward language
- Focus on basic facts and definitions
- Test direct recall of information
- Avoid complex analysis or inference""",
            
            'medium': base_prompt + """
- Test comprehension and application
- Include some analytical thinking
- Mix factual and conceptual questions
- Require understanding of relationships between concepts""",
            
            'hard': base_prompt + """
- Require deep analysis and critical thinking
- Test ability to synthesize information
- Include complex scenarios and edge cases
- Challenge students to apply concepts in new contexts"""
        }
        
        return difficulty_prompts.get(difficulty, difficulty_prompts['medium'])
    
    def _build_prompt(self, chunk_text: str, difficulty: str, num_questions: int) -> str:
        """Build the main prompt for question generation"""
        return f"""
Based on the following text, generate exactly {num_questions} multiple choice question(s).

TEXT:
{chunk_text}

FORMAT YOUR RESPONSE AS JSON:
{{
    "questions": [
        {{
            "question": "Your question here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
            "explanation": "Brief explanation of why this is correct",
            "difficulty": "{difficulty}",
            "topic": "Main topic/concept being tested"
        }}
    ]
}}

Requirements:
- Generate exactly {num_questions} question(s)
- Make sure correct_answer is the index (0-3) of the correct option
- Keep questions clear and concise
- Ensure all options are plausible
- Include helpful explanations
- Focus on the most important concepts in the text
"""
    
    def _parse_questions(self, questions_text: str, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse questions from GPT response"""
        try:
            # Try to extract JSON from the response
            start_idx = questions_text.find('{')
            end_idx = questions_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = questions_text[start_idx:end_idx]
            parsed = json.loads(json_text)
            
            questions = []
            for q_data in parsed.get('questions', []):
                question = {
                    'question': q_data.get('question', ''),
                    'options': q_data.get('options', []),
                    'correct_answer': q_data.get('correct_answer', 0),
                    'explanation': q_data.get('explanation', ''),
                    'difficulty': q_data.get('difficulty', 'medium'),
                    'topic': q_data.get('topic', 'General'),
                    'chunk_index': chunk['index'],
                    'source': self._get_source_info(chunk),
                    'created_at': datetime.now().isoformat()
                }
                
                # Validate question
                if self._validate_question(question):
                    questions.append(question)
            
            return questions
            
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response: {str(e)}")
            return self._fallback_parse(questions_text, chunk)
        except Exception as e:
            st.error(f"Error parsing questions: {str(e)}")
            return []
    
    def _fallback_parse(self, text: str, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback parsing method if JSON parsing fails"""
        # This is a simple fallback - in practice, you might want more sophisticated parsing
        questions = []
        
        # Try to extract questions using patterns
        lines = text.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            if line.endswith('?'):
                # Likely a question
                current_question = {
                    'question': line,
                    'options': [],
                    'correct_answer': 0,
                    'explanation': 'Generated from text content',
                    'difficulty': 'medium',
                    'topic': 'General',
                    'chunk_index': chunk['index'],
                    'source': self._get_source_info(chunk)
                }
            elif line.startswith(('A.', 'B.', 'C.', 'D.', 'A)', 'B)', 'C)', 'D)')):
                # Likely an option
                if current_question:
                    option_text = line[2:].strip()
                    current_question['options'].append(option_text)
                    
                    if len(current_question['options']) == 4:
                        questions.append(current_question)
                        current_question = None
        
        return questions
    
    def _validate_question(self, question: Dict[str, Any]) -> bool:
        """Validate that a question is properly formatted"""
        required_fields = ['question', 'options', 'correct_answer']
        
        for field in required_fields:
            if field not in question:
                return False
        
        # Check question content
        if not question['question'] or len(question['question'].strip()) < 10:
            return False
        
        # Check options
        if len(question['options']) != 4:
            return False
        
        for option in question['options']:
            if not option or len(option.strip()) < 2:
                return False
        
        # Check correct answer
        if not isinstance(question['correct_answer'], int) or not (0 <= question['correct_answer'] <= 3):
            return False
        
        return True
    
    def _get_source_info(self, chunk: Dict[str, Any]) -> str:
        """Get source information for citation"""
        if chunk['source_type'] == 'pdf':
            pages = chunk.get('pages_covered', [1])
            page_str = f"page {pages[0]}" if len(pages) == 1 else f"pages {min(pages)}-{max(pages)}"
            return f"Document {page_str}, chunk {chunk['index'] + 1}"
        else:
            paragraphs = chunk.get('paragraphs_covered', [1])
            para_str = f"paragraph {paragraphs[0]}" if len(paragraphs) == 1 else f"paragraphs {min(paragraphs)}-{max(paragraphs)}"
            return f"Document {para_str}, chunk {chunk['index'] + 1}"
    
    def generate_follow_up_questions(self, incorrect_answers: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate follow-up questions for incorrect answers"""
        if not incorrect_answers:
            return []
        
        # Find relevant chunks for incorrect topics
        topics = [q.get('topic', 'General') for q in incorrect_answers]
        relevant_chunks = []
        
        for chunk in chunks:
            chunk_text_lower = chunk['text'].lower()
            for topic in topics:
                if topic.lower() in chunk_text_lower:
                    relevant_chunks.append(chunk)
                    break
        
        if not relevant_chunks:
            relevant_chunks = chunks[:2]  # Use first 2 chunks as fallback
        
        # Generate new questions focusing on the topics they got wrong
        follow_up_questions = []
        for chunk in relevant_chunks[:2]:  # Limit to 2 chunks
            questions = self._generate_questions_for_chunk(chunk, 'easy', 2)
            follow_up_questions.extend(questions)
        
        return follow_up_questions[:5]  # Limit to 5 follow-up questions
