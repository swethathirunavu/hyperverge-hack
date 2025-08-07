import openai
import json
import os
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime

class QuizEngine:
    def __init__(self):
        self.api_key = self._get_api_key()
        if self.api_key:
            openai.api_key = self.api_key
    
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
    
    def get_explanation(self, user_question: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate AI explanation for user's question based on document chunks"""
        if not self.api_key:
            return "AI explanations are not available without an OpenAI API key."
        
        # Find most relevant chunks for the question
        relevant_chunks = self._find_relevant_chunks(user_question, chunks)
        
        # Combine relevant text
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks[:3]])  # Use top 3 chunks
        
        prompt = f"""
Based on the following document content, please provide a clear and helpful explanation for the user's question.

DOCUMENT CONTENT:
{context}

USER QUESTION: {user_question}

Please provide a comprehensive explanation that:
1. Directly addresses the user's question
2. Uses information from the document content
3. Is easy to understand
4. Includes relevant examples if available in the text
5. Cites which part of the document the information comes from

If the question cannot be answered from the document content, please say so clearly.
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI tutor who explains concepts clearly and accurately based on provided document content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _find_relevant_chunks(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find chunks most relevant to the user's question"""
        question_lower = question.lower()
        scored_chunks = []
        
        for chunk in chunks:
            chunk_text_lower = chunk['text'].lower()
            
            # Simple scoring based on keyword overlap
            question_words = set(question_lower.split())
            chunk_words = set(chunk_text_lower.split())
            
            # Calculate overlap score
            overlap = len(question_words.intersection(chunk_words))
            score = overlap / len(question_words) if question_words else 0
            
            scored_chunks.append((score, chunk))
        
        # Sort by score and return chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks if score > 0]
    
    def generate_hint(self, question: Dict[str, Any], user_answer: int) -> str:
        """Generate a helpful hint for a question"""
        if not self.api_key:
            return "Check the relevant section in the document for more context."
        
        correct_option = question['options'][question['correct_answer']]
        user_option = question['options'][user_answer] if 0 <= user_answer < len(question['options']) else "Invalid"
        
        prompt = f"""
The user answered a quiz question incorrectly. Provide a helpful hint without giving away the answer directly.

QUESTION: {question['question']}
CORRECT ANSWER: {correct_option}
USER'S ANSWER: {user_option}
EXPLANATION: {question.get('explanation', 'No explanation available')}

Provide a gentle hint that:
1. Doesn't reveal the correct answer directly
2. Points them in the right direction
3. Encourages them to think about the concept
4. Is supportive and educational

Keep the hint concise (2-3 sentences maximum).
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a supportive AI tutor who provides helpful hints without giving away answers directly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Think about {question.get('topic', 'the main concept')} and review the relevant section."
    
    def analyze_performance(self, questions: List[Dict[str, Any]], user_answers: Dict[int, int]) -> Dict[str, Any]:
        """Analyze user's quiz performance"""
        total_questions = len(questions)
        correct_answers = 0
        topic_performance = {}
        difficulty_performance = {}
        
        for i, question in enumerate(questions):
            user_answer = user_answers.get(i)
            is_correct = user_answer == question['correct_answer']
            
            if is_correct:
                correct_answers += 1
            
            # Track topic performance
            topic = question.get('topic', 'General')
            if topic not in topic_performance:
                topic_performance[topic] = {'correct': 0, 'total': 0}
            topic_performance[topic]['total'] += 1
            if is_correct:
                topic_performance[topic]['correct'] += 1
            
            # Track difficulty performance
            difficulty = question.get('difficulty', 'medium')
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = {'correct': 0, 'total': 0}
            difficulty_performance[difficulty]['total'] += 1
            if is_correct:
                difficulty_performance[difficulty]['correct'] += 1
        
        # Calculate percentages
        overall_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Find strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for topic, performance in topic_performance.items():
            percentage = (performance['correct'] / performance['total']) * 100
            if percentage >= 80:
                strengths.append(topic)
            elif percentage < 60:
                weaknesses.append(topic)
        
        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'overall_percentage': round(overall_percentage, 1),
            'topic_performance': topic_performance,
            'difficulty_performance': difficulty_performance,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'grade': self._calculate_grade(overall_percentage)
        }
    
    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade based on percentage"""
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
    
    def generate_study_recommendations(self, performance: Dict[str, Any], chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate study recommendations based on performance"""
        recommendations = []
        
        # Overall performance recommendations
        if performance['overall_percentage'] >= 80:
            recommendations.append("🎉 Excellent work! You have a strong grasp of the material.")
        elif performance['overall_percentage'] >= 60:
            recommendations.append("👍 Good progress! Focus on the areas below to improve further.")
        else:
            recommendations.append("📚 Keep studying! There's significant room for improvement.")
        
        # Topic-specific recommendations
        if performance['weaknesses']:
            weak_topics = ', '.join(performance['weaknesses'])
            recommendations.append(f"🎯 Focus on these topics: {weak_topics}")
            recommendations.append("💡 Try re-reading the relevant sections and taking notes on key concepts.")
        
        # Difficulty-based recommendations
        difficulty_perf = performance['difficulty_performance']
        if 'hard' in difficulty_perf and difficulty_perf['hard']['total'] > 0:
            hard_percentage = (difficulty_perf['hard']['correct'] / difficulty_perf['hard']['total']) * 100
            if hard_percentage < 50:
                recommendations.append("🧠 Practice more analytical thinking and application of concepts.")
        
        if 'easy' in difficulty_perf and difficulty_perf['easy']['total'] > 0:
            easy_percentage = (difficulty_perf['easy']['correct'] / difficulty_perf['easy']['total']) * 100
            if easy_percentage < 70:
                recommendations.append("📖 Review basic definitions and fundamental concepts.")
        
        return recommendations
