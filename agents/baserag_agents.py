from pathlib import Path
import json
import re

import ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseRAGAgent:

    def __init__(self, pdf_path, subject_name):

        self.pdf_path = Path(pdf_path)

        self.subject_name = subject_name

        self.db = None

    def initialize(self):

        loader = PyMuPDFLoader(str(self.pdf_path))

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma.from_documents(
            chunks,
            embeddings
        )

        print(f"{self.subject_name} Agent Ready!")

    def ask(self, question):

        results = self.db.similarity_search(question, k=5)

        context = ""

        for result in results:
            context += result.page_content + "\n"

        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[
                {
                    'role': 'system',
                    'content': f'''
You are a helpful {self.subject_name} teacher.

Teach students clearly and simply.

Generate quizzes when requested.

Only answer from the provided context.
'''
                },
                {
                    'role': 'user',
                    'content': f"""
Context:
{context}

Question:
{question}
"""
                }
            ]
        )

        return response['message']['content']

    def _get_context(self, query, k=6):

        if self.db is None:
            raise RuntimeError("Agent database is not initialized.")

        results = self.db.similarity_search(query, k=k)

        return "\n".join(result.page_content for result in results)

    def _chat(self, system_prompt, user_prompt):

        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ]
        )

        return response['message']['content']

    def _extract_json(self, raw_text):

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("The AI response did not contain valid JSON.")

        return json.loads(match.group(0))

    def get_study_context(self, chapter):

        query = f"{self.subject_name} {chapter} key concepts"
        return self._get_context(query=query, k=8)

    def generate_study_guide(self, chapter, depth="standard"):

        context = self.get_study_context(chapter)

        prompt = f"""
Create a study guide for {self.subject_name} focused on chapter/topic: {chapter}.

Depth: {depth}

Use ONLY the provided context and make the response easy to revise from.
Include:
1. A short overview
2. Key concepts
3. Important formulas, definitions, or facts if relevant
4. 3-5 self-check questions
5. A simple memory trick or recap
"""

        guide = self._chat(
            system_prompt=f"""
You are an expert {self.subject_name} tutor.
Your job is to turn study material into clear revision notes.
""",
            user_prompt=f"""
Context:
{context}

Instruction:
{prompt}
"""
        )

        return context, guide

    def generate_test_quiz(self, chapter, num_questions=5, difficulty="medium"):

        context = self.get_study_context(chapter)

        prompt = f"""
Create a self-test quiz for {self.subject_name} focused on chapter/topic: {chapter}.

Difficulty: {difficulty}
Number of questions: {num_questions}

Return ONLY valid JSON in this schema:
{{
  "title": "string",
  "questions": [
    {{
      "id": 1,
      "question": "string",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "string"
    }}
  ]
}}

Rules:
1. Use ONLY the provided context.
2. Make every question multiple choice with exactly 4 options.
3. Keep the questions clear and student-friendly.
4. Ensure correct_index is 0-based.
5. Do not wrap the JSON in markdown fences.
"""

        raw_response = self._chat(
            system_prompt=f"""
You are an expert {self.subject_name} teacher.
Generate a clean JSON quiz for self-testing.
""",
            user_prompt=f"""
Context:
{context}

Instruction:
{prompt}
"""
        )

        quiz_data = self._extract_json(raw_response)
        return context, quiz_data

    def generate_quiz(self, chapter, num_questions=5, difficulty="medium"):

        query = f"{self.subject_name} {chapter} key concepts"
        context = self._get_context(query=query, k=8)

        prompt = f"""
Create a {self.subject_name} quiz focused on chapter/topic: {chapter}.

Difficulty: {difficulty}
Number of questions: {num_questions}

Rules:
1. Use ONLY the provided context.
2. Include a mix of question types (MCQ and short answer) when possible.
3. Provide an answer key after the questions.
4. Keep wording clear for students.
5. If context is insufficient, state that clearly and still create best-effort questions from available content.
"""

        response = self._chat(
            system_prompt=f"""
You are an expert {self.subject_name} teacher.
Generate high-quality quizzes from retrieved study material.
""",
            user_prompt=f"""
Context:
{context}

Instruction:
{prompt}
"""
        )

        return response

    def generate_quiz_with_context(self, chapter, num_questions=5, difficulty="medium"):

        query = f"{self.subject_name} {chapter} key concepts"
        context = self._get_context(query=query, k=8)

        prompt = f"""
Create a {self.subject_name} quiz focused on chapter/topic: {chapter}.

Difficulty: {difficulty}
Number of questions: {num_questions}

Rules:
1. Use ONLY the provided context.
2. Include a mix of question types (MCQ and short answer) when possible.
3. Provide an answer key after the questions.
4. Keep wording clear for students.
5. If context is insufficient, state that clearly and still create best-effort questions from available content.
"""

        response = self._chat(
            system_prompt=f"""
You are an expert {self.subject_name} teacher.
Generate high-quality quizzes from retrieved study material.
""",
            user_prompt=f"""
Context:
{context}

Instruction:
{prompt}
"""
        )

        return context, response