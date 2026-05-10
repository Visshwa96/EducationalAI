from agents.baserag_agents import BaseRAGAgent


math_agent = BaseRAGAgent(
    pdf_path="data/math/math_book.pdf",
    subject_name="Math"
)