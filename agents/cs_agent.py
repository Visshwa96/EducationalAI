from agents.baserag_agents import BaseRAGAgent


cs_agent = BaseRAGAgent(
    pdf_path="data/cs/cs_book.pdf",
    subject_name="Computer Science"
)