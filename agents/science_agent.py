from agents.baserag_agents import BaseRAGAgent


science_agent = BaseRAGAgent(
    pdf_path="data/science/science_book.pdf",
    subject_name="Science"
)