from pathlib import Path

import streamlit as st

from agents.baserag_agents import BaseRAGAgent


UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdf(uploaded_file):
    sanitized_name = uploaded_file.name.replace(" ", "_")
    target_path = UPLOAD_DIR / sanitized_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def initialize_uploaded_agent(pdf_path, subject):
    agent = BaseRAGAgent(pdf_path=pdf_path, subject_name=subject)
    agent.initialize()
    return agent


def build_study_content(agent, subject, chapter, study_depth):
    if hasattr(agent, "generate_study_guide"):
        return agent.generate_study_guide(chapter=chapter, depth=study_depth)

    fallback_prompt = (
        f"Create a study guide for {subject} focused on chapter/topic: {chapter}. "
        f"Depth: {study_depth}. Include an overview, key concepts, and self-check questions."
    )
    return agent._get_context(query=f"{subject} {chapter} key concepts", k=8), agent.ask(fallback_prompt)


def build_quiz_content(agent, subject, chapter, num_questions, difficulty):
    if hasattr(agent, "generate_test_quiz"):
        return agent.generate_test_quiz(
            chapter=chapter,
            num_questions=num_questions,
            difficulty=difficulty,
        )

    fallback_prompt = (
        f"Return a multiple-choice quiz test for {subject} focused on {chapter}. "
        f"Use {num_questions} questions and difficulty {difficulty}."
    )
    raw_output = agent.ask(fallback_prompt)
    return agent._get_context(query=f"{subject} {chapter} key concepts", k=8), {
        "title": f"{subject} quiz test",
        "questions": [],
        "raw_output": raw_output,
    }


def quiz_session_key(result):
    return f"quiz_{result['subject']}_{result['chapter']}_{result['mode']}"


def score_self_test(quiz_data):
    questions = quiz_data.get("questions", [])
    score = 0
    results = []

    quiz_key = quiz_data.get("session_key", "quiz")

    for question in questions:
        key = f"{quiz_key}_answer_{question['id']}"
        selected_option = st.session_state.get(key)
        correct_index = question.get("correct_index", 0)
        correct_option = question["options"][correct_index]
        is_correct = selected_option == correct_option

        if is_correct:
            score += 1

        results.append(
            {
                "question": question,
                "selected_option": selected_option,
                "correct_option": correct_option,
                "is_correct": is_correct,
            }
        )

    return score, len(questions), results


st.set_page_config(
    page_title="EducationalAI Study and Quiz Builder",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Serif:wght@500;700&display=swap');

        :root {
            --bg-start: #fef6e4;
            --bg-end: #b8c1ec;
            --ink: #172c66;
            --card: #ffffff;
            --accent: #f582ae;
            --accent-2: #8bd3dd;
        }

        .stApp {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at 10% 20%, var(--bg-start), var(--bg-end));
        }

        .hero {
            background: linear-gradient(120deg, rgba(255,255,255,0.78), rgba(255,255,255,0.52));
            border: 1px solid rgba(23,44,102,0.15);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 30px rgba(23,44,102,0.08);
            animation: slideup 0.5s ease-out;
        }

        .hero h1 {
            font-family: 'IBM Plex Serif', serif;
            margin: 0;
            font-size: clamp(1.6rem, 2.8vw, 2.4rem);
        }

        .hero p {
            margin: 0.4rem 0 0;
            opacity: 0.9;
        }

        @keyframes slideup {
            from { transform: translateY(16px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>EducationalAI Study and Quiz Builder</h1>
        <p>Upload subject material, study the chapter, or generate a scored quiz test directly inside the app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = "Study"

mode = st.radio(
    "What would you like to do?",
    ["Study", "Generate Quiz"],
    horizontal=True,
    key="ui_mode",
)

current_result = st.session_state.get("generated_result")
if current_result and current_result.get("mode") != mode:
    st.session_state.pop("generated_result", None)

with st.form("quiz_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        subject = st.selectbox(
            "Subject focus",
            ["Science", "Math", "Computer Science"],
            index=0,
        )
        chapter = st.text_input(
            "Chapter or topic",
            placeholder="For example: Newton's Laws / Integrals / Recursion",
        )

    with col_right:
        uploaded_pdf = st.file_uploader(
            "Upload a PDF for this subject",
            type=["pdf"],
        )
        if mode == "Generate Quiz":
            num_questions = st.slider(
                "Number of questions",
                min_value=3,
                max_value=15,
                value=6,
                step=1,
            )
        else:
            num_questions = 6

        if mode == "Generate Quiz":
            difficulty = st.select_slider(
                "Difficulty",
                options=["easy", "medium", "hard"],
                value="medium",
            )
        else:
            difficulty = "medium"

        if mode == "Study":
            study_depth = st.select_slider(
                "Study depth",
                options=["quick", "standard", "deep"],
                value="standard",
            )
        else:
            study_depth = "standard"

    action_label = {
        "Study": "Build Study Guide",
        "Generate Quiz": "Generate Quiz",
    }[mode]
    submitted = st.form_submit_button(action_label)


if submitted:
    if not uploaded_pdf:
        st.error("Please upload a PDF file before continuing.")
    elif not chapter.strip():
        st.error("Please enter a chapter or topic focus.")
    else:
        try:
            pdf_path = save_uploaded_pdf(uploaded_pdf)

            with st.status("Building subject knowledge base from your file...", expanded=False):
                agent = initialize_uploaded_agent(pdf_path=pdf_path, subject=subject)

            if mode == "Study":
                with st.spinner("Creating study guide..."):
                    rag_context, study_guide = build_study_content(
                        agent=agent,
                        subject=subject,
                        chapter=chapter,
                        study_depth=study_depth,
                    )

                st.session_state.generated_result = {
                    "mode": mode,
                    "subject": subject,
                    "chapter": chapter,
                    "rag_context": rag_context,
                    "study_guide": study_guide,
                }

            elif mode == "Generate Quiz":
                with st.spinner("Generating chapter-focused quiz..."):
                    rag_context, quiz_output = build_quiz_content(
                        agent=agent,
                        subject=subject,
                        chapter=chapter,
                        num_questions=num_questions,
                        difficulty=difficulty,
                    )

                st.session_state.generated_result = {
                    "mode": mode,
                    "subject": subject,
                    "chapter": chapter,
                    "rag_context": rag_context,
                    "quiz_data": quiz_output,
                }

            st.success(f"{mode} content generated successfully.")

        except Exception as exc:
            st.exception(exc)


result = st.session_state.get("generated_result")

if result:
    st.markdown("---")
    st.markdown(f"## {result['mode']} Output")

    context_col, output_col = st.columns(2)

    with context_col:
        st.markdown("### RAG Context")
        st.text_area(
            "Retrieved context used for generation",
            value=result["rag_context"],
            height=420,
        )

    with output_col:
        if result["mode"] == "Study":
            st.markdown("### Study Guide")
            st.markdown(result["study_guide"])

        elif result["mode"] == "Generate Quiz":
            quiz_data = result["quiz_data"]
            questions = quiz_data.get("questions", [])
            st.markdown("### Quiz Test")
            st.caption(quiz_data.get("title", "Quiz test"))

            if not questions and quiz_data.get("raw_output"):
                st.markdown(quiz_data["raw_output"])
                st.info("This session is using a fallback quiz response. Restarting Streamlit will load the structured quiz test mode.")
                st.stop()

            quiz_key = quiz_session_key(result)
            quiz_data["session_key"] = quiz_key

            with st.form(f"self_test_form_{quiz_key}"):
                for question in questions:
                    st.radio(
                        f"Q{question['id']}. {question['question']}",
                        question["options"],
                        key=f"{quiz_key}_answer_{question['id']}",
                    )

                check_answers = st.form_submit_button("Check My Answers")

            if check_answers:
                score, total, results = score_self_test(quiz_data)
                st.metric("Score", f"{score}/{total}")

                for item in results:
                    question = item["question"]
                    st.markdown(f"**Q{question['id']}. {question['question']}**")
                    if item["is_correct"]:
                        st.success("Correct")
                    else:
                        st.error(f"Incorrect. Correct answer: {item['correct_option']}")
                    st.caption(question.get("explanation", ""))

            else:
                st.info("Choose an answer for each question and click Check My Answers when you are ready.")
