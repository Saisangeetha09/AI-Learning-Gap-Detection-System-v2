import os
import re
#import chromadb
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rag_pipeline import load_chroma
from database import *

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Learning Gap Detection System",
    page_icon=".",
    layout="wide"
)

# =========================================
# LOAD ENV VARIABLES
# =========================================
load_dotenv()

# =========================================
# OPENROUTER CLIENT
# =========================================
client_ai = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# =========================================
# CACHE EMBEDDING MODEL
# =========================================
@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# =========================================
# CACHE CHROMADB CLIENT
# =========================================
@st.cache_resource
def load_chroma():

    return chromadb.PersistentClient(path="chroma_db")

client = load_chroma()

#collection = client.get_or_create_collection(name="skills")

# =========================================
# LOAD DATASET
# =========================================
df = pd.read_csv("data/skills_dataset.csv")

topics = sorted(df["skill_name"].unique())

# =========================================
# TITLE
# =========================================
st.title("AI Learning Gap Detection System")

st.markdown("""
Adaptive AI Learning & Skill Assessment Platform
""")

# =========================================
# SESSION STATE
# =========================================
if "mcqs" not in st.session_state:
    st.session_state.mcqs = None

if "answers" not in st.session_state:
    st.session_state.answers = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "show_quiz" not in st.session_state:
    st.session_state.show_quiz = False

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("Profile")

# =========================================
# AUTHENTICATION
# =========================================
if not st.session_state.logged_in:

    auth_option = st.sidebar.radio(
        "Select Option",
        ["Login", "Signup"]
    )

    username = st.sidebar.text_input(
        "Username"
    )

    password = st.sidebar.text_input(
        "Password",
        type="password"
    )

    # =========================================
    # LOGIN
    # =========================================
    if auth_option == "Login":

        if st.sidebar.button("Login"):

            user = login_user(
                username,
                password
            )

            if user:

                st.session_state.logged_in = True
                st.session_state.username = username

                st.success("Login Successful!")

                st.rerun()

            else:

                st.error("Invalid Credentials")

    # =========================================
    # SIGNUP
    # =========================================
    else:

        if st.sidebar.button("Signup"):

            success = add_user(
                username,
                password
            )

            if success:

                st.success(
                    "Account Created Successfully!"
                )

            else:

                st.error(
                    "Username Already Exists"
                )

# =========================================
# AFTER LOGIN
# =========================================
else:

    st.sidebar.success(
        f"Welcome {st.session_state.username}"
    )

    # =========================================
    # LOGOUT
    # =========================================
    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.show_quiz = False

        st.rerun()

    # =========================================
    # GET SCORES
    # =========================================
    scores = get_scores(
        st.session_state.username
    )

    analytics = get_user_analytics(
        st.session_state.username
    )

    # =========================================
    # ANALYTICS
    # =========================================
    if analytics:

        avg_score = analytics[0]
        best_score = analytics[1]
        total_attempts = analytics[2]

        st.sidebar.markdown("## Analytics")

        st.sidebar.write(
            f"Total Number Of Tests Taken: {total_attempts}"
        )

        if best_score is not None:

            st.sidebar.write(
                f"Best Score: {best_score}"
            )

        else:

            st.sidebar.write(
                "Best Score: 0"
            )

        if avg_score is not None:

            st.sidebar.write(
                f"Average Score: {round(avg_score, 2)}"
            )

        else:

            st.sidebar.write(
                "Average Score: 0"
            )

    # =========================================
    # QUIZ HISTORY
    # =========================================
    if scores:

        st.sidebar.markdown("## Quiz History")

        for s in scores:

            st.sidebar.write(
                f"{s[0]} | {s[1]} | {s[2]}/{s[3]}"
            )

    # =========================================
    # TAKE QUIZ BUTTON
    # =========================================
    if st.sidebar.button("Take Quiz"):

        st.session_state.show_quiz = True

    # =========================================
    # SHOW DASHBOARD
    # =========================================
    if not st.session_state.show_quiz:

        st.subheader(
            f" Welcome {st.session_state.username}"
        )

        st.info(
            "Click 'Take Quiz' in sidebar "
            "to start assessment."
        )

    # =========================================
    # SHOW QUIZ
    # =========================================
    if st.session_state.show_quiz:

        # =========================================
        # TOPIC DROPDOWN
        # =========================================
        topic = st.sidebar.selectbox(
            "Select Topic",
            topics
        )

        # =========================================
        # DIFFICULTY
        # =========================================
        difficulty = st.sidebar.selectbox(
            "Select Difficulty",
            ["Beginner", "Intermediate", "Advanced"]
        )

        # =========================================
        # QUESTION COUNT
        # =========================================
        question_count = st.sidebar.slider(
            "Number of Questions",
            5,
            20,
            10
        )

        # =========================================
        # RETRIEVE CONTEXT
        # =========================================
        def retrieve_context(topic):

            query_embedding = embedding_model.encode(
                [topic]
            ).tolist()[0]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            docs = results["documents"][0]

            return "\n".join(docs)

        # =========================================
        # GENERATE MCQS
        # =========================================
        def generate_mcqs(
            topic,
            difficulty,
            question_count
        ):

            context = retrieve_context(topic)

            prompt = f"""
            You are an AI tutor.

            Using the following context:

            {context}

            Generate EXACTLY
            {question_count}
            {difficulty}
            level MCQs for {topic}.

            RULES:
            - Each question must start with number
            - Put each option on new line
            - Mention correct answer

            FORMAT:

            1. Question

            A) Option
            B) Option
            C) Option
            D) Option

            Correct Answer: A
            """

            response = client_ai.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            text = response.choices[0].message.content

            # Extract answers
            answers = re.findall(
                r"Correct Answer:\s*([A-D])",
                text
            )

            # Remove answers
            clean_text = re.sub(
                r"Correct Answer:\s*[A-D]",
                "",
                text
            )

            return clean_text, answers

        # =========================================
        # GENERATE ROADMAP
        # =========================================
        def generate_roadmap(
            topic,
            weak_areas
        ):

            prompt = f"""
            Student is weak in:

            {weak_areas}

            Generate personalized roadmap
            for improving {topic}.

            Use bullet points.
            """

            response = client_ai.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.choices[0].message.content

        # =========================================
        # GENERATE BUTTON
        # =========================================
        if st.sidebar.button("Generate MCQs"):

            with st.spinner(
                "Generating Quiz..."
            ):

                mcqs, answers = generate_mcqs(
                    topic,
                    difficulty,
                    question_count
                )

                st.session_state.mcqs = mcqs
                st.session_state.answers = answers

        # =========================================
        # DISPLAY QUIZ
        # =========================================
        if st.session_state.mcqs:

            st.subheader("Generated Quiz")

            questions = re.split(
                r'\n(?=\d+\.)',
                st.session_state.mcqs.strip()
            )

            user_answers = []

            q_count = 0

            for q in questions:

                if "A)" in q:

                    q_count += 1

                    lines = [
                        line.strip()
                        for line in q.split("\n")
                        if line.strip()
                    ]

                    question = ""

                    options = []

                    for line in lines:

                        line = line.strip()

                        # Detect question
                        if re.match(
                            r"^\d+\.",
                            line
                        ):
                            question = line

                        # Backup detection
                        elif not line.startswith(
                            (
                                "A)",
                                "B)",
                                "C)",
                                "D)"
                            )
                        ) and question == "":
                            question = line

                        # Detect options
                        elif line.startswith(
                            (
                                "A)",
                                "B)",
                                "C)",
                                "D)"
                            )
                        ):
                            options.append(line)

                    # Show question
                    if question:

                        st.markdown(
                            f"### {question}"
                        )

                    else:

                        st.markdown(
                            f"### Question {q_count}"
                        )

                    answer = st.radio(
                        f"Select answer for Question {q_count}",
                        options,
                        key=f"q_{q_count}"
                    )

                    user_answers.append(
                        answer[0]
                    )

            # =========================================
            # SUBMIT QUIZ
            # =========================================
            if st.button("Submit Quiz"):

                score = 0

                weak_areas = []

                correct_answers = (
                    st.session_state.answers
                )

                for i in range(
                    len(correct_answers)
                ):

                    if (
                        user_answers[i]
                        ==
                        correct_answers[i]
                    ):

                        score += 1

                    else:

                        # Store actual weak question
                        weak_areas.append(
                            questions[i]
                        )

                # =========================================
                # SAVE SCORE
                # =========================================
                save_score(
                    st.session_state.username,
                    topic,
                    difficulty,
                    score,
                    question_count
                )

                # =========================================
                # SCORE
                # =========================================
                st.success(
                    f"Score: "
                    f"{score}/{question_count}"
                )

                # =========================================
                # PERFORMANCE PIE CHART
                # =========================================
                fig, ax = plt.subplots(
                    figsize=(3, 3)
                )

                labels = [
                    "Correct",
                    "Wrong"
                ]

                values = [
                    score,
                    question_count - score
                ]

                colors = [
                    "#57d775",   # green
                    "#ed5b69"    # red
                ]

                # Small explode effect
                explode = (0.05, 0.05)

                ax.pie(
                    values,
                    labels=labels,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=explode,
                    textprops={
                        'fontsize': 10
                    }
                )

                ax.axis('equal')

                # REMOVE extra white space
                plt.tight_layout()

                # Display SMALLER chart
                st.pyplot(
                    fig,
                    use_container_width=False
                )
                # =========================================
                # WEAK AREAS
                # =========================================
                st.subheader(
                    "Weak Areas"
                )

                if weak_areas:

                    for area in weak_areas:

                        # Extract question only
                        weak_topic = area.split("\n")[0]

                        st.write(f"• {weak_topic}")
                else:

                    st.write(
                        "Excellent Performance!"
                    )

                # =========================================
                # ROADMAP
                # =========================================
                st.subheader(
                    "🗺️ Personalized Learning Roadmap"
                )

                roadmap = generate_roadmap(
                    topic,
                    ", ".join(weak_areas)
                )

                # =========================================
                # SHOW ROADMAP TEXT
                # =========================================
                st.markdown(roadmap)

                # =========================================
                # FLOWCHART ROADMAP
                # =========================================
                st.subheader(
                    "📌 Learning Flowchart"
                )

                steps = roadmap.split("\n")

                flow = Digraph()

                flow.attr(
                    rankdir='TB'
                )

                previous_node = None

                for i, step in enumerate(steps):

                    clean_step = (
                        step
                        .replace("-", "")
                        .replace("•", "")
                        .strip()
                    )

                    if clean_step != "":

                        node_name = f"Step_{i}"

                        flow.node(
                            node_name,
                            clean_step,
                            style="filled",
                            fillcolor="lightblue",
                            shape="box"
                        )

                        if previous_node:

                            flow.edge(
                                previous_node,
                                node_name
                            )

                        previous_node = node_name

                st.graphviz_chart(flow)
                # =========================================
                # RESOURCE RECOMMENDATIONS
                # =========================================
                st.subheader("📚 Recommended Resources")

                resource_data = df[
                    df["skill_name"] == topic
                ]

                if not resource_data.empty:

                    resource_link = resource_data.iloc[0][
                        "resource_link"
                    ]

                    st.markdown(
                        f"[Open Learning Resource]({resource_link})"
                    )

# =========================================
# FOOTER
# =========================================
st.markdown("---")

st.markdown(
    """AI-Powered Adaptive Learning Platform  
Built with Streamlit • ChromaDB • SQLite • OpenRouter"""
)