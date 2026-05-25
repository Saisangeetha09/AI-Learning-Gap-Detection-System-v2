import os
import re
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph
from dotenv import load_dotenv
from openai import OpenAI
from database import *

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Learning Gap Detection System",
    page_icon="🎓",
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
            f"Total Tests Taken: {total_attempts}"
        )

        st.sidebar.write(
            f"Best Score: {best_score if best_score else 0}"
        )

        st.sidebar.write(
            f"Average Score: {round(avg_score, 2) if avg_score else 0}"
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
    # DASHBOARD
    # =========================================
    if not st.session_state.show_quiz:

        st.subheader(
            f"Welcome {st.session_state.username}"
        )

        st.info(
            "Click 'Take Quiz' in sidebar to start assessment."
        )

    # =========================================
    # SHOW QUIZ
    # =========================================
    if st.session_state.show_quiz:

        topic = st.sidebar.selectbox(
            "Select Topic",
            topics
        )

        difficulty = st.sidebar.selectbox(
            "Select Difficulty",
            ["Beginner", "Intermediate", "Advanced"]
        )

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

            filtered = df[
                df["skill_name"].str.contains(
                    topic,
                    case=False,
                    na=False
                )
            ]

            if filtered.empty:
                filtered = df.head(5)

            context = ""

            for _, row in filtered.iterrows():

                context += f"""
                Skill: {row['skill_name']}
                Category: {row['category']}
                Difficulty: {row['difficulty_level']}
                Prerequisites: {row['prerequisites']}
                Learning Time: {row['learning_time_days']} days
                """

            return context

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

            answers = re.findall(
                r"Correct Answer:\s*([A-D])",
                text
            )

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

            with st.spinner("Generating Quiz..."):

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

                        if re.match(r"^\d+\.", line):
                            question = line

                        elif line.startswith(
                            ("A)", "B)", "C)", "D)")
                        ):
                            options.append(line)

                    st.markdown(f"### {question}")

                    answer = st.radio(
                        f"Select answer for Question {q_count}",
                        options,
                        key=f"q_{q_count}"
                    )

                    user_answers.append(answer[0])

            # =========================================
            # SUBMIT QUIZ
            # =========================================
            if st.button("Submit Quiz"):

                score = 0
                weak_areas = []

                correct_answers = st.session_state.answers

                for i in range(len(correct_answers)):

                    if user_answers[i] == correct_answers[i]:

                        score += 1

                    else:

                        weak_areas.append(
                            questions[i]
                        )

                save_score(
                    st.session_state.username,
                    topic,
                    difficulty,
                    score,
                    question_count
                )

                st.success(
                    f"Score: {score}/{question_count}"
                )

                # =========================================
                # PIE CHART
                # =========================================
                fig, ax = plt.subplots(
                    figsize=(3, 3)
                )

                labels = ["Correct", "Wrong"]

                values = [
                    score,
                    question_count - score
                ]

                colors = [
                    "#57d775",
                    "#ed5b69"
                ]

                explode = (0.05, 0.05)

                ax.pie(
                    values,
                    labels=labels,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=explode
                )

                ax.axis('equal')

                st.pyplot(
                    fig,
                    use_container_width=False
                )

                # =========================================
                # WEAK AREAS
                # =========================================
                st.subheader("Weak Areas")

                if weak_areas:

                    for area in weak_areas:

                        weak_topic = area.split("\n")[0]

                        st.write(f"• {weak_topic}")

                else:

                    st.write("Excellent Performance!")

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

                st.markdown(roadmap)

                # =========================================
                # FLOWCHART
                # =========================================
                st.subheader("📌 Learning Flowchart")

                steps = roadmap.split("\n")

                flow = Digraph()

                flow.attr(rankdir='TB')

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
                # RESOURCES
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
Built with Streamlit • SQLite • OpenRouter"""
)