import os
import re
import chromadb
import gradio as gr
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from quiz_data import correct_answers

# Load environment variables
load_dotenv()

# OpenRouter Client
client_ai = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB connection
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="skills")


# Retrieve Context
def retrieve_context(topic):

    query_embedding = embedding_model.encode([topic]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_docs = results["documents"][0]

    context = "\n".join(retrieved_docs)

    return context


# Generate MCQs
def generate_mcqs(topic, difficulty):

    try:

        context = retrieve_context(topic)

        prompt = f"""
        You are an AI tutor.

        Using the following context:

        {context}

        Generate 5 {difficulty} level multiple choice questions for the topic: {topic}

        Format strictly:

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

        mcqs = response.choices[0].message.content

        # Extract answers
        answers = re.findall(r"Correct Answer:\s*([A-D])", mcqs)

        correct_answers.clear()

        for i, ans in enumerate(answers):
            correct_answers[str(i + 1)] = ans

        # Remove answers from UI
        clean_mcqs = re.sub(r"Correct Answer:\s*[A-D]", "", mcqs)

        return clean_mcqs

    except Exception as e:

        return f"ERROR: {str(e)}"


# Generate Roadmap
def generate_roadmap(weak_areas, topic):

    prompt = f"""
    A student is weak in the following areas:

    {weak_areas}

    Generate a personalized learning roadmap for improving skills in {topic}.

    Keep it beginner friendly and step-by-step.
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


# Evaluate Answers
def evaluate_answers(user_answers, topic):

    try:

        user_answers = user_answers.upper()

        pairs = user_answers.split(",")

        score = 0

        weak_areas = []

        for pair in pairs:

            q_no, ans = pair.split(":")

            q_no = q_no.strip()
            ans = ans.strip()

            correct = correct_answers.get(q_no)

            if ans == correct:
                score += 1
            else:
                weak_areas.append(f"Question {q_no}")

        weak_text = ", ".join(weak_areas) if weak_areas else "None"

        roadmap = generate_roadmap(weak_text, topic)

        # Score Chart
        plt.figure(figsize=(4, 4))

        labels = ["Correct", "Wrong"]
        values = [score, 5 - score]

        plt.bar(labels, values)

        chart_path = "score_chart.png"

        plt.savefig(chart_path)

        result = f"""
Score: {score}/5

Weak Areas:
{weak_text}

Personalized Learning Roadmap:
{roadmap}
"""

        return result, chart_path

    except Exception as e:

        return f"ERROR: {str(e)}", None


# Gradio UI
with gr.Blocks() as demo:

    gr.Markdown("# Learning Gap Detection System")

    with gr.Row():

        topic_input = gr.Textbox(
            label="Enter Topic",
            placeholder="Example: Python"
        )

        difficulty_input = gr.Dropdown(
            choices=["Beginner", "Intermediate", "Advanced"],
            value="Beginner",
            label="Difficulty Level"
        )

    generate_button = gr.Button("Generate MCQs")

    mcq_output = gr.Textbox(
        label="Generated MCQs",
        lines=20
    )

    generate_button.click(
        fn=generate_mcqs,
        inputs=[topic_input, difficulty_input],
        outputs=mcq_output
    )

    user_answers = gr.Textbox(
        label="Enter Your Answers",
        placeholder="Example: 1:A,2:B,3:C,4:D,5:A"
    )

    evaluate_button = gr.Button("Evaluate Answers")

    evaluation_output = gr.Textbox(
        label="Evaluation Result",
        lines=12
    )

    score_chart = gr.Image(
        label="Performance Chart"
    )

    evaluate_button.click(
        fn=evaluate_answers,
        inputs=[user_answers, topic_input],
        outputs=[evaluation_output, score_chart]
    )

# Launch app
demo.launch()