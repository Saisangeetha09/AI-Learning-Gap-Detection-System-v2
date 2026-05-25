import sqlite3

# =========================================
# CONNECT DATABASE
# =========================================
conn = sqlite3.connect(
    "learning_gap.db",
    check_same_thread=False
)

cursor = conn.cursor()

# =========================================
# CREATE USERS TABLE
# =========================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# =========================================
# CREATE SCORES TABLE
# =========================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS scores (

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    topic TEXT,
    difficulty TEXT,
    score INTEGER,
    total_questions INTEGER
)
""")

conn.commit()

# =========================================
# ADD USER
# =========================================
def add_user(username, password):

    try:

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )

        conn.commit()

        return True

    except:

        return False

# =========================================
# LOGIN USER
# =========================================
def login_user(username, password):

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    data = cursor.fetchone()

    return data

# =========================================
# SAVE SCORE
# =========================================
def save_score(
    username,
    topic,
    difficulty,
    score,
    total_questions
):

    cursor.execute(
        """
        INSERT INTO scores (
            username,
            topic,
            difficulty,
            score,
            total_questions
        )

        VALUES (?, ?, ?, ?, ?)
        """,

        (
            username,
            topic,
            difficulty,
            score,
            total_questions
        )
    )

    conn.commit()

# =========================================
# GET USER SCORES
# =========================================
def get_scores(username):

    cursor.execute(
        """
        SELECT
            topic,
            difficulty,
            score,
            total_questions

        FROM scores

        WHERE username=?
        """,

        (username,)
    )

    data = cursor.fetchall()

    return data
# =========================================
# GET ANALYTICS
# =========================================
def get_user_analytics(username):

    cursor.execute(
        """
        SELECT
            AVG(score),
            MAX(score),
            COUNT(*)

        FROM scores

        WHERE username=?
        """,

        (username,)
    )

    data = cursor.fetchone()

    return data