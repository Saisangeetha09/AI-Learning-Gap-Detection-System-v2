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

    username TEXT PRIMARY KEY,
    password TEXT
)
""")

# =========================================
# CREATE SCORES TABLE
# =========================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS scores (

    username TEXT,
    topic TEXT,
    difficulty TEXT,
    score INTEGER,
    total INTEGER
)
""")

conn.commit()

# =========================================
# ADD USER
# =========================================
def add_user(username, password):

    try:

        cursor.execute(
            "INSERT INTO users VALUES (?, ?)",
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
    total
):

    cursor.execute(
        "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
        (
            username,
            topic,
            difficulty,
            score,
            total
        )
    )

    conn.commit()

# =========================================
# GET SCORES
# =========================================
def get_scores(username):

    cursor.execute(
        "SELECT topic, difficulty, score, total FROM scores WHERE username=?",
        (username,)
    )

    return cursor.fetchall()

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

    return cursor.fetchone()