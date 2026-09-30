import os
import re
import sqlite3
import uuid
import time
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    json,
    render_template,
    render_template_string,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    send_from_directory,
    flash
)

from werkzeug.utils import secure_filename
from pypdf import PdfReader
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename


# ============================================================
# OPTIONAL PACKAGES
# ============================================================

try:
    import ollama
except Exception:
    ollama = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "ai-courtroom-assistant-secret-key"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(
    BASE_DIR,
    "courtroom.db"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

DOCUMENT_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "documents"
)

EVIDENCE_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "evidence"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)

for folder in [
    UPLOAD_FOLDER,
    DOCUMENT_FOLDER,
    EVIDENCE_FOLDER,
    REPORT_FOLDER
]:
    os.makedirs(folder, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ============================================================
# SETTINGS
# ============================================================

ALLOWED_EXTENSIONS = {
    "pdf",
    "doc",
    "docx",
    "txt",
    "png",
    "jpg",
    "jpeg"
}

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen3:4b"
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db


def now():
    return datetime.now().isoformat(timespec="seconds")


def init_database():

    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            case_id TEXT NOT NULL,
            case_name TEXT NOT NULL,
            case_type TEXT,
            status TEXT DEFAULT 'Active',
            next_hearing TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            case_id TEXT,
            filename TEXT NOT NULL,
            original_name TEXT,
            file_type TEXT,
            description TEXT,
            uploaded_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            case_id TEXT,
            title TEXT NOT NULL,
            description TEXT,
            evidence_type TEXT,
            priority TEXT DEFAULT 'Medium',
            filename TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hearings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            case_id TEXT,
            hearing_date TEXT,
            hearing_time TEXT,
            court TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            case_id TEXT,
            event_title TEXT,
            event_date TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
    """)



    try:
         db.execute("ALTER TABLE cases ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
          pass 

    db.commit()

    # --------------------------------------------------------
    # SAFE MIGRATION FOR OLD DATABASE
    # --------------------------------------------------------

    migrations = [
        ("documents", "original_name", "TEXT"),
        ("documents", "file_type", "TEXT"),
        ("documents", "description", "TEXT"),
        ("evidence", "priority", "TEXT DEFAULT 'Medium'"),
        ("evidence", "filename", "TEXT"),
        ("cases", "description", "TEXT")
    ]

    for table, column, definition in migrations:
        try:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )
            db.commit()
        except sqlite3.OperationalError:
            pass

    db.close()


init_database()


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


# ============================================================
# HELPERS
# ============================================================

def current_user_id():
    return session.get("user_id")


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def get_file_text(filepath):

    extension = filepath.rsplit(".", 1)[-1].lower()

    try:

        if extension == "txt":

            with open(
                filepath,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                return file.read()

        if extension == "pdf" and PdfReader:

            reader = PdfReader(filepath)

            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            return text

        if extension == "docx" and Document:

            document = Document(filepath)

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

    except Exception:
        return ""

    return ""

def ask_ai_model(question):

    if not question:
        return "Please enter your question."

    try:
        import ollama

        print("AI QUESTION RECEIVED:", question)
        print("CALLING FAST OLLAMA...")

        response = ollama.chat(
    model="qwen2.5:0.5b",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer legal questions in simple English. "
                "Give only 2 or 3 short sentences. "
                "No reasoning. Information only."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ],
    think=False,
    keep_alive="10m",
    options={
        "temperature": 0.1,
        "num_predict": 50,
        "num_ctx": 512
    }
)
        print("FAST OLLAMA RESPONSE RECEIVED")

        answer = response["message"]["content"]

        return answer.strip()

    except Exception as e:

        print("OLLAMA ERROR:", e)

        return "AI Error: " + str(e)



# ============================================================
# LEGAL SECTION RECOMMENDATION
# ============================================================

def recommend_legal_sections(text):

    text = (text or "").strip()

    if not text:
        return [{
            "section": "No information provided",
            "reason": "Please enter a case description."
        }]

    try:

        import ollama

        prompt = f"""
You are an Indian legal information assistant.

Analyze this case description:

{text}

Identify potentially relevant Indian legal provisions.

Rules:
- Use BNS provisions where applicable.
- Do not invent section numbers.
- Give short reasons.
- If information is insufficient, clearly mention it.
- Provide informational suggestions only.
- Keep the answer simple.

Format:
Section: [name and number]
Reason: [short explanation]
"""

        response = ollama.chat(
            model="qwen3:4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful legal information "
                        "assistant. Do not invent legal sections."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 300
            }
        )

        answer = response["message"]["content"].strip()

        if not answer:

            return [{
                "section": "No recommendation available",
                "reason": "AI returned an empty response."
            }]

        return [{
            "section": "AI Legal Analysis",
            "reason": answer
        }]

    except Exception as e:

        print("LEGAL SECTION AI ERROR:", e)

        return [{
            "section": "AI recommendation unavailable",
            "reason": (
                "Please make sure Ollama is running "
                "and qwen3:4b is available."
            )
        }]

# ============================================================
# HOME - OPEN DASHBOARD FIRST
# ============================================================

@app.route("/")
def home():
    return redirect(url_for("dashboard"))

# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            flash(
                "Username and password are required.",
                "error"
            )

            return render_template("login.html")

        db = get_db()

        user = db.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
               OR email = ?
            """,
            (
                username,
                username
            )
        ).fetchone()

        db.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session.clear()

            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid username/email or password.",
            "error"
        )

    return render_template("login.html")




    # ============================================================
# ACCOUNT RESET - TEMPORARY
# ============================================================

@app.route("/reset-account", methods=["GET", "POST"])
def reset_account():

    message = ""

    if request.method == "POST":

        old_username = request.form.get(
            "old_username", ""
        ).strip()

        new_username = request.form.get(
            "new_username", ""
        ).strip()

        new_email = request.form.get(
            "new_email", ""
        ).strip()

        new_password = request.form.get(
            "new_password", ""
        )

        if not new_username or not new_email or not new_password:

            message = "Please fill all fields."

            return render_template_string("""
                <h2>Account Reset</h2>
                <p style="color:red;">{{ message }}</p>

                <form method="POST">

                    <input
                        name="old_username"
                        placeholder="Old username"
                        required
                    ><br><br>

                    <input
                        name="new_username"
                        placeholder="New username"
                        required
                    ><br><br>

                    <input
                        name="new_email"
                        type="email"
                        placeholder="New email"
                        required
                    ><br><br>

                    <input
                        name="new_password"
                        type="password"
                        placeholder="New password"
                        required
                    ><br><br>

                    <button type="submit">
                        Reset Account
                    </button>

                </form>
            """, message=message)

        db = get_db()

        try:

            if old_username:

                user = db.execute(
                    """
                    SELECT id
                    FROM users
                    WHERE username = ?
                    OR email = ?
                    """,
                    (
                        old_username,
                        old_username
                    )
                ).fetchone()

            else:

                user = db.execute(
                    """
                    SELECT id
                    FROM users
                    LIMIT 1
                    """
                ).fetchone()

            if not user:

                message = "No existing account found."

            else:

                db.execute(
                    """
                    UPDATE users
                    SET username = ?,
                        email = ?,
                        password = ?
                    WHERE id = ?
                    """,
                    (
                        new_username,
                        new_email,
                        generate_password_hash(new_password),
                        user["id"]
                    )
                )

                db.commit()

                session["user_id"] = user["id"]
                session["username"] = new_username

                db.close()

                return redirect(
                    url_for("dashboard")
                )

        except sqlite3.IntegrityError:

            message = (
                "That username or email is already "
                "used by another account."
            )

        finally:

            db.close()

    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Account Reset</title>

            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f4f6fb;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }

                .box {
                    background: white;
                    padding: 35px;
                    width: 380px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
                }

                h2 {
                    text-align: center;
                    margin-bottom: 25px;
                }

                input {
                    width: 100%;
                    padding: 12px;
                    margin-bottom: 12px;
                    box-sizing: border-box;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }

                button {
                    width: 100%;
                    padding: 12px;
                    border: none;
                    border-radius: 8px;
                    background: #4f46e5;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                }

                .error {
                    color: red;
                    text-align: center;
                }

                .info {
                    font-size: 13px;
                    color: #666;
                    margin-bottom: 15px;
                }
            </style>
        </head>

        <body>

            <div class="box">

                <h2>Reset Account</h2>

                {% if message %}
                    <p class="error">{{ message }}</p>
                {% endif %}

                <p class="info">
                    Enter your old username/email if you know it.
                    Then create your new login details.
                </p>

                <form method="POST">

                    <input
                        type="text"
                        name="old_username"
                        placeholder="Old username or email"
                    >

                    <input
                        type="text"
                        name="new_username"
                        placeholder="New username"
                        required
                    >

                    <input
                        type="email"
                        name="new_email"
                        placeholder="New email"
                        required
                    >

                    <input
                        type="password"
                        name="new_password"
                        placeholder="New password"
                        required
                    >

                    <button type="submit">
                        Reset Account
                    </button>

                </form>

            </div>

        </body>
        </html>
    """, message=message)


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            flash(
                "Username and password are required.",
                "error"
            )

            return render_template("register.html")

        db = get_db()

        try:

            db.execute(
                """
                INSERT INTO users
                (
                    username,
                    email,
                    password,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    generate_password_hash(password),
                    now()
                )
            )

            db.commit()

            flash(
                "Registration successful. Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            flash(
                "Username or email already exists.",
                "error"
            )

        finally:
            db.close()

    return render_template("register.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    user_id = session.get("user_id")

    cases = []
    documents = []
    evidence = []
    hearings = []
    unread = 0

    case_counts = {
        "total": 0,
        "filed": 0,
        "hearing": 0,
        "pending": 0,
        "closed": 0
    }

    if user_id:
        db = get_db()

        case_counts = db.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE
    WHEN LOWER(status) IN ('filed', 'active')
    THEN 1 ELSE 0
END) AS filed,

SUM(CASE
    WHEN LOWER(status) IN ('hearing', 'active')
    THEN 1 ELSE 0
END) AS hearing,

SUM(CASE
    WHEN LOWER(status) IN ('pending', 'active')
    THEN 1 ELSE 0
END) AS pending,
SUM(CASE
    WHEN LOWER(status) = 'closed'
    THEN 1 ELSE 0
END) AS closed
            FROM cases
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()

        cases = db.execute(
            """
            SELECT *
            FROM cases
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (user_id,)
        ).fetchall()

        documents = db.execute(
            """
            SELECT *
            FROM documents
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (user_id,)
        ).fetchall()

        evidence = db.execute(
            """
            SELECT *
            FROM evidence
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (user_id,)
        ).fetchall()

        hearings = db.execute(
            """
            SELECT
                hearings.*,
                cases.case_name
            FROM hearings
            LEFT JOIN cases
                ON hearings.case_id = cases.case_id
                AND hearings.user_id = cases.user_id
            WHERE hearings.user_id = ?
            AND hearings.hearing_date >= date('now')
            ORDER BY
                hearings.hearing_date ASC,
                hearings.hearing_time ASC
            LIMIT 10
            """,
            (user_id,)
        ).fetchall()

        unread = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM notifications
            WHERE user_id = ?
            AND is_read = 0
            """,
            (user_id,)
        ).fetchone()["total"]

        db.close()

    stats = {
        "active_cases": sum(
            1 for case in cases
            if case["status"] != "Closed"
        ),
        "documents": len(documents),
        "evidence": len(evidence),
        "notifications": unread,
        "hearings": len(hearings)
    }

    return render_template(
        "dashboard.html",
        cases=cases,
        documents=documents,
        evidence=evidence,
        hearings=hearings,
        stats=stats,
        notifications_count=unread,
        username=session.get("username", "Suwetha"),
        case_counts=case_counts
    )

# ============================================================
# PROJECT
# ============================================================

@app.route("/project")
@login_required
def project():

    try:
        return render_template("project.html")
    except Exception:
        return redirect(url_for("dashboard"))


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
@login_required
def profile():

    db = get_db()

    user = db.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (current_user_id(),)
    ).fetchone()

    db.close()

    return render_template(
        "profile.html",
        user=user
    )


@app.route(
    "/profile/update",
    methods=["POST"]
)
@login_required
def update_profile():

    username = request.form.get(
        "username",
        ""
    ).strip()

    email = request.form.get(
        "email",
        ""
    ).strip()

    if not username:

        flash(
            "Username cannot be empty.",
            "error"
        )

        return redirect(
            url_for("profile")
        )

    db = get_db()

    try:

        db.execute(
            """
            UPDATE users
            SET username = ?,
                email = ?
            WHERE id = ?
            """,
            (
                username,
                email,
                current_user_id()
            )
        )

      

        db.commit()

        session["username"] = username

        flash(
            "Profile updated successfully.",
            "success"
        )

    except sqlite3.IntegrityError:

        flash(
            "Username or email is already in use.",
            "error"
        )

    finally:
        db.close()

    return redirect(
        url_for("profile")
    )


# ============================================================
# ASK AI ASSISTANT
# ============================================================

@app.route("/ask", methods=["GET", "POST"])
@login_required
def ask():

    question = ""
    answer = None

    if request.method == "POST":

        start_time = time.time()

        data = request.get_json(silent=True) or request.form

        question = data.get(
            "question",
            data.get("text", "")
        ).strip()

        if not question:
            return jsonify({
                "success": False,
                "answer": "Please enter your question."
            }), 400

        print("QUESTION RECEIVED:", question)

        answer = ask_ai_model(question)

        end_time = time.time()

        print("AI RESPONSE TIME:",
              round(end_time - start_time, 2),
              "seconds")

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer
        })

    return render_template(
        "ask_ai.html",
        question=question,
        answer=answer
    )
# ============================================================
# LEGAL DOCUMENT
# ============================================================

@app.route(
    "/legal_document",
    methods=["GET", "POST"]
)
@app.route(
    "/legal-documents",
    methods=["GET", "POST"]
)
@login_required
def legal_documents():

    db = get_db()

    documents = db.execute(
        """
        SELECT *
        FROM documents
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (current_user_id(),)
    ).fetchall()

    db.close()

    return render_template(
        "legal_document.html",
        documents=documents
    )


# IMPORTANT:
# This endpoint is kept exactly as "legal_document"
# for templates using url_for("legal_document").

@app.route(
    "/legal_document_page"
)
@login_required
def legal_document():

    return legal_documents()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.route(
    "/upload_document",
    methods=["POST"]
)
@login_required
def upload_document():

    file = request.files.get("file")

    title = request.form.get(
        "title",
        ""
    ).strip()

    category = request.form.get(
        "category",
        ""
    ).strip()

    if not file or not file.filename:

        flash(
            "Please select a document.",
            "error"
        )

        return redirect(
            url_for("legal_document")
        )

    if not allowed_file(file.filename):

        flash(
            "File type is not allowed.",
            "error"
        )

        return redirect(
            url_for("legal_document")
        )

    original_name = secure_filename(
        file.filename
    )

    if not original_name:

        flash(
            "Invalid file name.",
            "error"
        )

        return redirect(
            url_for("legal_document")
        )

    extension = original_name.rsplit(
        ".",
        1
    )[1].lower()

    saved_name = (
        uuid.uuid4().hex
        + "."
        + extension
    )

    # Make sure upload folder exists
    os.makedirs(
        DOCUMENT_FOLDER,
        exist_ok=True
    )

    file.save(
        os.path.join(
            DOCUMENT_FOLDER,
            saved_name
        )
    )

    # Store title and category in description
    description = (
        f"Title: {title}\n"
        f"Category: {category}"
    )

    db = get_db()

    db.execute(
        """
        INSERT INTO documents
        (
            user_id,
            case_id,
            filename,
            original_name,
            file_type,
            description,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            current_user_id(),
            request.form.get(
                "case_id",
                ""
            ),
            saved_name,
            original_name,
            extension,
            description,
            now()
        )
    )

    db.commit()
    db.close()

    flash(
        "Document uploaded successfully.",
        "success"
    )

    return redirect(
        url_for("legal_document")
    )


    # ============================================================
# VIEW DOCUMENT
# ============================================================

@app.route("/view_document/<int:document_id>")
@login_required
def view_document(document_id):

    db = get_db()

    document = db.execute(
        """
        SELECT *
        FROM documents
        WHERE id = ?
        AND user_id = ?
        """,
        (
            document_id,
            current_user_id()
        )
    ).fetchone()

    db.close()

    if not document:

        flash(
            "Document not found.",
            "error"
        )

        return redirect(
            url_for("legal_document")
        )

    file_path = os.path.join(
        DOCUMENT_FOLDER,
        document["filename"]
    )

    if not os.path.exists(file_path):

        flash(
            "Document file is missing.",
            "error"
        )

        return redirect(
            url_for("legal_document")
        )

    return send_from_directory(
        DOCUMENT_FOLDER,
        document["filename"],
        as_attachment=False
    )

# ============================================================
# DOWNLOAD DOCUMENT
# ============================================================

@app.route(
    "/download_document/<filename>"
)
@login_required
def download_document(filename):

    return send_from_directory(
        DOCUMENT_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# EVIDENCE MANAGER
# ============================================================

@app.route("/evidence_manager")
@login_required
def evidence_manager():

    db = get_db()

    evidence = db.execute(
        """
        SELECT *
        FROM evidence
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (current_user_id(),)
    ).fetchall()

    db.close()

    return render_template(
        "evidence_manager.html",
        evidence=evidence
    )





# ============================================================
# ADD EVIDENCE
# ============================================================

@app.route(
    "/add_evidence",
    methods=["POST"]
)
@login_required
def add_evidence():

    case_number = request.form.get(
        "case_number",
        ""
    ).strip()

    evidence_name = request.form.get(
        "evidence_name",
        ""
    ).strip()

    evidence_type = request.form.get(
        "evidence_type",
        "Other"
    ).strip()

    priority = request.form.get(
        "priority",
        "Medium"
    ).strip()

    description = request.form.get(
        "description",
        ""
    ).strip()


    if not evidence_name:

        flash(
            "Evidence name is required.",
            "error"
        )

        return redirect(
            url_for("evidence_manager")
        )


    db = get_db()


    db.execute(
        """
        INSERT INTO evidence
        (
            user_id,
            title,
            case_number,
            evidence_name,
            evidence_type,
            priority,
            description,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            current_user_id(),
            evidence_name,
            case_number,
            evidence_name,
            evidence_type,
            priority,
            description,
            now()
        )
    )


    db.commit()
    db.close()


    flash(
        "Evidence added successfully.",
        "success"
    )


    return redirect(
        url_for("evidence_manager")

    )
# ============================================================
# DELETE EVIDENCE
# ============================================================

@app.route("/delete_evidence/<int:evidence_id>", methods=["POST"])
@login_required
def delete_evidence(evidence_id):

    db = get_db()

    db.execute(
        """
        DELETE FROM evidence
        WHERE id = ?
        AND user_id = ?
        """,
        (
            evidence_id,
            current_user_id()
        )
    )

    db.commit()
    db.close()

    flash(
        "Evidence deleted successfully.",
        "success"
    )

    return redirect(
        url_for("evidence_manager")
    )
# ============================================================
# CASE TRACKER
# ============================================================

# ============================================================
# CASE TRACKER
# ============================================================

@app.route("/case_tracker")
@login_required
def case_tracker():

    db = get_db()

    cases = db.execute(
        """
        SELECT *
        FROM cases
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (current_user_id(),)
    ).fetchall()

    # Dashboard-la irundhu varum case ID
    selected_case_id = request.args.get("case_id", "")

    db.close()

    return render_template(
        "case_tracker.html",
        cases=cases,
        selected_case_id=selected_case_id
    )

    # ============================================================
# DELETE CASE
# ============================================================

@app.route("/delete_case/<int:case_id>", methods=["POST"])
@login_required
def delete_case(case_id):

    db = get_db()

    case = db.execute(
        """
        SELECT *
        FROM cases
        WHERE id = ?
        AND user_id = ?
        """,
        (case_id, current_user_id())
    ).fetchone()

    if not case:
        db.close()

        flash("Case not found.", "error")

        return redirect(url_for("case_tracker"))

    db.execute(
        """
        DELETE FROM cases
        WHERE id = ?
        AND user_id = ?
        """,
        (case_id, current_user_id())
    )

    db.commit()
    db.close()

    flash("Case deleted successfully.", "success")

    return redirect(url_for("case_tracker"))

@app.route(
    "/create_case",
    methods=["POST"]
)
@login_required
def create_case():

    case_id = request.form.get(
        "case_id",
        ""
    ).strip()

    if not case_id:

        case_id = "CASE-" + uuid.uuid4().hex[:6].upper()

    case_name = request.form.get(
        "case_name",
        "New Case"
    ).strip()

    case_type = request.form.get(
        "case_type",
        "General"
    )

    status = request.form.get(
        "status",
        "Active"
    )

    next_hearing = request.form.get(
        "next_hearing",
        ""
    )

    description = request.form.get(
        "description",
        ""
    )

    db = get_db()

    db.execute(
        """
        INSERT INTO cases
        (
            user_id,
            case_id,
            case_name,
            case_type,
            status,
            next_hearing,
            description,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            current_user_id(),
            case_id,
            case_name,
            case_type,
            status,
            next_hearing,
            description,
            now()
        )
    )

    db.commit()
    db.close()

    flash(
        "Case created successfully.",
        "success"
    )

    return redirect(
        url_for("case_tracker")
    )


@app.route(
    "/case/<int:case_id>"
)
@login_required
def view_case(case_id):

    db = get_db()

    case = db.execute(
        """
        SELECT *
        FROM cases
        WHERE id = ?
          AND user_id = ?
        """,
        (
            case_id,
            current_user_id()
        )
    ).fetchone()

    db.close()

    if not case:

        flash(
            "Case not found.",
            "error"
        )

        return redirect(
            url_for("case_tracker")
        )

    return render_template(
        "case_tracker.html",
        cases=[case],
        selected_case=case
    )


@app.route(
    "/case/<int:case_id>/update",
    methods=["POST"]
)
@login_required
def update_case(case_id):

    db = get_db()

    db.execute(
        """
        UPDATE cases
        SET case_name = ?,
            case_type = ?,
            status = ?,
            next_hearing = ?,
            description = ?
        WHERE id = ?
          AND user_id = ?
        """,
        (
            request.form.get("case_name", ""),
            request.form.get("case_type", ""),
            request.form.get("status", "Active"),
            request.form.get("next_hearing", ""),
            request.form.get("description", ""),
            case_id,
            current_user_id()
        )
    )

    db.commit()
    db.close()

    return redirect(
        url_for("case_tracker")
    )


@app.route(
    "/case/<int:case_id>/delete",
    methods=["POST"]
)

# ============================================================
# RANKING & INSIGHTS
# ============================================================

@app.route("/ranking")
@login_required
def ranking():

    db = get_db()

    cases = db.execute(
        """
        SELECT *
        FROM cases
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (current_user_id(),)
    ).fetchall()

    evidence_rows = db.execute(
        """
        SELECT case_id, COUNT(*) AS evidence_count
        FROM evidence
        WHERE user_id = ?
        GROUP BY case_id
        """,
        (current_user_id(),)
    ).fetchall()

    document_rows = db.execute(
        """
        SELECT case_id, COUNT(*) AS document_count
        FROM documents
        WHERE user_id = ?
        GROUP BY case_id
        """,
        (current_user_id(),)
    ).fetchall()

    db.close()


    # --------------------------------------------------------
    # CREATE EVIDENCE COUNT DICTIONARY
    # --------------------------------------------------------

    evidence_counts = {}

    for row in evidence_rows:

        evidence_counts[row["case_id"]] = row["evidence_count"]


    # --------------------------------------------------------
    # CREATE DOCUMENT COUNT DICTIONARY
    # --------------------------------------------------------

    document_counts = {}

    for row in document_rows:

        document_counts[row["case_id"]] = row["document_count"]


    # --------------------------------------------------------
    # RANK CASES
    # --------------------------------------------------------

    ranked_cases = []

    for case in cases:

        case_id = case["case_id"]

        evidence_count = evidence_counts.get(
            case_id,
            0
        )

        document_count = document_counts.get(
            case_id,
            0
        )


        # BASE SCORE

        score = 40


        # EVIDENCE SCORE

        score += min(
            evidence_count * 10,
            30
        )


        # DOCUMENT SCORE

        score += min(
            document_count * 10,
            20
        )


        # STATUS SCORE

        if case["status"] == "Active":

            score += 10

        elif case["status"] == "Hearing":

            score += 15

        elif case["status"] == "Pending":

            score += 5

        elif case["status"] == "Resolved":

            score += 5


        # LIMIT SCORE TO 100

        score = min(
            score,
            100
        )


        # PRIORITY

        if score >= 80:

            priority = "High"

        elif score >= 60:

            priority = "Medium"

        else:

            priority = "Low"


        ranked_cases.append({

            "case_id": case_id,

            "case_name": case["case_name"],

            "case_type": case["case_type"],

            "status": case["status"],

            "score": score,

            "priority": priority,

            "evidence_count": evidence_count,

            "document_count": document_count

        })


    # --------------------------------------------------------
    # SORT CASES BY SCORE
    # --------------------------------------------------------

    ranked_cases.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_cases = len(ranked_cases)


    high_priority = len([

        case for case in ranked_cases

        if case["priority"] == "High"

    ])


    resolved_cases = len([

        case for case in ranked_cases

        if case["status"] == "Resolved"

    ])


    if total_cases > 0:

        success_rate = round(
            (
                resolved_cases /
                total_cases
            ) * 100
        )

    else:

        success_rate = 0


    # --------------------------------------------------------
    # AI INSIGHTS
    # --------------------------------------------------------

    insights = []


    if total_cases == 0:

        insights.append(
            "No cases have been added yet. Add cases using Case Tracker."
        )


    else:

        top_case = ranked_cases[0]

        insights.append(
            f"{top_case['case_name']} currently has the highest case score of "
            f"{top_case['score']}%."
        )


        for case in ranked_cases:

            if case["evidence_count"] == 0:

                insights.append(
                    f"{case['case_name']} does not have any evidence records yet."
                )

            if case["document_count"] == 0:

                insights.append(
                    f"{case['case_name']} does not have any legal documents yet."
                )


        if high_priority > 0:

            insights.append(
                f"{high_priority} case(s) currently require high priority attention."
            )


    # --------------------------------------------------------
    # OVERALL ANALYTICS
    # --------------------------------------------------------

    total_evidence = sum(

        case["evidence_count"]

        for case in ranked_cases

    )


    total_documents = sum(

        case["document_count"]

        for case in ranked_cases

    )


    if total_cases > 0:

        evidence_strength = min(
            round(
                (
                    total_evidence /
                    (total_cases * 3)
                ) * 100
            ),
            100
        )

        document_accuracy = min(
            round(
                (
                    total_documents /
                    (total_cases * 3)
                ) * 100
            ),
            100
        )

    else:

        evidence_strength = 0

        document_accuracy = 0


    return render_template(

        "ranking.html",

        ranked_cases=ranked_cases,

        total_cases=total_cases,

        high_priority=high_priority,

        resolved_cases=resolved_cases,

        success_rate=success_rate,

        insights=insights,

        evidence_strength=evidence_strength,

        document_accuracy=document_accuracy

    )



# ============================================================
# HEARING REMINDER
# ============================================================

@app.route("/hearing_reminder")
@login_required
def hearing_reminder():

    db = get_db()

    hearings = db.execute(
        """
        SELECT *
        FROM hearings
        WHERE user_id = ?
        ORDER BY hearing_date ASC
        """,
        (current_user_id(),)
    ).fetchall()

    db.close()

    return render_template(
        "hearing_remainder.html",
        hearings=hearings
    )


# Compatibility endpoint
@app.route("/hearing_remainder")
@login_required
def hearing_remainder():

    return hearing_reminder()



# ============================================================
# ADD HEARING
# ============================================================

@app.route(
    "/add_hearing",
    methods=["POST"]
)
@login_required
def add_hearing():

    db = get_db()

    db.execute(
        """
        INSERT INTO hearings
        (
            user_id,
            case_id,
            hearing_date,
            hearing_time,
            court,
            notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            current_user_id(),
            request.form.get("case_id", ""),
            request.form.get("hearing_date", ""),
            request.form.get("hearing_time", ""),
            request.form.get("court", ""),
            request.form.get("notes", ""),
            now()
        )
    )

    db.commit()
    db.close()

    flash(
        "Hearing reminder added.",
        "success"
    )

    return redirect(
        url_for("hearing_reminder")
    )

    # ============================================================
# DELETE HEARING
# ============================================================

@app.route("/delete_hearing/<int:hearing_id>", methods=["POST"])
@login_required
def delete_hearing(hearing_id):

    db = get_db()

    db.execute(
        """
        DELETE FROM hearings
        WHERE id = ?
        AND user_id = ?
        """,
        (hearing_id, current_user_id())
    )

    db.commit()
    db.close()

    flash("Hearing deleted successfully.", "success")

    return redirect(url_for("hearing_reminder"))


@app.route("/case_timeline", methods=["GET"])
@login_required
def case_timeline():

    db = get_db()

    cases = db.execute(
        """
        SELECT *
        FROM cases
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (current_user_id(),)
    ).fetchall()

    selected_case_id = request.args.get(
        "case_id", ""
    ).strip()

    selected_case_name = ""
    events = []

    if selected_case_id:

        selected_case = db.execute(
            """
            SELECT case_name
            FROM cases
            WHERE user_id = ?
            AND case_id = ?
            """,
            (
                current_user_id(),
                selected_case_id
            )
        ).fetchone()

        if selected_case:

            selected_case_name = selected_case["case_name"]

            events = db.execute(
                """
                SELECT *
                FROM timeline
                WHERE user_id = ?
                AND case_id = ?
                ORDER BY event_date ASC, id ASC
                """,
                (
                    current_user_id(),
                    selected_case_id
                )
            ).fetchall()

    db.close()

    return render_template(
        "case_timeline.html",
        cases=cases,
        selected_case_id=selected_case_id,
        selected_case_name=selected_case_name,
        events=events
    ) # ============================================================
# ADD TIMELINE EVENT
# ============================================================

@app.route(
    "/add_timeline",
    methods=["POST"]
)
@login_required
def add_timeline():

    db = get_db()

    db.execute(
        """
        INSERT INTO timeline
        (
            user_id,
            case_id,
            event_title,
            event_date,
            description,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            current_user_id(),
            request.form.get("case_id", ""),
            request.form.get(
                "event_title",
                "Case Event"
            ),
            request.form.get(
                "event_date",
                ""
            ),
            request.form.get(
                "description",
                ""
            ),
            now()
        )
    )

    db.commit()
    db.close()
    return redirect(
        url_for("case_timeline")
    )



@app.route("/document_analyzer", methods=["GET", "POST"])
@login_required
def document_analyzer():

    if request.method == "GET":
        return render_template(
            "document_analyzer.html",
            analysis=None
        )

    try:
        uploaded_file = (
            request.files.get("file")
            or request.files.get("document")
            or request.files.get("case_document")
        )

        if not uploaded_file or not uploaded_file.filename:
            return jsonify({
                "success": False,
                "message": "Please upload a document."
            }), 400

        if not allowed_file(uploaded_file.filename):
            return jsonify({
                "success": False,
                "message": "Unsupported file type."
            }), 400

        original_name = secure_filename(
            uploaded_file.filename
        )

        saved_name = (
            uuid.uuid4().hex + "_" + original_name
        )

        file_path = os.path.join(
            DOCUMENT_FOLDER,
            saved_name
        )

        uploaded_file.save(file_path)

        extracted_text = get_file_text(file_path)

        if not extracted_text.strip():
            return jsonify({
                "success": False,
                "message": (
                    "Text could not be extracted. "
                    "Please upload a text-based PDF, DOCX or TXT file."
                )
            }), 400

        prompt = (
            "Analyze this legal document in simple language.\n"
            "Identify important points, dates, parties, legal issues "
            "and possible missing information.\n"
            "Do not invent facts.\n\n"
            + extracted_text[:15000]
        )

        analysis = ask_ai_model(prompt)

        return jsonify({
            "success": True,
            "analysis": analysis,
            "result": analysis
        })

    except Exception as e:

        print("DOCUMENT ANALYZER ERROR:", e)

        return jsonify({
            "success": False,
            "message": (
                "Document analysis failed. "
                "Please check the Flask terminal."
            )
        }), 500
        # ----------------------------------------------------
        # OPTION 1: Analyze entered text
        # ----------------------------------------------------

        if text:

            prompt = (
                "Analyze this legal document in simple language.\n"
                "Identify important points, dates, parties, "
                "legal issues and possible missing information.\n"
                "Do not invent facts.\n\n"
                + text
            )

            analysis = ask_ai_model(prompt)

        # ----------------------------------------------------
        # OPTION 2: Analyze uploaded document
        # ----------------------------------------------------

        elif uploaded_file and uploaded_file.filename:

            if not allowed_file(uploaded_file.filename):

                analysis = "Unsupported file type."

            else:

                try:

                    original_name = secure_filename(
                        uploaded_file.filename
                    )

                    saved_name = (
                        uuid.uuid4().hex
                        + "_"
                        + original_name
                    )

                    file_path = os.path.join(
                        DOCUMENT_FOLDER,
                        saved_name
                    )

                    uploaded_file.save(file_path)

                    extracted_text = get_file_text(file_path)

                    if extracted_text:

                        prompt = (
                            "Analyze this legal document "
                            "in simple language.\n"
                            "Identify important points, dates, "
                            "parties, legal issues and possible "
                            "missing information.\n"
                            "Do not invent facts.\n\n"
                            + extracted_text[:15000]
                        )

                        analysis = ask_ai_model(prompt)

                    else:

                        analysis = (
                            "The document was uploaded, "
                            "but text could not be extracted. "
                            "Please upload a text-based PDF, "
                            "DOCX or TXT file."
                        )

                except Exception as e:

                    print("DOCUMENT ANALYZER ERROR:", e)

                    analysis = (
                        "An error occurred while analyzing "
                        "the document. Please check the "
                        "Flask terminal."
                    )

        else:

            analysis = (
                "Please enter text or upload a document."
            )

    return render_template(
        "document_analyzer.html",
        analysis=analysis
    )




@app.route("/report-generator", methods=["GET", "POST"])
@app.route("/report_generator", methods=["GET", "POST"])
@login_required
def report_generator():

    db = get_db()

    report = None
    selected_case = None

    case_id = ""
    report_type = "Complete Case Report"

    # Get all cases
    cases = db.execute(
        "SELECT * FROM cases ORDER BY id DESC"
    ).fetchall()

    # Get cases table columns
    case_columns = [
        row[1]
        for row in db.execute(
            "PRAGMA table_info(cases)"
        ).fetchall()
    ]

    if request.method == "POST":

        case_id = request.form.get(
            "case_id", ""
        ).strip()
        print("REPORT GENERATOR POST RECEIVED")
        print("CASE ID:", case_id)
        print("REPORT TYPE:", report_type)

        report_type = request.form.get(
            "report_type",
            "Complete Case Report"
        )

        # Find selected case
        if case_id:

            if "case_id" in case_columns:

                selected_case = db.execute(
                    """
                    SELECT *
                    FROM cases
                    WHERE CAST(case_id AS TEXT) = ?
                    """,
                    (case_id,)
                ).fetchone()

            elif (
                "case_number" in case_columns
            ):

                selected_case = db.execute(
                    """
                    SELECT *
                    FROM cases
                    WHERE CAST(case_number AS TEXT) = ?
                    """,
                    (case_id,)
                ).fetchone()

            elif (
                "id" in case_columns
                and case_id.isdigit()
            ):

                selected_case = db.execute(
                    """
                    SELECT *
                    FROM cases
                    WHERE id = ?
                    """,
                    (case_id,)
                ).fetchone()

        # Count documents
        documents_count = db.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        # Count evidence
        evidence_count = db.execute(
            "SELECT COUNT(*) FROM evidence"
        ).fetchone()[0]

        # Count total cases
        cases_count = db.execute(
            "SELECT COUNT(*) FROM cases"
        ).fetchone()[0]

        # Count active cases
        if "status" in case_columns:

            active_cases_count = db.execute(
                """
                SELECT COUNT(*)
                FROM cases
                WHERE status = 'Active'
                """
            ).fetchone()[0]

        else:

            active_cases_count = 0

        # Default case information
        case_name = "Not available"
        case_type = "Not available"
        case_status = "Not available"
        case_description = (
            "No case description available."
        )

        # Selected case information
        if selected_case:

            selected_keys = selected_case.keys()

            if "case_name" in selected_keys:

                case_name = selected_case["case_name"]

            elif "name" in selected_keys:

                case_name = selected_case["name"]

            if "case_type" in selected_keys:

                case_type = selected_case["case_type"]

            if "status" in selected_keys:

                case_status = selected_case["status"]

            if "description" in selected_keys:

                case_description = (
                    selected_case["description"]
                )

        # Final report data
        report = {
            "case_id": case_id,
            "report_type": report_type,
            "case_name": case_name,
            "case_type": case_type,
            "case_status": case_status,
            "case_description": case_description,
            "case_found": selected_case is not None,
            "cases": cases_count,
            "active_cases": active_cases_count,
            "documents": documents_count,
            "evidence": evidence_count
        }

    db.close()

    return render_template(
        "report_generator.html",
        report=report,
        selected_case=selected_case,
        cases=cases
    )
# ============================================================
# NOTIFICATIONS
# ============================================================

@app.route("/notifications")
@login_required
def notifications():
    db = get_db()

    items = db.execute(
        """
        SELECT *
        FROM notifications
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (current_user_id(),)
    ).fetchall()

    total_notifications = db.execute(
        """
        SELECT COUNT(*)
        FROM notifications
        WHERE user_id = ?
        """,
        (current_user_id(),)
    ).fetchone()[0]

    unread_notifications = db.execute(
        """
        SELECT COUNT(*)
        FROM notifications
        WHERE user_id = ?
        AND is_read = 0
        """,
        (current_user_id(),)
    ).fetchone()[0]

    db.close()

    return render_template(
        "notifications.html",
        notifications=items,
        total_notifications=total_notifications,
        unread_notifications=unread_notifications
    )


@app.route("/notification")
@login_required
def notification():
    return notifications()


# ============================================================
# MARK ALL NOTIFICATIONS AS READ
# ============================================================

@app.route("/notifications/mark-read", methods=["POST"])
@login_required
def mark_notifications_read():
    db = get_db()

    db.execute(
        """
        UPDATE notifications
        SET is_read = 1
        WHERE user_id = ?
        """,
        (current_user_id(),)
    )

    db.commit()
    db.close()

    return redirect(url_for("notifications"))


# ============================================================
# CLEAR ALL NOTIFICATIONS
# ============================================================

@app.route("/notifications/clear", methods=["POST"])
@login_required
def clear_notifications():
    db = get_db()

    db.execute(
        """
        DELETE FROM notifications
        WHERE user_id = ?
        """,
        (current_user_id(),)
    )

    db.commit()
    db.close()

    return redirect(url_for("notifications"))


# ============================================================
# ADD NOTIFICATION
# ============================================================

def create_notification(user_id, title, message):
    db = get_db()

    db.execute(
        """
        INSERT INTO notifications
        (
            user_id,
            title,
            message,
            is_read,
            created_at
        )
        VALUES (?, ?, ?, 0, ?)
        """,
        (
            user_id,
            title,
            message,
            now()
        )
    )

    db.commit()
    db.close()
# ============================================================
# AI CASE RISK ANALYZER
# ============================================================

@app.route("/risk_analyser", methods=["GET", "POST"])
@login_required
def risk_analyser():

    result = None

    if request.method == "POST":

        case_id = request.form.get("case_id", "").strip()
        uploaded_file = request.files.get("case_file")

        if not case_id:
            result = {
                "level": "Needs Review",
                "score": 0,
                "message": "Please enter Case ID."
            }

        elif not uploaded_file or uploaded_file.filename == "":
            result = {
                "level": "Needs Review",
                "score": 0,
                "message": "Please upload a PDF file."
            }

        elif not uploaded_file.filename.lower().endswith(".pdf"):
            result = {
                "level": "Needs Review",
                "score": 0,
                "message": "Only PDF files are supported."
            }

        else:
            try:
                # Extract PDF text
                reader = PdfReader(uploaded_file)

                extracted_text = ""

                for page in reader.pages:
                    extracted_text += (page.extract_text() or "") + "\n"

                extracted_text = extracted_text.strip()

                if not extracted_text:
                    result = {
                        "level": "Needs Review",
                        "score": 0,
                        "message": "No readable text found. Upload a text-based PDF."
                    }

                else:

                    # Limit text sent to Ollama
                    extracted_text = extracted_text[:18000]

                    prompt = f"""
You are an AI legal case preparation assistant.

Analyze the following case document.

Case ID: {case_id}

Document:
{extracted_text}

Return ONLY valid JSON in this format:

{{
  "summary": "Simple language summary of the case",
  "risk_score": 50,
  "risk_level": "Medium",
  "risk_factors": [
    "Risk factor 1",
    "Risk factor 2"
  ],
  "missing_information": [
    "Information that is missing from the document"
  ],
  "preparation_checklist": [
    "General preparation item"
  ],
  "recommendation": "General informational recommendation"
}}

Rules:
- Use only information available in the document.
- Do not invent facts, evidence, laws, or case details.
- If information is missing, clearly say it is not mentioned.
- Risk score must be between 0 and 100.
- 0 means fewer identified risks; 100 means more identified risks.
- Risk level must be Low, Medium, or High.
- This is an indicative AI assessment, not a prediction of the court's final decision.
- Do not guarantee any legal outcome.
"""

                    # Call existing Ollama AI
                    response = ollama.chat(
                        model="gemma2:2b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        options={
                            "temperature": 0.2,
                            "num_predict": 700
                        }
                    )

                    ai_response = response["message"]["content"].strip()

                    # Remove Markdown code fences
                    ai_response = re.sub(
                        r"```json|```",
                        "",
                        ai_response
                    ).strip()

                    ai_result = json.loads(ai_response)

                    # Validate risk score
                    score = float(ai_result.get("risk_score", 50))
                    score = max(0, min(100, score))
                    score = round(score)

                    # Determine level from score
                    if score < 35:
                        level = "Low"
                    elif score < 70:
                        level = "Medium"
                    else:
                        level = "High"

                    result = {
                        "case_id": case_id,
                        "score": score,
                        "level": level,
                        "message": (
                            "AI analysis completed. "
                            "This is an indicative assessment "
                            "based only on the uploaded document."
                        ),
                        "extracted_text": extracted_text,
                        "summary": ai_result.get("summary", ""),
                        "risk_factors": ai_result.get("risk_factors", []),
                        "missing_information": ai_result.get(
                            "missing_information", []
                        ),
                        "preparation_checklist": ai_result.get(
                            "preparation_checklist", []
                        ),
                        "recommendation": ai_result.get(
                            "recommendation", ""
                        )
                    }

            except json.JSONDecodeError:
                result = {
                    "case_id": case_id,
                    "score": 0,
                    "level": "Needs Review",
                    "message": "Ollama returned an invalid response. Please try again."
                }

            except Exception as e:
                print("RISK ANALYZER ERROR:", str(e))

                result = {
                    "case_id": case_id,
                    "score": 0,
                    "level": "Needs Review",
                    "message": "Unable to analyze the PDF. Check that Ollama is running."
                }

    return render_template(
        "risk_analyzer.html",
        result=result
    )


@app.route("/risk_analyzer", methods=["GET", "POST"])
@login_required
def risk_analyzer():
    return risk_analyser()
# ============================================================
# CASE OUTCOME PREDICTION
# ============================================================

@app.route("/case-outcome", methods=["GET", "POST"])
@login_required
def case_outcome():

    if request.method == "GET":
        return render_template("case_outcome.html")

    try:
        case_number = request.form.get("case_number", "").strip()
        case_type = request.form.get("case_type", "").strip()
        pdf_file = request.files.get("case_pdf")

        if not case_number:
            return jsonify({
                "success": False,
                "error": "Please enter Case Number"
            }), 400

        if not pdf_file or pdf_file.filename == "":
            return jsonify({
                "success": False,
                "error": "Please upload a case PDF"
            }), 400

        if not pdf_file.filename.lower().endswith(".pdf"):
            return jsonify({
                "success": False,
                "error": "Only PDF files are supported"
            }), 400

        # Extract PDF text
        reader = PdfReader(pdf_file)
        extracted_text = ""

        for page in reader.pages:
            page_text = page.extract_text() or ""
            extracted_text += page_text + "\n"

        if not extracted_text.strip():
            return jsonify({
                "success": False,
                "error": "Could not extract text from this PDF. It may be scanned."
            }), 400

        # Limit text to avoid very large AI requests
        extracted_text = extracted_text[:18000]

        prompt = f"""
You are an AI legal document analysis assistant.

Analyze the following case document.

Case Number: {case_number}
Case Type: {case_type}

Document Text:
{extracted_text}

Return ONLY valid JSON in this format:

{{
    "summary": "Short summary of the case",
    "good_percentage": 50,
    "bad_percentage": 50,
    "evidence_strength": "Strong, Moderate, or Weak",
    "supporting_points": [
        "Point 1",
        "Point 2"
    ],
    "risk_points": [
        "Risk 1",
        "Risk 2"
    ],
    "recommendation": "General informational recommendation"
}}

Important:
- Percentages are only indicative AI estimates.
- Do not claim certainty about the court's final decision.
- The good and bad percentages must total 100.
- Use only information available in the document.
"""

        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 700
            }
        )

        ai_response = response["message"]["content"].strip()

        # Remove markdown code fences if AI adds them
        ai_response = re.sub(
            r"```json|```",
            "",
            ai_response
        ).strip()

        result = json.loads(ai_response)

        # Ensure percentages are numbers
        good = float(result.get("good_percentage", 50))
        bad = float(result.get("bad_percentage", 50))

        total = good + bad

        if total <= 0:
            good = 50
            bad = 50
        else:
            good = round((good / total) * 100, 1)
            bad = round(100 - good, 1)

        result["good_percentage"] = good
        result["bad_percentage"] = bad

        return jsonify({
            "success": True,
            "case_number": case_number,
            "result": result
        })

    except json.JSONDecodeError:
       return jsonify({
            "success": False,
            "error": "AI returned an invalid response. Please try again."
        }), 500

    except Exception as e:
        print("CASE OUTCOME ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
# ============================================================
# DUPLICATE DOCUMENT DETECTION
# ============================================================

@app.route(
    "/duplicate_detection",
    methods=["GET", "POST"]
)
@login_required
def duplicate_detection():

    duplicates = []

    db = get_db()

    documents = db.execute(
        """
        SELECT *
        FROM documents
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (current_user_id(),)
    ).fetchall()

    db.close()

    seen = set()

    for document in documents:

        name = (
            document["original_name"]
            or
            document["filename"]
            or
            ""
        ).lower().strip()

        if name in seen:

            duplicates.append(document)

        else:

            seen.add(name)

    return render_template(
        "duplicate_detection.html",
        documents=documents,
        duplicates=duplicates
    )


# ============================================================
# VOICE QUERY
# ============================================================

@app.route(
    "/voice_query"
)
@login_required
def voice_query():

    return render_template(
        "voice_query.html"
    )


# ============================================================
# MULTI-LANGUAGE
# ============================================================

@app.route(
    "/multi_language"
)
@login_required
def multi_language():

    return render_template(
        "multi_language.html"
    )


@app.route(
    "/multi-language"
)
@login_required
def multi_language_support():

    return multi_language()


# ============================================================
# CHANGE LANGUAGE
# ============================================================

@app.route("/multi_language/change", methods=["POST"])
@login_required
def change_language():

    data = request.get_json(silent=True) or request.form

    language = data.get("language", "English").strip()

    allowed_languages = [
        "English",
        "Tamil",
        "Hindi"
    ]

    if language not in allowed_languages:
        language = "English"

    session["language"] = language

    return jsonify({
        "success": True,
        "language": language
    })



# ============================================================
# TRANSLATION API
# ============================================================

@app.route("/translate", methods=["POST"])
@login_required
def translate_api():

    data = request.get_json(silent=True) or request.form

    text = data.get("text", "").strip()
    source_language = data.get("source_language", "English").strip()
    target_language = data.get("language", "Tamil").strip()

    if not text:
        return jsonify({
            "success": False,
            "message": "Text is required."
        })

    if source_language == target_language:
        return jsonify({
            "success": True,
            "language": target_language,
            "translation": text
        })

    try:
        import ollama

        prompt = f"""
Translate the following text from {source_language} to {target_language}.

Rules:
- Give ONLY the translated text.
- Do not explain anything.
- Do not add extra information.
- Preserve the original meaning.

Text:
{text}
"""

        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fast and accurate translation assistant. "
                        "Translate text directly and clearly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 150
            }
        )

        translation = response["message"]["content"].strip()

        return jsonify({
            "success": True,
            "language": target_language,
            "translation": translation
        })

    except Exception as e:

        print("TRANSLATION ERROR:", e)

        return jsonify({
            "success": False,
            "message": "Ollama translation failed. Please make sure Ollama is running."
        })

# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "application": "AI Powered Courtroom Assistant"
    })




# ------------------------------------------------------------
# ASK-AI MODEL FUNCTION
# ------------------------------------------------------------

def ask_ai_model(question):

    if not question:
        return "Please enter your question."

    try:
        import ollama

        response = ollama.chat(
            model="qwen2.5:0.5b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal information assistant. "
                        "Give clear, simple and short answers. "
                        "Answer in 2 or 3 sentences only. "
                        "Do not show thinking. "
                        "This is informational guidance, "
                        "not professional legal advice."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            think=False,
            keep_alive="10m",

            options={
                "temperature": 0.1,
                "num_predict": 80,
                "num_ctx": 512
            }
        )

        answer = response["message"].get(
            "content", ""
        ).strip()

        print("AI RESPONSE:", repr(answer))

        if not answer:
            return "The AI returned an empty response."

        return answer

    except Exception as e:

        print("OLLAMA ERROR:", e)

        return (
            "AI connection failed. "
            "Please make sure Ollama is running."
        )
    # ------------------------------------------------------------
# VOICE AI RESPONSE
# ------------------------------------------------------------

@app.route("/voice-ai-response", methods=["POST"])
@login_required
def voice_ai_response():

    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({
            "success": False,
            "answer": "Please enter a question."
        }), 400

    answer = ask_ai_model(question)

    return jsonify({
        "success": True,
        "answer": answer
    })



    # ============================================================
# AI CASE OUTCOME ANALYSIS
# ============================================================

@app.route("/predict-case-outcome", methods=["POST"])
@login_required
def predict_case_outcome():

    case_number = request.form.get("case_number", "").strip()
    case_type = request.form.get("case_type", "").strip()
    evidence_strength = request.form.get("evidence_strength", "").strip()
    case_details = request.form.get("case_details", "").strip()

    if not case_details:
        return jsonify({
            "success": False,
            "answer": "Please enter case details."
        })

    question = f"""
Analyze the following legal case for educational purposes.

Case Number: {case_number}
Case Type: {case_type}
Evidence Strength: {evidence_strength}
Case Details: {case_details}

Provide:
1. Key factors
2. Supporting evidence
3. Possible legal risks
4. Further review required
5. A neutral educational assessment

Do not claim to predict the actual court judgment.
Do not provide guaranteed confidence percentages.
Mention that a qualified lawyer should review the case.
"""

    answer = ask_ai_model(question)

    return jsonify({
        "success": True,
        "answer": answer
    })

# ============================================================
# DATABASE COLUMN FIX
# ============================================================

def fix_database_columns():

    db = get_db()

    try:
        db.execute(
            "ALTER TABLE documents ADD COLUMN case_id TEXT"
        )

    except sqlite3.OperationalError:
        pass

    db.commit()
    db.close()


# ============================================================
# EVIDENCE DATABASE COLUMN FIX
# ============================================================

def fix_evidence_columns():

    db = get_db()

    columns = [
        ("case_number", "TEXT"),
        ("evidence_name", "TEXT"),
        ("evidence_type", "TEXT"),
        ("priority", "TEXT"),
        ("description", "TEXT"),
        ("created_at", "TEXT")
    ]

    for column_name, column_type in columns:

        try:

            db.execute(
                f"ALTER TABLE evidence ADD COLUMN {column_name} {column_type}"
            )

        except sqlite3.OperationalError:
            pass

    db.commit()
    db.close()

@app.route("/delete_document/<int:document_id>", methods=["POST"])
@login_required
def delete_document(document_id):
    db = get_db()

    document = db.execute(
        "SELECT * FROM documents WHERE id = ? AND user_id = ?",
        (document_id, session["user_id"])
    ).fetchone()

    if not document:
        return "Document not found", 404

    filename = document["filename"]

    db.execute(
        "DELETE FROM documents WHERE id = ? AND user_id = ?",
        (document_id, session["user_id"])
    )
    db.commit()

    file_path = os.path.join(DOCUMENT_FOLDER, filename)

    if os.path.isfile(file_path):
        os.remove(file_path)

    return redirect(url_for("legal_documents"))


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    fix_database_columns()
    fix_evidence_columns()

    print("=" * 60)
    print("AI POWERED COURTROOM ASSISTANT")
    print("=" * 60)
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
    host="127.0.0.1",
    port=5000,
    debug=True,
    threaded=True
)
    