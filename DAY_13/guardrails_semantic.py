from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# CONFIGURATION

MAX_QUERY_LENGTH = 500
MIN_WORDS = 3

SIMILARITY_THRESHOLD = 0.35

# PROMPT INJECTION PATTERNS

PROMPT_INJECTION_PATTERNS = [

    "ignore previous instructions",

    "ignore all previous instructions",

    "forget previous instructions",

    "forget everything",

    "system prompt",

    "developer message",

    "act as",

    "pretend to be",

    "jailbreak",

    "override",

    "bypass",

    "disable safety",

    "reveal your prompt",

    "show hidden prompt"

]

# RESTRICTED WORDS

BANNED_WORDS = [

    "hack",

    "malware",

    "virus",

    "exploit",

    "password",

    "phishing",

    "ransomware"

]

# DOMAIN DESCRIPTION

DOMAIN_DESCRIPTION = """
This assistant answers questions only about an Employee Handbook.

Supported topics include

Annual Leave

Casual Leave

Sick Leave

Attendance Policy

Salary

Payroll

Benefits

Insurance

Performance Review

Remote Work

Hybrid Work

Holiday Policy

Company Rules

Security Policy

VPN

Code of Conduct

Learning and Development

Reimbursement

Employee Benefits
"""

# LOAD EMBEDDING MODEL

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# CREATE DOMAIN EMBEDDING
# (Only once)

domain_embedding = embedding_model.embed_query(
    DOMAIN_DESCRIPTION
)

# EMPTY QUESTION

def validate_empty(question):

    if len(question.strip()) == 0:

        return False, "Please enter a question."

    return True, ""


# LENGTH CHECK

def validate_length(question):

    if len(question) > MAX_QUERY_LENGTH:

        return False, "Question is too long."

    return True, ""


# MINIMUM WORDS

def validate_min_words(question):

    if len(question.split()) < MIN_WORDS:

        return False, "Please ask a more descriptive question."

    return True, ""


# PROMPT INJECTION DETECTION

def detect_prompt_injection(question):

    q = question.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:

        if pattern in q:

            return False, "Prompt Injection Detected."

    return True, ""


# RESTRICTED WORDS

def detect_banned_words(question):

    q = question.lower()

    for word in BANNED_WORDS:

        if word in q:

            return False, f"Restricted keyword detected: {word}"

    return True, ""


# SEMANTIC DOMAIN VALIDATION

def semantic_domain_validation(question):

    question_embedding = embedding_model.embed_query(
        question
    )

    similarity = cosine_similarity(

        [question_embedding],

        [domain_embedding]

    )[0][0]

    print("=" * 50)
    print("Semantic Similarity :", similarity)
    print("=" * 50)

    if similarity >= SIMILARITY_THRESHOLD:

        return True, ""

    return False, "Question is outside the supported domain."


# MAIN GUARDRAIL FUNCTION

def run_input_guardrails(question):

    validators = [

        validate_empty,

        validate_length,

        validate_min_words,

        detect_prompt_injection,

        detect_banned_words,

        semantic_domain_validation

    ]

    for validator in validators:

        status, message = validator(question)

        if not status:

            return status, message

    return True, ""