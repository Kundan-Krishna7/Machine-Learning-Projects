from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def read_docx(file_path):

    doc = Document(file_path)

    text = []

    for para in doc.paragraphs:
        text.append(para.text)

    return " ".join(text)


resumes = []
resume_names = []

folder = "resumes/Resumes"

for file in os.listdir(folder):

    if file.endswith(".docx"):

        path = os.path.join(folder, file)

        resumes.append(
            read_docx(path)
        )

        resume_names.append(file)


vectorizer = TfidfVectorizer(
    stop_words="english"
)

resume_vectors = vectorizer.fit_transform(
    resumes
)


def get_top_resumes(job_description, top_n=5):

    jd_vector = vectorizer.transform(
        [job_description]
    )

    similarity = cosine_similarity(
        jd_vector,
        resume_vectors
    )[0]

    scores = []

    for i in range(len(resume_names)):

        scores.append(
            (
                resume_names[i],
                round(similarity[i] * 100, 2)
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:top_n]