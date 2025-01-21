import os
from dotenv import load_dotenv
from core.data_ingestion import ingest_data
from core.text_processing import preprocess_document
from core.embeddings import create_vector_store
from core.interview_logic import extract_key_requirements, extract_company_values
from core.question_generation import generate_interview_questions
from core.conversation_flow import run_interview

def main():
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # 1) Ingest Data
    # Assumes PDF for resume
    job_post_raw, company_profile_raw, resume_raw = ingest_data(
        job_path="data/job_post.txt",
        company_path="data/company_profile.txt",
        resume_path="data/VijayRudraraju_Resume.pdf",
        do_chunking=True
    )

    # 2) Preprocess data with SpaCy
    job_post_clean = preprocess_document(job_post_raw)
    company_profile_clean = preprocess_document(company_profile_raw)
    resume_clean = preprocess_document(resume_raw)


    # 4) Create Vector Stores and split into smaller chunks
    job_chunks = job_post_clean.split("\n")
    company_chunks = company_profile_clean.split("\n")
    resume_chunks = resume_clean.split("\n")


    job_vectorstore = create_vector_store(job_chunks, openai_api_key)
    company_vectorstore = create_vector_store(company_chunks, openai_api_key)
    resume_vectorstore = create_vector_store(resume_chunks, openai_api_key)

    # 5) Generate Interview Questions
    questions = generate_interview_questions(
        job_vectorstore,
        company_vectorstore,
        resume_vectorstore,
        openai_api_key=openai_api_key,
        n_questions=7
    )

    # 6) Conduct Interview
    print("\n----- AI-Driven Interview -----\n")
    interview_responses = run_interview(questions)

    # 7) Summarize
    print("\n----- Interview Session Summary -----\n")
    for q, ans in interview_responses.items():
        print(f"Q: {q}\nA: {ans}\n")



if __name__ == "__main__":
    main()

