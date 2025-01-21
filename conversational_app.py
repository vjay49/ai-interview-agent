import os
from dotenv import load_dotenv
from core.data_ingestion import ingest_data
from core.text_processing import preprocess_document
from core.interview_logic import extract_key_requirements, extract_company_values
from agents.conversational_agent import ConversationalInterviewAgent
from core.voice_interface import voice_interview_flow

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    
    # Ingest data and preprocess
    job_post_raw, company_profile_raw, resume_raw = ingest_data(
        job_path="data/job_post.txt",
        company_path="data/company_profile.txt",
        resume_path="data/VijayRudraraju_Resume.pdf",
        do_chunking=True
    )
    job_post_clean = preprocess_document(job_post_raw)
    company_profile_clean = preprocess_document(company_profile_raw)
    resume_clean = preprocess_document(resume_raw)


    # Now create the agent
    agent = ConversationalInterviewAgent(
        openai_api_key=openai_api_key,
        job_requirements=job_post_clean,
        company_profile=company_profile_clean,
        resume_summary=resume_clean
    )

    # Run interview
    voice_interview_flow(agent, max_rounds=5)

if __name__ == "__main__":
    main()