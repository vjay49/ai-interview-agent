import openai
from langchain_openai import OpenAI
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.schema import PromptValue

def generate_interview_questions(job_vectorstore, company_vectorstore, resume_vectorstore, openai_api_key: str, n_questions: int = 5) -> list:
    """
    Generate interview questions by retrieving context from each vector store and using it with LLM prompt.
    """
    openai.api_key = openai_api_key
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)

     # Build retrieval-based QA chain for each domain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    job_chain = create_retrieval_chain(job_vectorstore.as_retriever(), combine_docs_chain)
    company_chain = create_retrieval_chain(company_vectorstore.as_retriever(), combine_docs_chain)
    resume_chain = create_retrieval_chain(resume_vectorstore.as_retriever(), combine_docs_chain)

    
    # Gather relevant facts
    job_facts_result = job_chain.invoke({"input":"Summarize the top 5 job requirements in detail."})
    company_facts_result  = company_chain.invoke({"input":"Summarize the company's mission and core values in detail."})
    resume_facts_result  = resume_chain.invoke({"input":"Summarize the candidate's top 10 key technical skills in detail."})

    job_facts = job_facts_result["answer"]
    company_facts = company_facts_result["answer"]
    resume_facts = resume_facts_result["answer"]

    # Consolidate into final prompt
    prompt_template = PromptTemplate(
        template="""
    You are an AI recruiting agent.
    Job Requirements: {job_facts}
    Company Values: {company_facts}
    Candidate Background: {resume_facts}

    Please generate {n_questions} interview questions focusing on:
    1) Technical abilities,
    2) Cultural fit,
    3) Alignment with company values.

    Ensure the questions you ask focus on how a candidate's background and skills fit with the company's job requirements and values.
    Note: questions should also try to distinguish if a candidate meets the technical requirements for the job.
    """,
        input_variables=["job_facts", "company_facts", "resume_facts", "n_questions"]
    )
    prompt_value: PromptValue = prompt_template.format_prompt(
        job_facts=job_facts,
        company_facts=company_facts,
        resume_facts=resume_facts,
        n_questions=n_questions
    )
    
    llm_result = llm.invoke(prompt_value)
    questions = [q.strip() for q in llm_result.split('\n') if q.strip()]
    return questions