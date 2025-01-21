# ai-interview-agent
AI Interview Agent to Dialogue with Potential Candidates
Version 1:
- Files:
    - app.py (main entry point program to run simple interview which just takes job post, company profile, and resume and creates questions for candidate)
    - data_ingestion.py (for working with input parsing and file storage for the job post, company profile, and resume in pdf format - added some tools such as downloading files from remote URLs for future scalability)
    - embeddings.py (create vector store using FAISS)
    - interview_logic.py (not-used but created to text extract relevant information from input files to lower input size and make system scalable)
    - text_processing.py (for input file and text pre-processing/chunking)
    - conversation_flow.py (used run interview function to command the interview flow)
    - question_generation.py (question generation agent that retrieved information from vector store and created relevant questions based on prompt provided)

Version 2:
- Files:
    - conversational_agent.py (agent that is MUCH more responsive to the user and their responses and is able to have smoother dialogue and even ask follow-up questions if necessary)
    - conversation_flow.py (used dynamic interview fow function to command the flow of how the interview works)
    - conversational_app.py (main entry point program to run this more dynamic interview with the more interactive interview agent)

Final Version:
- Version 2 Files + New Files:
    - voice_interface.py (commands the voice interview flow and gives the interviewer a voice and takes in audio information from interviewee's microphone when responding to interviewer questions)