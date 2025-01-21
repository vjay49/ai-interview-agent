from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

SYSTEM_TEMPLATE = """
You are an AI Interviewer and recruiting agent. You have the following context about requirements for your job, company information, and candidate background:

- JOB REQUIREMENTS:
{job_requirements}

- COMPANY PROFILE:
{company_profile}

- RESUME SUMMARY:
{resume_summary}

Your goal:
1) Ask the candidate questions about their experience, technical skills, and cultural fit relevant to what is needed to meet the job requirements and company values.
2) Do not repeat questions already asked.
3) If the candidate's prior response suggests a need for elaboration, ask a follow-up. Otherwise, move to another interview topic.
4) Keep a professional yet friendly tone.

Note: questions should try to distinguish if a candidate meets the technical and cultural fit requirements for the job.
"""

class ConversationalInterviewAgent:
    def __init__(self, openai_api_key: str, job_requirements: str, company_profile: str, resume_summary: str):
        """
        Initialize the chat LLM with memory, and store the combined system prompt.
        """
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=openai_api_key,
            model_name="gpt-4o"
        )
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Format system message with the provided context
        self.system_context = SYSTEM_TEMPLATE.format(
            job_requirements=job_requirements,
            company_profile=company_profile,
            resume_summary=resume_summary
        )

    def generate_next_prompt(self, user_input: str, is_first_turn: bool = False) -> str:
        """
        Generate the next question or statement from the AI interviewer, referencing any prior conversation in memory.
        """
        # 1) Convert memory’s chat messages into a list of LangChain message objects
        history_msgs = self.memory.chat_memory.messages
        messages_so_far = []
        for m in history_msgs:
            if m.type == "human":
                messages_so_far.append(HumanMessage(content=m.content))
            else:
                messages_so_far.append(AIMessage(content=m.content))

        # 2) The system message with job, company, and resume context
        system_message = SystemMessage(content=self.system_context)

        # 3) If it is not the first turn, we have an actual user answer to record
        if not is_first_turn and user_input.strip():
            messages_so_far.append(HumanMessage(content=user_input))

        # 4) add a directive to produce the next question or follow-up
        next_question_directive = SystemMessage(
            content="Please provide the next interview question or a follow-up question based on the conversation so far."
        )

        # Combine system context + chat history + final directive
        full_message_list = [system_message] + messages_so_far + [next_question_directive]

        # 5) ChatOpenAI to get next question
        response_message = self.llm.invoke(full_message_list)
        ai_text = response_message.content  # The AI's next question

        # 6) Store user turn in memory
        if not is_first_turn and user_input.strip():
            self.memory.save_context({"input": user_input}, {"output": ai_text})
        else:
            # First turn does not store user inpt
            pass

        return ai_text
