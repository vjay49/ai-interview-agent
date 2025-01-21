def run_interview(questions: list) -> dict:
    """
    A simple command-line interview flow.
    Returns a dict of question->candidate answer.
    """
    answers = {}
    for idx, question in enumerate(questions, 1):
        print(f"AI Interviewer (Q{idx}): {question}")
        candidate_answer = input("Candidate: ")
        answers[question] = candidate_answer
    return answers



def dynamic_interview_flow(agent, max_rounds=5):
    """
    Conducts an interactive interview with up to max_rounds. The AI starts by asking an initial question.
    """
    print("\n--- Starting Interview ---\n")

    # 1) First turn, no user input, so is_first_turn=True
    ai_question = agent.generate_next_prompt("", is_first_turn=True)
    print(f"AI Interviewer: {ai_question}\n")

    # 2) For next turns, capture user input, and pass to agent
    for i in range(max_rounds):
        user_answer = input("Candidate: ")
        if user_answer.lower() in ["quit", "exit", "stop"]:
            print("Interview ended by user.\n")
            break
        
        # Agent references entire conversation  and new user input
        if i < max_rounds - 1:
            ai_question = agent.generate_next_prompt(user_answer, is_first_turn=False)
            print(f"AI Interviewer: {ai_question}\n")

    print("Interview Complete.\n")



