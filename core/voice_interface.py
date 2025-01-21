import speech_recognition as sr
import pyttsx3

def speak_text(text: str):
    """
    Convert text to speech using pyttsx3.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 210) # Speed of speech
    engine.setProperty("volume", 1.0) # Volume
    engine.say(text)
    engine.runAndWait()

def listen_from_mic() -> str:
    """
    Listen from the default microphone and convert speech to text using the SpeechRecognition library.
    """
    r = sr.Recognizer()
    r.pause_threshold = 1.0
    r.non_speaking_duration = 0.5

    with sr.Microphone() as source:
        print("Listening... please speak now.")
        audio_data = r.listen(source, timeout=5)
    try:
        # Recognize speech using Google Speech API
        text = r.recognize_google(audio_data)
        print(f"[You said]: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def voice_interview_flow(agent, max_rounds=5):
    """
    Conducts a interview via microphone for user input and speaker output for agent responses. 
    """
    print("\n--- Starting Interview ---\n")

    # 1) First turn, no user input, so is_first_turn=True
    ai_question = agent.generate_next_prompt("", is_first_turn=True)
    print(f"[AI Interviewer]: {ai_question}")
    speak_text(ai_question)

    # 2) For subsequent rounds, capture speech input from user, feed to agent
    for i in range(max_rounds):
        # Listen for the candidate's answer
        user_answer = listen_from_mic()
        if user_answer.strip().lower() in ["quit", "exit", "stop"]:
            print("Interview ended by user.\n")
            speak_text("Interview ended by user.")
            break
        
        # Pass the recognized text to the agent
        if i < max_rounds - 1:
            ai_question = agent.generate_next_prompt(user_answer, is_first_turn=False)
            print(f"[AI Interviewer]: {ai_question}")
            speak_text(ai_question)

    print("Interview Complete.\n")
    speak_text("Interview complete.")
