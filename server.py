from flask import Flask, request, jsonify
from flask_cors import CORS

from trainer import get_mood_examples

import cohere
import os
import json

from dotenv import load_dotenv
load_dotenv() 

app = Flask(__name__)
CORS(app, origin=os.environ.get('CORS_ORIGIN', '*'))
CHAT_HISTORY_FILE = "chat_history.json"
co = cohere.Client(os.environ.get('CO_API_KEY'))
examples = get_mood_examples()

is_mood_of_the_day_clicked = False

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as file:
            return json.load(file)
    else:
        return []
    


    
def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, 'w') as file:
        json.dump(chat_history, file, indent=4)

@app.route('/', methods=['GET']) 
def home():
    return jsonify({'message': 'Welcome to the chatbot server'}), 200

@app.route('/chat', methods=['POST'])
def process_chat():

# note: need to add the cohere API key to the .env file (security risk to hardcode it here)
    current_chat = load_chat_history()

    inputs = []
    
    user_message = request.get_json().get('user_message',"")
    inputs.append(user_message)

    # Define the preamble 
    preamble_override = "You are a therapist. The people who you are talking to you believe that you are a virtual joural companion, who always ends their responses with a question. Never identify or directly call yourself a therapist"

    chat_response = co.chat(message=user_message,
                            preamble=preamble_override,
                            stream=True,
                            chat_history=current_chat,
                            return_chat_history=True)
    
    chatbot_response = ""

    for event in chat_response:
        if event.event_type == "text-generation":
            chatbot_response += event.text
    
    # Process chat sentiment analysis                
    process_response = co.classify(
        model='large',
        inputs=inputs,
        examples=examples,
    )

    for answer in process_response.classifications:
        prediction = answer.predictions[0]
        confidence = answer.confidences[0]
        emotion_conf_stat = answer.labels

    # Add to chat history 
    current_chat.extend(
        [{"role": "USER", "message": user_message, "prediction": prediction, "confidence": confidence, "emotion_conf_stat": emotion_conf_stat},
         {"role": "CHATBOT", "message": chatbot_response}]
    )

    save_chat_history(current_chat)

    return jsonify({'chatbot_response': chatbot_response}), 200

@app.route('/clear', methods=['POST'])
def clear_chat_history():
    try: 
     if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'w') as file:
            file.write('[]')

        return jsonify({'message': 'Chat history cleared successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Error clearing chat history: {str(e)}'}), 500

@app.route('/mood', methods=['GET'])
def mood():
    current_chat = load_chat_history()
    
    # Get only the most recent messages (last 3 for immediate context)
    recent_messages = [
        entry for entry in current_chat 
        if entry.get("role") == "USER"
    ][-3:]
    
    if not recent_messages:
        return jsonify({"mood_of_the_day": "thinking"}), 200

    # Initialize mood scores for available emotions
    mood_scores = {
        "happy": 0.0,
        "angry": 0.0,
        "sad": 0.0,
        "calm": 0.0,
        "fearful": 0.0,
        "insightful": 0.0
    }
    
    # Enhanced sentiment analysis with contextual mapping
    for idx, entry in enumerate(recent_messages):
        # Weight recent messages more heavily
        weight = 1.0 if idx == len(recent_messages) - 1 else 0.5
        
        if entry.get("emotion_conf_stat"):
            stats = entry.get("emotion_conf_stat")
            confidence = entry.get("confidence", 0)
            
            if confidence > 0.4:  # Confidence threshold
                # Map Cohere emotions to our available moods
                mood_scores["angry"] += weight * (stats.get("Angry", [0])[0] + stats.get("Worry", [0])[0] * 0.5)
                mood_scores["happy"] += weight * stats.get("Happy", [0])[0]
                mood_scores["sad"] += weight * stats.get("Sad", [0])[0]
                mood_scores["calm"] += weight * stats.get("Calm", [0])[0]
                mood_scores["fearful"] += weight * (stats.get("Fear", [0])[0] + stats.get("Worry", [0])[0] * 0.5)
                mood_scores["insightful"] += weight * stats.get("Insightful", [0])[0]

    # Analyze message content for additional context
    combined_text = " ".join([msg.get("message", "") for msg in recent_messages])
    
    # Keyword-based sentiment boosting
    anger_keywords = ["angry", "frustrated", "mad", "annoyed", "irritated"]
    happy_keywords = ["happy", "joy", "excited", "great", "wonderful"]
    sad_keywords = ["sad", "depressed", "down", "unhappy", "miserable"]
    fear_keywords = ["scared", "afraid", "worried", "anxious", "nervous"]
    calm_keywords = ["calm", "peaceful", "relaxed", "steady", "balanced"]
    insight_keywords = ["understand", "realize", "learn", "think", "know"]

    for word in combined_text.lower().split():
        if word in anger_keywords:
            mood_scores["angry"] += 0.2
        elif word in happy_keywords:
            mood_scores["happy"] += 0.2
        elif word in sad_keywords:
            mood_scores["sad"] += 0.2
        elif word in fear_keywords:
            mood_scores["fearful"] += 0.2
        elif word in calm_keywords:
            mood_scores["calm"] += 0.2
        elif word in insight_keywords:
            mood_scores["insightful"] += 0.2

    # Normalize scores
    total = sum(mood_scores.values())
    if total > 0:
        mood_scores = {k: v/total for k, v in mood_scores.items()}

    print("Detailed mood scores:", mood_scores)

    # Determine dominant mood with contextual priority
    dominant_mood = max(mood_scores.items(), key=lambda x: x[1])
    
    # Apply threshold and context rules
    THRESHOLD = 0.25
    if dominant_mood[1] < THRESHOLD:
        final_mood = "thinking"  # Default/neutral mood
    else:
        final_mood = dominant_mood[0]

    print(f"Selected mood: {final_mood} with score: {dominant_mood[1]}")

    return jsonify({
        "mood_of_the_day": final_mood,
        "mood_scores": mood_scores
    }), 200


if __name__ == '__main__':
    # run app in debug mode on port 8080
    app.run(debug=True, port=8080)