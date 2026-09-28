import json
import re
from typing import Dict, Any, Optional
import ollama
from memory import TutorMemory

SYSTEM_PROMPT = """You are Elena, a world-class, charismatic, and encouraging English Spoken Communication Coach.
Your mission is to help the user speak fluent, confident, natural English through interactive conversation.

Guidelines for your Persona & Teaching:
1. Speak Suggestions & Corrections Aloud: When the user makes a grammar mistake or awkward phrasing, Elena MUST verbally explain the tip in her spoken response in a warm, friendly way (for example: "Quick tip: instead of 'X', you can say 'Y' because..."). Then answer their point and end with an open-ended question.
2. Spoken Conversational Cadence: Keep spoken replies natural, engaging, and clear (2 to 4 sentences).
3. Actionable Improvement Points: Provide 2-3 specific, constructive bullet points in the JSON.
4. Positive Reinforcement: Always encourage the student and build their confidence!

{memory_context}

You MUST respond in valid JSON format with the following keys:
{
  "spoken_response": "The natural, conversational text that Elena will speak aloud to the student (2-4 sentences max, friendly and engaging).",
  "improvement_points": [
    "Specific improvement point 1 (e.g., grammar or tense adjustment)",
    "Specific improvement point 2 (e.g., pronunciation/stress or natural idiom tip)"
  ],
  "has_correction": true/false,
  "original_mistake": "The exact phrase the user said that had a grammar or phrasing issue, or null",
  "corrected_version": "The natural, correct English phrase, or null",
  "correction_explanation": "Brief, friendly 1-sentence explanation of why, or null",
  "better_phrasing": "A more natural, native-sounding alternative to how the user phrased something, or null",
  "new_vocabulary": {
    "word_or_idiom": "A useful vocabulary word or idiom related to the topic, or null",
    "definition": "Simple definition",
    "example": "An example sentence"
  },
  "fluency_tip": "A quick spoken English tip on intonation, stress, or connected speech (optional, or null)"
}
"""

def clean_spoken_text(text: str) -> str:
    if not text:
        return "That's a great thought! Tell me a little more about it."
    
    cleaned = text.strip()
    # Unwrap markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # If it contains JSON structure, extract the actual spoken value
    if cleaned.startswith("{") or '"spoken_response"' in cleaned:
        match = re.search(r'"spoken_response"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        if match:
            cleaned = match.group(1)
        else:
            # Remove JSON key remnants
            cleaned = re.sub(r'\{.*?"spoken_response"\s*:\s*', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'"\s*,\s*"[a-zA-Z_]+".*$', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'[{}\[\]"]', '', cleaned)

    # Unescape common sequences
    cleaned = cleaned.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
    # Strip any accidental brackets or stray JSON keys
    cleaned = re.sub(r'^\s*[\{\[\"]+', '', cleaned)
    cleaned = re.sub(r'[\}\]\"]+\s*$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned if cleaned else "That sounds great! How do you feel about practicing this further?"

class EnglishTutor:
    def __init__(self, model_name: str = "gpt-oss:120b-cloud", db_path: str = "tutor_memory.db"):
        self.model_name = model_name
        self.memory = TutorMemory(db_path)
        self.client = ollama.Client()

    def get_available_models(self):
        try:
            models_info = self.client.list()
            return [m.get('name', m.get('model')) for m in models_info.get('models', [])]
        except Exception:
            return ["gpt-oss:120b-cloud", "gemma4:latest"]

    def chat(self, user_input: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """Processes user speech/text, returns structured tutor feedback & spoken reply."""
        # 1. Fetch memory summary
        memory_summary = self.memory.get_memory_summary_for_prompt()
        system_instruction = SYSTEM_PROMPT.replace("{memory_context}", f"STUDENT CONTEXT & MEMORY:\n{memory_summary}")
        
        if topic:
            system_instruction += f"\nCURRENT TOPIC / SCENARIO: {topic}"

        # 2. Build message list
        messages = [{"role": "system", "content": system_instruction}]
        
        # Recent history
        recent = self.memory.get_recent_history(limit=5)
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Current user input
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.7, "num_ctx": 2048, "num_predict": 450},
                format="json"
            )
            raw_content = response['message']['content']
            parsed = self._parse_json_response(raw_content)
        except Exception as e:
            print(f"[Tutor] Ollama chat error: {e}")
            parsed = {
                "spoken_response": "That's a really interesting thought! Could you tell me a little more about that?",
                "improvement_points": ["Speak in complete, natural sentences", "Maintain a steady conversational pace"],
                "has_correction": False,
                "original_mistake": None,
                "corrected_version": None,
                "correction_explanation": None,
                "better_phrasing": None,
                "new_vocabulary": None,
                "fluency_tip": "Keep speaking continuously without worrying about minor slips!"
            }

        # Ensure spoken_response is 100% clean plain human speech (never JSON)
        parsed["spoken_response"] = clean_spoken_text(parsed.get("spoken_response", ""))

        # 3. Update Memory
        self.memory.add_message(role="user", content=user_input)
        self.memory.add_message(
            role="assistant",
            content=parsed["spoken_response"],
            feedback=parsed
        )

        # Record mistake if detected
        if parsed.get("has_correction") and parsed.get("original_mistake") and parsed.get("corrected_version"):
            self.memory.record_mistake(
                original=parsed["original_mistake"],
                corrected=parsed["corrected_version"],
                explanation=parsed.get("correction_explanation", ""),
                category="grammar"
            )

        # Record new vocabulary if taught
        new_vocab = parsed.get("new_vocabulary")
        if new_vocab and isinstance(new_vocab, dict) and new_vocab.get("word_or_idiom"):
            self.memory.add_vocabulary(
                word=new_vocab["word_or_idiom"],
                definition=new_vocab.get("definition", ""),
                example=new_vocab.get("example", "")
            )

        return parsed

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Pass 1: standard JSON parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Ensure spoken_response is sanitized
                data["spoken_response"] = clean_spoken_text(data.get("spoken_response", ""))
                return data
        except Exception:
            pass

        # Pass 2: outer regex match
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    data["spoken_response"] = clean_spoken_text(data.get("spoken_response", ""))
                    return data
            except Exception:
                pass

        # Pass 3: resilient key-by-key regex extraction
        spoken_match = re.search(r'"spoken_response"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        spoken = spoken_match.group(1) if spoken_match else ""

        pts = []
        pts_match = re.search(r'"improvement_points"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        if pts_match:
            pts = re.findall(r'"((?:[^"\\]|\\.)*)"', pts_match.group(1))

        orig_mistake = re.search(r'"original_mistake"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        corr_version = re.search(r'"corrected_version"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        corr_exp = re.search(r'"correction_explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        better_phr = re.search(r'"better_phrasing"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)

        return {
            "spoken_response": clean_spoken_text(spoken),
            "improvement_points": pts if pts else ["Express your ideas with natural sentence rhythm", "Use full, connected phrases"],
            "has_correction": bool(orig_mistake and corr_version),
            "original_mistake": orig_mistake.group(1) if orig_mistake else None,
            "corrected_version": corr_version.group(1) if corr_version else None,
            "correction_explanation": corr_exp.group(1) if corr_exp else None,
            "better_phrasing": better_phr.group(1) if better_phr else None,
            "new_vocabulary": None,
            "fluency_tip": "Keep speaking continuously without worrying about minor slips!"
        }
