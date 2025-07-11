import streamlit as st
import os
import re
import pyttsx3  # For offline pronunciation (pyttsx3 used for offline, gTTS for online)
from gtts import gTTS  # For Google Text-to-Speech (online)
from io import BytesIO
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from nltk import pos_tag, word_tokenize
from eng_to_ipa import convert as ipa_convert
from nltk.corpus import wordnet as wn

# === FAISS Path ===
FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")

# === NLTK Setup ===
nltk_path = os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Roaming", "nltk_data")
os.environ["NLTK_DATA"] = nltk_path

import nltk
if os.path.exists(nltk_path) and nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_path)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"❌ NLTK download failed: {e}")
        return False

download_nltk_data()

# === Helper Functions for Pronunciation Playback ===
def play_pronunciation(word, accent="en"):
    try:
        audio_fp = BytesIO()
        # Online TTS using gTTS (Google TTS)
        tts = gTTS(text=word, lang=accent)  # Accent option passed here
        tts.save("temp_audio.mp3")
        with open("temp_audio.mp3", "rb") as f:
            audio_fp.write(f.read())
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"❌ Pronunciation playback failed: {e}")
        return None

# === Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
    You are a dictionary assistant. Provide detailed word information including definitions, examples, tenses, parts of speech, and more.
    Format response as:
    Word: [word]
    Definition: [meaning]
    Examples:
    - [example 1]
    - [example 2]
    
    Tenses:
    - Present: [present tense]
    - Past: [past tense]
    - Future: [future tense]

    Suffixes: [list of suffixes]
    Prefixes: [list of prefixes]
    Synonyms: [list of synonyms]
    Antonyms: [list of antonyms]
    Parts of Speech: [parts of speech]
    Vowels: [list of vowels]

    Pronunciation: [IPA with breakage]
    
    Context: {context}
    Question: {question}

    Answer:""".strip()
)

# === Helper Functions for Word Details ===
def extract_target_word(query):
    try:
        match = re.search(r"(define|meaning of|what is)\s+(\w+)", query.lower())
        return match.group(2) if match else query.strip().split()[-1]
    except:
        return query.strip()

def safe_pos_tag(word):
    try:
        pos_tagged = pos_tag(word_tokenize(word))
        return pos_tagged[0][1] if pos_tagged else "UNK"
    except:
        return "UNK"

def safe_ipa_convert(word):
    try:
        return ipa_convert(word)
    except:
        return word

def get_tenses(word):
    tenses = {
        'present': wn.morphy(word, wn.VERB) if wn.morphy(word, wn.VERB) else word,
        'past': wn.morphy(word, wn.VERB),
        'future': f"will {word}"
    }
    if tenses['past'] is None:
        tenses['past'] = word + "ed"  # Default past tense if WordNet can't detect it
    return tenses

def get_affixes(word):
    suffixes = ['ed', 'ing', 's', 'ly', 'ness', 'ment', 'ation', 'able', 'ive', 'ful']
    prefixes = ['un', 're', 'pre', 'dis', 'mis', 'non', 'in', 'over', 'under']
    suffixes_found = [suffix for suffix in suffixes if word.endswith(suffix)]
    prefixes_found = [prefix for prefix in prefixes if word.startswith(prefix)]
    return suffixes_found, prefixes_found

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def get_antonyms(word):
    antonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    return list(antonyms)

def get_vowels(word):
    vowels = "aeiou"
    return [char for char in word if char in vowels]

# === LangChain imports ===
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True).as_retriever()

def load_chain():
    try:
        llm = ChatGroq(
            groq_api_key="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",
            model_name="llama3-8b-8192"
        )
        retriever = load_retriever()
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            combine_docs_chain_kwargs={"prompt": prompt_template},
            verbose=True
        )
    except Exception as e:
        st.error(f"❌ Failed to load chain: {e}")
        return None

# === Streamlit UI ===
st.set_page_config(page_title="Oxford Dictionary AI", page_icon="📘")
st.title("📘 Oxford Dictionary Assistant")
st.markdown("Ask about a word — its definition, usage, tenses, synonyms, and more.")

# Initialize session state if needed
if "chat_chain" not in st.session_state:
    st.write("⏳ Initializing conversational chain...")
    st.session_state.chat_chain = load_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.chat_chain is None:
    st.error("❌ Could not initialize the assistant. Please check your configuration.")
    st.stop()

# Input for user query
user_input = st.text_input("🔍 Your query:", placeholder="e.g., What is the past tense of 'abandon'?")

# Handle user input and fetch results
if user_input:
    st.write("⏳ Sending query to LLM...")
    with st.spinner("Thinking... 💭"):
        try:
            word = extract_target_word(user_input)
            pos = safe_pos_tag(word)
            ipa = safe_ipa_convert(word)

            # Get tenses, affixes, synonyms, antonyms, vowels
            tenses = get_tenses(word)
            suffixes, prefixes = get_affixes(word)
            synonyms = get_synonyms(word)
            antonyms = get_antonyms(word)
            vowels = get_vowels(word)

            # Combine POS and IPA into the question string
            enhanced_question = f"{user_input.strip()} (POS: {pos}, IPA: /{ipa}/)"

            result = st.session_state.chat_chain.invoke({
                "question": enhanced_question
            })

            # Format detailed response
            response = f"\nResponse:\n{result['answer']}\n\n"
            response += f"Tenses:\n- Present: {tenses['present']}\n- Past: {tenses['past']}\n- Future: {tenses['future']}\n"
            response += f"Suffixes: {suffixes}\nPrefixes: {prefixes}\nSynonyms: {synonyms}\nAntonyms: {antonyms}\n"
            response += f"Parts of Speech: {pos}\nVowels: {vowels}\nPronunciation: /{ipa}/"

            # Add pronunciation playback (choose accent)
            accent_choice = st.radio("Choose Accent:", ["American (en-us)", "British (en-uk)"])
            audio_fp = play_pronunciation(word, accent="en-us" if accent_choice == "American (en-us)" else "en-uk")
            if audio_fp:
                st.audio(audio_fp, format="audio/mp3")

            # Add to bookmarks if user clicks on bookmark
            if st.button(f"Bookmark '{word}'"):
                if word not in st.session_state.bookmarked_words:
                    st.session_state.bookmarked_words.append(word)
                    st.success(f"✅ '{word}' has been bookmarked!")

            st.session_state.chat_history.append((user_input, response))
            st.success("✅ Answer received.")
        except Exception as e:
            st.error(f"❌ An error occurred:\n\n{e}")

# Display bookmarked words
if "bookmarked_words" in st.session_state and st.session_state.bookmarked_words:
    st.markdown("### 📚 Bookmarked Words")
    for i, word in enumerate(st.session_state.bookmarked_words, 1):
        st.markdown(f"{i}. **{word}**")

# Show conversation history
if st.session_state.chat_history:
    st.markdown("### 🧠 Conversation History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**{i}. Q: {q}**")
        st.markdown(f"<div style='padding-left: 15px; color: #333;'>{a}</div>", unsafe_allow_html=True)
        st.markdown("---")

# Footer
st.markdown("<small>Built with 💬 LangChain + 🔍 FAISS + 🤗 HuggingFace/Groq + 🌐 Streamlit</small>", unsafe_allow_html=True)