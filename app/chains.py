import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import nltk
from nltk import pos_tag, word_tokenize
from eng_to_ipa import convert as ipa_convert
import traceback
from nltk.corpus import wordnet as wn

# === Phase 2: NLTK Setup ===
nltk_path = os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Roaming", "nltk_data")
if nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_path)

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)

# === Hardcoded API Key ===
GROQ_API_KEY = "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"  # ← your working key

# === Paths ===
BASE_DIR = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data"
FAISS_DIR = os.path.join(BASE_DIR, "index")

# === Helper Functions ===
def get_pos(word):
    try:
        tagged = pos_tag(word_tokenize(word))
        pos = tagged[0][1] if tagged else "N/A"
        # Handling specific POS cases
        if pos.startswith("V"):  # Verbs
            return "verb"
        elif pos.startswith("N"):  # Nouns
            return "noun"
        elif pos.startswith("J"):  # Adjectives
            return "adjective"
        elif pos.startswith("R"):  # Adverbs
            return "adverb"
        return "unknown"
    except Exception as e:
        print(f"⚠️ POS tagging error: {e}")
        return "N/A"

def get_ipa(word):
    try:
        return ipa_convert(word)
    except Exception as e:
        print(f"⚠️ IPA conversion error: {e}")
        return word

def get_tenses(word):
    # Using WordNet to handle verb tense variations
    tenses = {
        'present': wn.morphy(word, wn.VERB) if wn.morphy(word, wn.VERB) else word,
        'past': wn.morphy(word, wn.VERB),
        'future': f"will {word}"
    }
    if tenses['past'] is None:
        tenses['past'] = word + "ed"  # Default past tense if WordNet can't detect it
    return tenses

def get_affixes(word):
    # Expanding list of common suffixes and prefixes for better detection
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

# === Load Retriever ===
def load_retriever():
    print("🔍 Loading FAISS index...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True).as_retriever()
    except Exception as e:
        print(f"❌ Retriever error: {e}")
        raise

# === Load LLM with hardcoded key ===
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192"
    )

# === Get Conversational Chain ===
def get_conversational_chain():
    return ConversationalRetrievalChain.from_llm(
        llm=load_llm(),
        retriever=load_retriever(),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=True
    )

# === CLI Test ===
if __name__ == "__main__":
    print("🚀 Testing Assistant...")
    try:
        word = "abandon"
        pos = get_pos(word)
        ipa = get_ipa(word)
        tenses = get_tenses(word)
        suffixes, prefixes = get_affixes(word)
        synonyms = get_synonyms(word)
        antonyms = get_antonyms(word)
        vowels = get_vowels(word)

        chain = get_conversational_chain()
        result = chain.invoke({
            "question": f"Define '{word}' (POS: {pos}, IPA: /{ipa}/)"
        })
        
        # Adding the word's extra details to the response
        response = f"\nResponse:\n{result['answer']}\n\n"
        response += f"Tenses:\n- Present: {tenses['present']}\n- Past: {tenses['past']}\n- Future: {tenses['future']}\n"
        response += f"Suffixes: {suffixes}\nPrefixes: {prefixes}\nSynonyms: {synonyms}\nAntonyms: {antonyms}\n"
        response += f"Parts of Speech: {pos}\nVowels: {vowels}\nPronunciation: /{ipa}/"
        
        print(response)
    except Exception as e:
        print("❌ Main error:")
        traceback.print_exc()
