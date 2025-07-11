# check_api_key.py
from langchain_groq import ChatGroq

# 🔐 Replace this with your actual Groq API key
api_key = "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"

# Debug info
print(f"API Key exists: {api_key is not None}")
print(f"API Key starts with: {api_key[:10]}...")
print(f"API Key length: {len(api_key)}")

# Try a basic LLM call
try:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192"
    )

    response = llm.invoke("Say hello")
    print("✅ API Key is valid!")
    print("Response:", response.content)

except Exception as e:
    print(f"❌ API Key test failed: {e}")
    print("\n🔧 Troubleshooting steps:")
    print("1. Visit https://console.groq.com/keys")
    print("2. Create or regenerate your API key")
    print("3. Paste it directly into this script or your .env file")
