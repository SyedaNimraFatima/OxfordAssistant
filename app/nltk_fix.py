# nltk_fix.py - Run this script to fix NLTK issues

import os
import nltk
import ssl

# Handle SSL certificate issues (common on Windows)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path
nltk_path = os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Roaming', 'nltk_data')
os.makedirs(nltk_path, exist_ok=True)

if nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_path)

print(f"📂 NLTK data path: {nltk_path}")

# List of required packages
required_packages = [
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

print("🔄 Downloading NLTK packages...")

for package in required_packages:
    try:
        nltk.download(package, download_dir=nltk_path, quiet=False)
        print(f"✅ {package} downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download {package}: {e}")

print("\n🔍 Checking downloaded packages...")
for package in required_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'taggers/{package}')
        print(f"✅ {package} found")
    except LookupError:
        print(f"❌ {package} not found")

print("\n🧪 Testing tokenization...")
try:
    from nltk import word_tokenize, pos_tag
    test_text = "hello world"
    tokens = word_tokenize(test_text)
    tags = pos_tag(tokens)
    print(f"✅ Tokenization test passed: {tokens} -> {tags}")
except Exception as e:
    print(f"❌ Tokenization test failed: {e}")

print("\n✅ NLTK setup complete!")