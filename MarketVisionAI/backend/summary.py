# Using the Google Gemini API for summarization with a hardcoded API key
# Install: pip install google-genai

from google import genai  # Gemini API client library

# Hardcode your Gemini API key (not recommended for production!)
API_KEY = "AIzaSyAsxQ2l6tnGZr9LUq7PbZaFgxp4Wm6Js5c"

# Initialize Gemini API client with explicit API key
client = genai.Client(api_key=API_KEY)

# Call the Gemini API for summarization
response = client.models.generate_content(
    model="gemini-2.5-flash",  # Choose Gemini model
    contents="Here is a long article about climate change. Summarize it in 5 sentences."
)

# Print the Gemini API response (summarized text)
print(response.text)