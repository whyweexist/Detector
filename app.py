import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv('gemini_key'))

# Streamlit app settings
st.set_page_config(page_title="Disease Analytics", page_icon=":robot")

# Prompt template for detailed analysis
system_prompt = """ 

As a Highly Skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital.your expertise
is crucial in idetifying any anomalies, diseases or health issues that maybe present in the images.

Your Responsibilities:
1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Finding reports: Document all observed anomalies or signs of diseases. Clearly articulate these findings in a Structured form.
3. Recommendations and Next steps: Based on your analysis, suggest potential next steps, including further tests or treatment as applicable.
4. if appropriate, recommend possible treatment options or interventions.

Important notes:
Scope of response: only respond if the image is related to human health.
Clarity of image: In cases where the image quality impedes clear analysis, note that certian aspects are 'Unable to be determind based on the provided image.' 
"""

# Configuration for the generative model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Create the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# Streamlit app title and description
st.title("👨‍⚕️Vital Image 📷Analytics")
st.subheader("We Will Help You Identify Medical Images!")

# File uploader for medical images
uploaded_file = st.file_uploader("Upload Medical Image to Analyze", type=["png", "jpg", "jpeg", "webp"])

# Button to generate analysis
submit_button = st.button("Generate Analysis")

if submit_button and uploaded_file:
    
    st.image(uploaded_file, caption="Uploaded Medical Image", use_column_width=True,width=200)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Upload the image to Gemini
    uploaded_file_info = upload_to_gemini(temp_file_path, mime_type=uploaded_file.type)

    # Create a chat session with the model and send the image for analysis
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    uploaded_file_info,
                    system_prompt,
                ],
            },
        ]
    )

    response = chat_session.send_message(system_prompt)

    # Display the analysis results
    st.subheader("Analysis Results")
    st.write(response.text)
