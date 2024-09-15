import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv('gemini_key'))

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

def analyze_image(image_path):
    # Ensure the uploaded file is one of the allowed types
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in allowed_extensions:
        return "Unsupported file type. Please upload a PNG, JPG, JPEG, or WEBP image."

    # Upload the image to Gemini
    uploaded_file_info = upload_to_gemini(image_path, mime_type="image/png")

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
    return response.text

# Gradio interface
iface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Textbox(label="Analysis Results"),
    title="👨‍⚕️ Vital Image 📷 Analytics",
    description="Upload a medical image to get a detailed analysis.",
)

if __name__ == "__main__":
    iface.launch()
