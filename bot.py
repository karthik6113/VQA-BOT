import streamlit as st
from transformers import ViLTFeatureExtractor, ViLTokenizer, ViLForQuestionAnswering
from PIL import Image
import torch
import sounddevice as sd
import numpy as np

# Load the VQA model and tokenizer
model_name = "dandelin/vilt-b32-finetuned-vqa"
feature_extractor = ViLTFeatureExtractor.from_pretrained(model_name)
tokenizer = ViLTokenizer.from_pretrained(model_name)
model = ViLForQuestionAnswering.from_pretrained(model_name)

def process_image(image):
    """Resize and convert image to a format the model accepts"""
    image = Image.open(image).convert('RGB').resize((224, 224))
    return image

def text_to_speech(text):
    """Convert text to speech using sounddevice"""
    try:
        st.write(f"Answer: {text}")

        # Use sounddevice to play audio
        with sd.OutputStream() as stream:
            # Convert text to speech audio here (not implemented)
            # For example, you can use TTS (text-to-speech) libraries like pyttsx3, gTTS, etc.
            # Replace the following line with the appropriate TTS library usage.
            audio_data = np.zeros((10000, 2))  # Example placeholder for audio data
            stream.write(audio_data)

    except Exception as e:
        st.error(f"Error during audio playback: {e}")

def main():
    st.title("Visual Question Answering Bot")

    # Upload image section
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # Question input section
    question = st.text_input("Ask a question about the image:")

    # Submit button and processing
    if st.button("Ask") and uploaded_image is not None:
        if question:
            try:
                # Process the image
                image = process_image(uploaded_image)

                # Encode the image and question into features
                inputs = feature_extractor(images=image, text=question, return_tensors="pt")

                # Perform prediction with the model
                with torch.no_grad():
                    outputs = model(**inputs)

                # Decode the output to get the answer
                answer = tokenizer.decode(outputs['answer'][0])

                # Output answer in audio
                text_to_speech(answer)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.write("Please enter a question.")
    else:
        st.write("Please upload an image and ask a question.")

if __name__ == "__main__":
    main()
