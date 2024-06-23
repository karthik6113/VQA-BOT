import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# Load the VQA model and processor
model_name = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForQuestionAnswering.from_pretrained(model_name)

def process_image(image):
    """Convert image to RGB format"""
    image = Image.open(image).convert('RGB')
    return image

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

                # Preprocess text and image using processor
                inputs = processor(images=image, text=question, return_tensors="pt")

                # Perform prediction with the model
                outputs = model(**inputs)

                # Extract the answer from the model's output
                answer = processor.decode(outputs.logits.argmax(-1).item()).strip()
                confidence = outputs.logits.softmax(-1).max().item() * 100

                st.write(f"Answer: {answer} (Confidence: {confidence:.2f}%)")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.write("Please enter a question.")
    else:
        st.write("Please upload an image and ask a question.")

if __name__ == "__main__":
    main()
