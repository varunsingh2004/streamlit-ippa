import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def load_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

def apply_fourier_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum.astype(np.uint8)

def apply_edge_detection(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    elif method == "Canny":
        return cv2.Canny(gray, 100, 200)
    return gray

def apply_smoothing(image, method, kernel_size):
    if method == "Gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "Median":
        return cv2.medianBlur(image, kernel_size)
    return image

def apply_custom_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def main():
    st.set_page_config(page_title="Image Processing App", layout="wide")
    st.title("Advanced Image Processing Application")

    # Sidebar
    st.sidebar.title("Processing Options")
    processing_option = st.sidebar.selectbox(
        "Select Processing Technique",
        ["Frequency Domain", "Edge Detection", "Smoothing", "Custom Convolution"]
    )

    # Main content
    uploaded_image = load_image()

    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_image, channels="BGR")

        with col2:
            st.subheader("Processed Image")
            
            if processing_option == "Frequency Domain":
                processed_image = apply_fourier_transform(uploaded_image)
                st.image(processed_image, clamp=True)
                
            elif processing_option == "Edge Detection":
                method = st.sidebar.selectbox("Edge Detection Method", ["Sobel", "Canny"])
                processed_image = apply_edge_detection(uploaded_image, method)
                st.image(processed_image, clamp=True)
                
            elif processing_option == "Smoothing":
                method = st.sidebar.selectbox("Smoothing Method", ["Gaussian", "Median"])
                kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
                processed_image = apply_smoothing(uploaded_image, method, kernel_size)
                st.image(processed_image, channels="BGR")
                
            elif processing_option == "Custom Convolution":
                st.sidebar.subheader("Custom Kernel (3x3)")
                kernel = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        kernel[i, j] = st.sidebar.number_input(f"Value [{i},{j}]", value=0.0, format="%.2f")
                
                if st.sidebar.button("Apply Convolution"):
                    kernel = kernel / kernel.sum() if kernel.sum() != 0 else kernel
                    processed_image = apply_custom_convolution(uploaded_image, kernel)
                    st.image(processed_image, channels="BGR")

            # Download button for processed image
            if 'processed_image' in locals():
                buf = io.BytesIO()
                if len(processed_image.shape) == 2:  # Grayscale
                    Image.fromarray(processed_image).save(buf, format="PNG")
                else:  # Color
                    Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
