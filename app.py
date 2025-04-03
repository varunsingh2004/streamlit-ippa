import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def load_image():
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], help="Supports JPG, JPEG, PNG")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Convert PIL Image to RGB numpy array
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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

def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def apply_adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def apply_unsharp_mask(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def display_image(image, is_gray=False):
    """Helper function to properly display images in Streamlit"""
    if is_gray:
        return st.image(image, clamp=True)
    else:
        return st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
def apply_filters(image):
    """Apply various filters to the image based on user selection."""
    filter_option = st.sidebar.selectbox(
        "Select Filter",
        ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Box Filter"]
    )

    kernel_size = st.sidebar.slider("Kernel Size", 1, 31, 5, step=2)

    if filter_option == "Gaussian Blur":
        processed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_option == "Median Blur":
        processed_image = cv2.medianBlur(image, kernel_size)
    elif filter_option == "Bilateral Filter":
        sigma_color = st.sidebar.slider("Sigma Color", 10, 150, 75)
        sigma_space = st.sidebar.slider("Sigma Space", 10, 150, 75)
        processed_image = cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)
    elif filter_option == "Box Filter":
        processed_image = cv2.boxFilter(image, -1, (kernel_size, kernel_size))

    return processed_image


def main():
    st.set_page_config(page_title="Advanced Image Processing", layout="wide")
    st.title("Image Processing Toolbox")
    st.markdown("Visualize image processing!")

    uploaded_image = load_image()
    processing_option = st.sidebar.selectbox(
        "Choose a Processing Technique",
        ["Frequency Domain", "Edge Detection", "Filters", "Histogram Equalization", "Sketchify", "Unsharp Masking"]
    )

    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            display_image(uploaded_image)

        with col2:
            st.subheader("Processed Image")
            processed_image = None
            is_gray = True  # Most operations return grayscale images

            if processing_option == "Frequency Domain":
                processed_image = apply_fourier_transform(uploaded_image)

            elif processing_option == "":
                st.subheader("Original Image")
                display_image(uploaded_image)

            elif processing_option == "Edge Detection":
                method = st.sidebar.selectbox("Method", ["Sobel", "Canny"])
                processed_image = apply_edge_detection(uploaded_image, method)

            elif processing_option == "Histogram Equalization":
                processed_image = apply_histogram_equalization(uploaded_image)

            elif processing_option == "Sketchify":
                processed_image = apply_adaptive_threshold(uploaded_image)

            elif processing_option == "Unsharp Masking":
                processed_image = apply_unsharp_mask(uploaded_image)
                is_gray = False  # Unsharp masking returns a color image

            elif processing_option == "Filters":
                processed_image = apply_filters(uploaded_image)  # Integration of apply_filters()
                is_gray = False  # Filters return color images

            if processed_image is not None:
                display_image(processed_image, is_gray)

                # Prepare image for download
                buf = io.BytesIO()
                if len(processed_image.shape) == 2:
                    Image.fromarray(processed_image).save(buf, format="PNG")
                else:
                    Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")

                st.download_button(
                    "Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
