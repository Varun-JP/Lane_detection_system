import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from moviepy.editor import VideoFileClip

from reference_files.detection import process_image


def set_page_config() -> None:
    st.set_page_config(
        page_title="Road Lane Detection Demo",
        layout="wide",
    )


def render_landing_section() -> None:
    st.title("Road Lane Detection System")
    st.markdown(
        """
This app wraps a classical **computer vision lane detection pipeline** in a modern web UI.

**What it does**
- Detects lane markings on **images**, **pre-recorded videos**, and **webcam snapshots**.
- Uses **Canny edge detection**, a **trapezoidal region of interest**, and **Hough line transforms**
  with temporal smoothing to keep lane lines stable across frames.

Use the sidebar to pick a mode and try it on your own data.
"""
    )

    with st.expander("How the algorithm works"):
        st.markdown(
            """
1. **Blur & edges** – Apply Gaussian blur and Canny edge detection to highlight strong edges.
2. **Region of interest** – Keep only a trapezoidal region that roughly covers the road ahead.
3. **Hough transform** – Detect line segments corresponding to left and right lane markings.
4. **Smoothing** – Track the dominant left/right lines over time with a moving average.
5. **Overlay** – Draw the final lane lines on top of the original frame.
"""
        )

    with st.expander("Limitations & future work"):
        st.markdown(
            """
- Works best on **daytime highway footage** with clear lane markings.
- Can struggle with **heavy rain, snow, shadows, or worn-out paint**.
- Future extensions could:
  - Add a **deep-learning-based** lane detector.
  - Estimate **curvature** and **vehicle offset** from lane center.
"""
        )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def handle_image_mode() -> None:
    st.subheader("Image lane detection")
    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    if uploaded_file is None:
        st.info("Upload an image to see lane detection results.")
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr_image is None:
        st.error("Could not read the uploaded image. Please try a different file.")
        return

    start_time = time.perf_counter()
    processed_bgr = process_image(bgr_image)
    elapsed = time.perf_counter() - start_time

    original_rgb = bgr_to_rgb(bgr_image)
    processed_rgb = bgr_to_rgb(processed_bgr)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input image**")
        st.image(original_rgb, use_column_width=True)
    with col2:
        st.markdown("**Detected lanes**")
        st.image(processed_rgb, use_column_width=True)

    st.caption(f"Processed in {elapsed * 1000:.1f} ms")


def handle_video_mode() -> None:
    st.subheader("Video lane detection")
    uploaded_video = st.file_uploader(
        "Upload a short driving video (a few seconds works best)",
        type=["mp4", "avi", "mov"],
    )

    if uploaded_video is None:
        st.info("Upload a video to see lane detection over time.")
        return

    # Save uploaded video to a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as temp_input:
        temp_input.write(uploaded_video.read())
        input_path = temp_input.name

    st.write("Processing video, this may take a moment...")

    def process_frame(frame: np.ndarray) -> np.ndarray:
        """
        MoviePy provides frames as RGB. Convert to BGR for the OpenCV-based
        pipeline, run process_image, then convert back to RGB for MoviePy.
        """
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processed_bgr = process_image(bgr)
        return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    start_time = time.perf_counter()
    clip = VideoFileClip(input_path)
    processed_clip = clip.fl_image(process_frame)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    # Write processed video using MoviePy/ffmpeg; suppress verbose logging
    processed_clip.write_videofile(output_path, audio=False, verbose=False, logger=None)

    elapsed = time.perf_counter() - start_time

    st.markdown("**Processed video with lane overlays**")
    st.caption(f"Finished processing in {elapsed:.1f} seconds.")
    with open(output_path, "rb") as f:
        st.video(f.read())


def handle_webcam_mode() -> None:
    st.subheader("Webcam snapshot lane detection")
    st.info("Capture a snapshot from your webcam and see lanes overlaid on the image.")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is None:
        return

    image = Image.open(img_file_buffer)
    image_rgb = np.array(image)
    image_bgr = rgb_to_bgr(image_rgb)

    start_time = time.perf_counter()
    processed_bgr = process_image(image_bgr)
    elapsed = time.perf_counter() - start_time

    processed_rgb = bgr_to_rgb(processed_bgr)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Webcam snapshot**")
        st.image(image_rgb, use_column_width=True)
    with col2:
        st.markdown("**Detected lanes**")
        st.image(processed_rgb, use_column_width=True)

    st.caption(f"Processed in {elapsed * 1000:.1f} ms")


def main() -> None:
    set_page_config()
    render_landing_section()

    st.sidebar.header("Demo controls")
    mode = st.sidebar.radio(
        "Choose input type",
        ("Image", "Video", "Webcam snapshot"),
    )

    if mode == "Image":
        handle_image_mode()
    elif mode == "Video":
        handle_video_mode()
    else:
        handle_webcam_mode()


if __name__ == "__main__":
    main()

