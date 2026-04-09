import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageStat
from io import BytesIO
import pytesseract
from skimage.metrics import structural_similarity as ssim




og_id_path= "og.jpeg" 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="ID Fraud Detector", layout="centered")


def normalize(value,low,high):
    return max(0,min(1,(value - low)/(high - low))) if high > low else 0

def score_to_level(score):
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    return "LOW"

def validate_image(image):
    width, height = image.size
    return width >= 300 and height >= 200


def ela_check(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    recompressed = Image.open(BytesIO(buffer.getvalue()))

    ela = ImageChops.difference(image, recompressed)
    mean_diff = sum(ImageStat.Stat(ela).mean) / 3

    score = normalize(mean_diff, 2, 25)
    return score, ela

def blur_check(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()

    score = 1 - normalize(var, 60, 600)
    return score

def edge_check(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    density = np.mean(edges > 0)
    score = normalize(density, 0.18, 0.45)

    return score, edges


def extract_text(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()


def compare_with_original(test_img):
    try:
        original_img = Image.open(og_id_path).convert("RGB")
    except:
        return None, None, "Original ID not found"

    img1 = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2GRAY)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    score, diff = ssim(img1, img2, full=True)
    return score, diff, None



def generate_report(result, ela, blur, edge, similarity):
    return f"""
FRAUD DETECTION REPORT
----------------------------------------

Risk Summary
- Risk Score       : {result['risk_score']}
- Risk Level       : {result['risk_level']}
- Final Decision   : {result['decision']}
- Similarity Score : {round(similarity,3) if similarity is not None else "N/A"}

Analysis Breakdown
- ELA Score        : {round(ela,3)}
- Blur Score       : {round(blur,3)}
- Edge Score       : {round(edge,3)}

OCR Extracted Text
----------------------------------------
{result['ocr_text'] if result['ocr_text'] else "No text detected"}

"""


def analyze(image):
    if not validate_image(image):
        return None, "Invalid Image (Too Small)"

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    ela_score, ela_img = ela_check(image)
    blur_score = blur_check(img_bgr)
    edge_score, edge_img = edge_check(img_bgr)

    final_score = np.mean([ela_score, blur_score, edge_score])
    level = score_to_level(final_score)

    text = extract_text(image)

    similarity, diff_map, sim_error = compare_with_original(image)

    decision = (
        " Suspicious Document" if level == "HIGH"
        else (" Needs Review" if level == "MEDIUM"
        else " Looks Normal")
    )

    result = {
        "risk_score": round(float(final_score), 3),
        "risk_level": level,
        "decision": decision,
        "ocr_text": text
    }

    report = generate_report(result, ela_score, blur_score, edge_score, similarity)

    return {
        "result": result,
        "report": report,
        "ela": ela_img,
        "edges": edge_img,
        "similarity": similarity,
        "diff": diff_map,
        "sim_error": sim_error
    }, None



st.title(" ID Fraud Detection System")

uploaded = st.file_uploader("Upload Test ID", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded ID", use_container_width=True)

    output, error = analyze(image)

    if error:
        st.error(error)
    else:
        result = output["result"]

        st.subheader(" Result")
        st.metric("Risk Score", result["risk_score"])
        st.progress(result["risk_score"])

        if result["risk_level"] == "HIGH":
            st.error(result["decision"])
        elif result["risk_level"] == "MEDIUM":
            st.warning(result["decision"])
        else:
            st.success(result["decision"])

 
        st.subheader(" Similarity Check")

        if output["sim_error"]:
            st.warning(output["sim_error"])
        else:
            st.metric("Similarity Score", round(output["similarity"], 3))

            if output["similarity"] > 0.85:
                st.success(" Matches Original ID")
            elif output["similarity"] > 0.6:
                st.warning(" Partial Match")
            else:
                st.error(" Not Matching (Possible Fake)")

            diff_img = (output["diff"] * 255).astype("uint8")
            st.image(diff_img, caption="Difference Map")

        st.subheader(" Detailed Report")
        st.text(output["report"])


        st.subheader(" Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.image(output["ela"], caption="ELA")

        with col2:
            st.image(output["edges"], caption="Edges")