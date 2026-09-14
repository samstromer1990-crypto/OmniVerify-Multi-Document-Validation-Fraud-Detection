import cv2
import numpy as np
import os

def generate_doc(output_path, doc_type="aadhaar"):
    # White background
    width, height = 800, 500
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)

    if doc_type == "aadhaar":
        cv2.putText(image, "AADHAAR", (280, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(image, "Name: John Doe", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "Number: 1234 5678 9012", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "DOB: 01/01/1990", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "Gender: M", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(image, (width - 170, height - 170), (width - 30, height - 30), (0, 0, 0), -1)

    elif doc_type == "pan":
        cv2.putText(image, "INCOME TAX DEPARTMENT - PAN", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "Name: Jane Smith", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "Father's Name: Robert Smith", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "PAN: ABCDE1234F", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "DOB: 05/05/1985", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    elif doc_type == "license":
        cv2.putText(image, "DRIVING LICENCE", (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, "Name: Alan Turing", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "DL No: 1234567890123456", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "DOB: 23/06/1912", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "Validity: 2050", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    generate_doc("data/samples/sample_aadhaar.jpg", "aadhaar")
    generate_doc("data/samples/sample_pan.jpg", "pan")
    generate_doc("data/samples/sample_license.jpg", "license")
    print("All three synthetic documents generated in data/samples/")
