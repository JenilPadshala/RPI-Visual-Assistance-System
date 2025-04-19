import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import os

app = FastAPI()


class ImageData(BaseModel):
    image_base64: str


@app.post("/upload-image/")
async def upload_image(data: ImageData):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data.image_base64)

        # Create a unique filename
        filename = f"text.jpg"
        filepath = os.path.join("received_images", filename)

        # Ensure the output directory exists
        os.makedirs("received_images", exist_ok=True)

        # Save the image
        with open(filepath, "wb") as f:
            f.write(image_data)

        # run text recognition script
        result = subprocess.run(
            ["python3", "text_recognition.py"], capture_output=True, text=True
        )

        # extract the recognized text
        text = result.stdout.strip().strip('"')
        print(text)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
