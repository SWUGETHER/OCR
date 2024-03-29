from fastapi import FastAPI, UploadFile
import os
import ocr
import recommendation as rec
import uuid

app = FastAPI()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/extract_data")
async def extract_data(file: UploadFile):
    filename = f"{str(uuid.uuid4())}.jpg"
    UPLOAD_DIR = "./temp_data/" + filename

    with open(os.path.join(UPLOAD_DIR), "wb") as fp:
        fp.write(file.file.read())
    
    # OCR
    name, comp = ocr.read_image(UPLOAD_DIR)

    # Content-Based Filtering Recommendation
    result = rec.recommend(name, comp)

    # 이미지 파일 삭제
    os.remove(os.path.join(UPLOAD_DIR))

    return result