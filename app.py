from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io

from helper.helper_functions import *

app = FastAPI()
@app.get("/")
def root():
    return "Hello World"

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file)

    extract_output = extract_average_colors(image)

    colored_mask = extract_output[0]

    img_io = io.BytesIO()
    colored_mask.save(img_io, 'JPEG')
    img_io.seek(0)

    # save image and send back to streamlit

    # return StreamingResponse(img_io, media_type="image/jpeg")

    return StreamingResponse(img_io, media_type="image/jpeg")

@app.post("/predict")
async def predict_colour(file: UploadFile = File(...)):
    image = Image.open(file.file)

    extract_output = extract_average_colors(image)

    skin = extract_output[1]
    average_skin_color = extract_output[2]
    average_brows_color = extract_output[3]
    average_hair_color = extract_output[4]
    average_lip_color = extract_output[5]
    average_eye_color = extract_output[6]

    skin_tone_classification = predict_skin_tone_classification(average_skin_color)

    predicted_season = predict_season(average_eye_color, average_hair_color, average_lip_color, average_brows_color)


    # save image and send back to streamlit

    # return StreamingResponse(img_io, media_type="image/jpeg")

    return JSONResponse(content={"skin_tone_classification": skin_tone_classification
                                 ,"predicted_season": predicted_season
                                 ,"skin":skin
                                 ,"average_skin_color":average_skin_color
                                 ,"average_brows_color":average_brows_color
                                 ,"average_hair_color":average_hair_color
                                 ,"average_lip_color":average_lip_color
                                 ,"average_eye_color":average_eye_color})

    # img_io = io.BytesIO() # Create an in-memory bytes buffer.
    # flipped_image.save(img_io, 'PNG') # Save the flipped image to the buffer in JPEG format.
    # img_io.seek(0)# Move the file pointer to the beginning of the buffer.
    # return StreamingResponse(img_io, media_type="image/png") #officially type for JPEG images


# @app.post("/flip-image")
# async def flip_image(file: UploadFile = File(...)):
#     image = Image.open(file.file)
#     flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
#     img_io = io.BytesIO() # Create an in-memory bytes buffer.
#     flipped_image.save(img_io, 'PNG') # Save the flipped image to the buffer in JPEG format.
#     img_io.seek(0)# Move the file pointer to the beginning of the buffer.
#     return StreamingResponse(img_io, media_type="image/png") #officially type for JPEG images
