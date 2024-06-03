from typing import Annotated, Union
from server import SignTranslationServer
from fastapi import FastAPI, Response, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

import onnxruntime as ort
import logging

from steps.extract_frames import read_video
from steps.extract_keypoints import extract_keypoints
from steps.preprocess import preprocess
from steps.inference import inference
from steps.get_gloss import translate_result
from steps.center_crop import center_crop

logger = logging.getLogger(__name__)

app = FastAPI()
# app = SignTranslationServer()


@app.post("/predict")
async def predict(video: Annotated[UploadFile, None] = None):
    if not video:
        return Response(status_code=400, content="No file provided")

    # Save video to a file for debugging
    with open("video.mp4", "wb") as f:
        f.write(await video.read())
        video.seek(0)
        
    vid = await read_video(video)
    vid_resized = center_crop(vid)
    kps = extract_keypoints(vid_resized)
    sgn_video, sgn_video_low, sgn_heatmap, sgn_heatmap_low = preprocess(vid, kps)
    result = inference(sgn_video, sgn_video_low, sgn_heatmap, sgn_heatmap_low)
    translated_result = translate_result(result)

    response = dict(
        result=translated_result[0],
        other_options=translated_result[1:]
    )
    
    return JSONResponse(status_code=200, content=response)

if __name__ == "__main__":
    # app.run()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
