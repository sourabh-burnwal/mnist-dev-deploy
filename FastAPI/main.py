import os

import aiofiles
import uvicorn
from fastapi import FastAPI, File, UploadFile

from config import project_config
from core.model import Classification
from utils import logger
from utils.response import api_response

app = FastAPI(title=project_config["PROJECT_NAME"])

api_config = project_config["API"]
os.makedirs(api_config["static_dir"], exist_ok=True)

classification_model = Classification()
classification_model.load_pt_model(model_path=project_config["MODEL"]['pt_model_path'])


@app.post("/getClassificationResults")
async def get_mnist_classification(
        image: UploadFile = File(...)
):
    try:
        logger.info("API: /getClassificationResults started")
        # Saving the image into local directory
        image_path = f"{api_config['static_dir']}/{image.filename}"
        async with aiofiles.open(image_path, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)
        logger.info(f"Saved image into local directory {image_path}")

        # Call the model inference
        prediction = classification_model.predict(image_path=image_path)

        # Remove the files which has been saved
        os.remove(image_path)
        logger.info(f"API: /getClassificationResults has been completed")
        return api_response(200, "Success", prediction)
    except Exception as ex:
        print("Exception occurred: ", ex)
        logger.error(f"Exception occurred at getClassificationResults api: {ex}")
        return api_response(500, "Error in API", str(ex))


if __name__ == '__main__':
    logger.info("API Services started")
    uvicorn.run("main:app", host=api_config['host'], port=api_config['port'], reload=api_config['reload'])
