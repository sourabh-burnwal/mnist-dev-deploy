import cv2
import numpy as np
import tritonclient.grpc as grpcclient

MODEL_METADATA = {
    "model_name": "mnist",
    "input_tensor": {
        "name": "onnx::Gemm_0",
        "data_type": "FP32",
        "shape": (1, 784)
    },
    "output_tensor": {
        "name": "12"
    }
}

def preprocess(img_path: str) -> np.array:
    """Reads, and resizes the input image"""
    raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    raw_img = raw_img.astype(np.float32)
    img = cv2.resize(raw_img, (28, 28))
    img = img.reshape(MODEL_METADATA["input_tensor"]['shape'])
    return img


def prepare_input(data: np.array) -> grpcclient.InferInput:
    """Prepares input tensor for inference request"""
    inputs = []
    inputs.append(
        grpcclient.InferInput(
            MODEL_METADATA["input_tensor"]['name'],
            data.shape,
            MODEL_METADATA["input_tensor"]['data_type'],
        )
    )
    inputs[0].set_data_from_numpy(data)
    return inputs


def infer(client: grpcclient.InferenceServerClient, input_data: grpcclient.InferInput) -> np.array:
    """Sends inference request to Triton.""" 
    outputs = [grpcclient.InferRequestedOutput(MODEL_METADATA["output_tensor"]['name'])]
    response = client.infer(
        model_name=MODEL_METADATA["model_name"],
        inputs=input_data,
        outputs=outputs
    )
    response_np = response.as_numpy(MODEL_METADATA["output_tensor"]['name'])
    return response_np


def postprocess(raw_inference: np.array) -> int:
    """Postprocess the raw inference output to desired format."""
    raw_inference = list(raw_inference[0])
    label = raw_inference.index(max(raw_inference))
    return label


def do_inference(img_path: str) -> int:
    """Executes required methods for inference"""
    input_data = preprocess(img_path)
    prepared_input = prepare_input(input_data)
    client = grpcclient.InferenceServerClient("localhost:8001")
    raw_output = infer(client, prepared_input)
    output = postprocess(raw_output)
    return output


if __name__ == '__main__':
    output = do_inference("FastAPI/test_data/3.png")
    print(output)