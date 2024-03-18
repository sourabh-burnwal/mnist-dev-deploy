import torch
from torch.autograd import Variable


def pt_to_onnx(pt_model_path, onnx_model_path):
    pt_model = torch.load(pt_model_path)
    dummy_input = Variable(torch.randn(1, 784))
    print(type(dummy_input), dummy_input.shape)
    torch.onnx.export(pt_model, dummy_input.cuda(), onnx_model_path)
    print("Model converted into ONNX format")


if __name__ == '__main__':
    pt_path = "models/new_model.pt"
    onnx_path = "models/new_model.onnx"
    pt_to_onnx(pt_path, onnx_path)
