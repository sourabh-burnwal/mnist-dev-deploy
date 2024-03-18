import os
from time import time

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm


class Classification:
    def __init__(self):
        self.ort_session = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using: ", self.device)

        # Define a transform to normalize the data
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,)),
                                             ])

    def load_pt_model(self, model_path):
        self.model = torch.load(model_path, map_location=self.device)

    def load_onnx_model(self, model_path):
        self.model = onnx.load(model_path)
        ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=ep_list)

    @staticmethod
    def build_model():
        # Layer details for the neural network
        input_size = 784
        hidden_sizes = [128, 64]
        output_size = 10

        # Build a feed-forward network
        model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[1], output_size),
                              nn.LogSoftmax(dim=1))
        print(model)

        return model

    def pre_process(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        # print("Image size", image.shape)
        image_tensor = self.transform(image)
        # print(image_tensor.shape)
        image_tensor = image_tensor.view(1, 784)
        print(type(image_tensor), image_tensor.shape)
        return image_tensor

    def predict(self, image_path: str):
        image_tensor = self.pre_process(image_path)
        out = self.model(image_tensor.cuda())
        # print(out)
        ps = torch.exp(out)
        probab = list(ps.detach().cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        # print(pred_label)
        return pred_label

    def predict_onnx(self, image_path):
        image_tensor = self.pre_process(image_path)
        input_name = self.ort_session.get_inputs()[0].name
        # print("input_name", input_name)
        # print("output_name", output_name)
        res = self.ort_session.run(None, {input_name: image_tensor.cpu().numpy()})
        return np.argmax(res[0])

    def download_and_load_data(self, batch_size, download_dir, shuffle=True):
        os.makedirs(download_dir, exist_ok=True)

        # Download and load the training data
        trainset = datasets.MNIST(download_dir, download=True, train=True, transform=self.transform)
        valset = datasets.MNIST(download_dir, download=True, train=False, transform=self.transform)
        print("Data for model training downloaded")
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=shuffle)

        return trainloader, valloader

    def train(self, epochs: int = 15, batch_size: int = 32, model_path="my_mnist_model.pt"):
        model = self.build_model()
        model.to(self.device)

        trainloader, valloader = self.download_and_load_data(batch_size, "temp")

        # Training process
        print("Model training started:")
        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
        criterion = nn.NLLLoss()
        time0 = time()
        for e in tqdm(range(epochs)):
            running_loss = 0
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

                # Training pass
                optimizer.zero_grad()

                output = model(images.cuda())
                loss = criterion(output, labels.cuda())

                # This is where the model learns by backpropagation
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
        print("\nTraining Time (in minutes) =", (time() - time0) / 60)

        # Validation process
        print("Model validation stated")
        correct_count, all_count = 0, 0
        for images, labels in tqdm(valloader):
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                # Turn off gradients to speed up this part
                with torch.no_grad():
                    logps = model(img.cuda())

                # Output of the network are log-probabilities, need to take exponential for probabilities
                ps = torch.exp(logps)
                probab = list(ps.cpu().numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if true_label == pred_label:
                    correct_count += 1
                all_count += 1

        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count / all_count))

        torch.save(model, model_path)
        print(f"Model saved to {model_path} successfully")


if __name__ == '__main__':
    # Model Training
    # classification_obj = Classification()
    # model_path = "new_model.pt"
    # classification_obj.train(model_path=model_path)

    # Model Inference
    # model_path = "../models/new_model.pt"
    # classification_obj = Classification()
    # classification_obj.load_pt_model(model_path=model_path)
    # img_path = "/home/pavan/Mialo/my/sourabh_assigment/test_data/3.png"
    # classification_obj.predict(image_path=img_path)

    model_path = "../models/new_model.onnx"
    classification_obj = Classification()
    classification_obj.load_onnx_model(model_path=model_path)
    img_path = "/home/pavan/Mialo/my/sourabh_assigment/test_data/3.png"
    classification_obj.predict_onnx(image_path=img_path)
