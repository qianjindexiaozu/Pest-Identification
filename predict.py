import os
import json
import sys

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget

from model import resnet34


def main():
    # 将渲染模式换成xcb
    os.environ["XDG_SESSION_TYPE"] = "xcb"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./images.jpeg"
    # 通过pyqt5可视化操作页面
    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(300, 300)
    w.move(300, 300)
    w.setWindowTitle("农业病害虫识别")
    w.show()

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet34(num_classes=8).to(device)

    # load model weights
    weights_path = "./resNet34-after.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()