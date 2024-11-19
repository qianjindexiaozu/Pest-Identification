import os
import json
import sys

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFileDialog, QLabel, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from model import resnet34


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("农业病害虫识别")
        self.resize(800, 600)

        # 创建一个垂直布局
        self.layout = QVBoxLayout(self)

        # 创建一个按钮，点击后选择图片
        self.select_button = QPushButton("选择图片", self)
        self.select_button.clicked.connect(self.select_image)
        self.layout.addWidget(self.select_button)

        # 创建一个用于显示图片的QLabel
        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        # 创建Matplotlib的画布
        self.canvas = FigureCanvas(plt.figure())
        self.layout.addWidget(self.canvas)

        # 初始化必要的变量
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # 加载类别索引
        json_path = './class_indices.json'
        assert os.path.exists(json_path), f"文件 '{json_path}' 不存在。"
        with open(json_path, "r") as f:
            self.class_indict = json.load(f)

        # 加载模型
        self.model = resnet34(num_classes=8).to(self.device)
        weights_path = "./resNet34-after.pth"
        assert os.path.exists(weights_path), f"文件 '{weights_path}' 不存在。"
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))

    def select_image(self):
        # 每次选择新图片之前，清空画布
        self.clear_canvas()

        # 打开文件选择对话框，选择图片
        img_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)")

        # 如果用户取消选择或没有选择文件
        if not img_path:
            print("未选择图片，程序退出。")
            return

        print(f"选择的图片路径: {img_path}")

        # 确保文件存在
        assert os.path.exists(img_path), f"文件 '{img_path}' 不存在。"

        # 加载图片
        img = Image.open(img_path)

        # 在Matplotlib画布中显示图片
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴
        self.canvas.draw()

        # 图片预处理
        img = self.data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        # 进行预测
        self.model.eval()
        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   probability: {:.3f}".format(self.class_indict[str(predict_cla)],
                                                               predict[predict_cla].numpy())

        # 更新标题和Matplotlib画布中的文本
        ax.set_title(print_res, fontsize=12)
        self.canvas.draw()

        # 打印其他预测结果
        for i in range(len(predict)):
            print("class: {:10}   probability: {:.3f}".format(self.class_indict[str(i)],
                                                               predict[i].numpy()))

    def clear_canvas(self):
        # 清空画布
        self.canvas.figure.clf()
        # 重新添加一个新的子图
        self.canvas.figure.add_subplot(111)
        self.canvas.draw()


def main():
    # 将渲染模式换成xcb
    os.environ["XDG_SESSION_TYPE"] = "xcb"

    app = QApplication(sys.argv)

    # 创建窗口
    window = MyWindow()

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
