import sys
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy, QRadioButton, QButtonGroup, QGroupBox, QSpinBox, QSlider, QFrame, QCheckBox, QPushButton
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import QMessageBox
import base64
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.style as mplstyle
import numpy as np

mplstyle.use("fast")
matplotlib.use("Qt5Agg")
plt.ioff()  # 禁用交互模式
config = {
    "font.family": "sans-serif",  # 使用无衬线体
    "font.serif": ["Microsoft YaHei"],
    "font.size": 14,  # 五号，10.5磅
    "axes.unicode_minus": False,
    "figure.constrained_layout.use": True,  # 自动调整布局,但是慢
}
plt.rcParams.update(config)


def length_of(vector):
    # return np.sqrt(sum([x**2 for x in vector]))

    # 检查是否是np数组
    # if not isinstance(vector, np.ndarray):
    #     vector = np.array(vector)
    # 用np矢量化
    return np.sqrt(np.sum(vector**2, axis=-1))





class ArrowArray(QLabel):
    def __init__(
        self,
        m=1,
        n=0,
        e_field=True,
        TM=True,
        omega=5e8,
    ):

        self.TM = TM
        self.is_e = e_field

        self.m = m
        self.n = n

        self.MIU = 4 * np.pi * 1e-7
        self.EPSILON = 8.854187817e-12

        self.omega = omega

        self.a = 4
        self.b = 2
        self.c = 10

        self.update_k()

        self.H_m = self.E_m = 1

        self.set_field_func()

        # 找到最长的向量
        # max_field = 0
        # for i in range(int(self.c / 0.5)):
        #     for j in range(int(self.a / 0.5)):
        #         for k in range(int(self.b / 0.5)):
        #             field = self.field_func([[i * 0.5, j * 0.5, k * 0.5]], 0)
        #             if length_of(field) > max_field:
        #                 max_field = length_of(field)
        self.update_Hm_Em()

    def update_k(self):
        self.delta_t = 2 * np.pi / self.omega
        self.k = self.omega * np.sqrt(self.MIU * self.EPSILON)
        self.kc = np.pi * np.sqrt(self.m**2 / self.a**2 + self.n**2 / self.b**2)
        # print(k, kc)

        if self.k < self.kc:
            raise ValueError("k < kc")

        self.beta = np.sqrt(self.k**2 - self.kc**2)  # 这是虚数部分
        self.gamma = 1j * self.beta

    def set_field_func(self):
        if self.TM:
            if self.is_e:
                self.field_func = self.field_func_TM_e
            else:
                self.field_func = self.field_func_TM_m
        else:
            if self.is_e:
                self.field_func = self.field_func_TE_e
            else:
                self.field_func = self.field_func_TE_m

    def update_Hm_Em(self):
        self.H_m = self.E_m = 1
        points = np.mgrid[0:self.c:0.5, 0:self.a:0.5, 0:self.b:0.5]
        points = points.reshape(3, -1).T
        fields = self.field_func(points, 0)
        max_field = np.max(length_of(fields))

        self.scale = 0.85

        self.H_m = self.E_m = 1 / max_field * self.scale

    """
    def field_func_TM_e(self, point, t):
        # c,a,b x,y,z
        x, y, z = point

        vector = np.array(
            [
                np.real(self.E_m * np.sin(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
                np.real(-self.gamma * self.m * np.pi / self.a / (self.kc**2) * self.E_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
                np.real(-self.gamma * self.n * np.pi / self.b / (self.kc**2) * self.E_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
            ]
        )

        # print(point)
        return vector

    def field_func_TM_m(self, point, t):
        # c,a,b x,y,z
        x, y, z = point

        vector = np.array(
            [
                0,
                np.real(1j * self.n * np.pi / self.b / (self.kc**2) * self.E_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
                np.real(-1j * self.m * np.pi / self.a / (self.kc**2) * self.E_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
            ]
        )

        # print(point)
        return vector

    def field_func_TE_e(self, point, t):
        # c,a,b x,y,z
        x, y, z = point

        vector = np.array(
            [
                0,
                np.real(1j * self.n * np.pi / self.b / (self.kc**2) * self.H_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
                np.real(-1j * self.m * np.pi / self.a / (self.kc**2) * self.H_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
                # 0,
                # np.sin(-0.5*x +t*10e8),
            ]
        )

        # print(point)
        return vector

    def field_func_TE_m(self, point, t):
        # c,a,b x,y,z
        x, y, z = point

        vector = np.array(
            [
                np.real(self.H_m * np.cos(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
                np.real(self.gamma * self.m * np.pi / self.a / (self.kc**2) * self.H_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
                np.real(self.gamma * self.n * np.pi / self.b / (self.kc**2) * self.H_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
            ]
        )

        # print(point)
        return vector
    """
    def field_func_TM_e(self, points, t):
        # c,a,b x,y,z
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        E_x = np.real(self.E_m * np.sin(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
        E_y = np.real(-self.gamma * self.m * np.pi / self.a / (self.kc**2) * self.E_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
        E_z = np.real(-self.gamma * self.n * np.pi / self.b / (self.kc**2) * self.E_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),

        return np.vstack([E_x, E_y, E_z]).T

    def field_func_TM_m(self, points, t):
        # c,a,b x,y,z
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        E_x = np.zeros_like(x),
        E_y = np.real(1j * self.n * np.pi / self.b / (self.kc**2) * self.E_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
        E_z = np.real(-1j * self.m * np.pi / self.a / (self.kc**2) * self.E_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),

        return np.vstack([E_x, E_y, E_z]).T

    def field_func_TE_e(self, points, t):
        # c,a,b x,y,z
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        E_x = np.zeros_like(x),
        E_y = np.real(1j * self.n * np.pi / self.b / (self.kc**2) * self.H_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
        E_z = np.real(-1j * self.m * np.pi / self.a / (self.kc**2) * self.H_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
        # 0,
        # np.sin(-0.5*x +t*10e8),

        return np.vstack([E_x, E_y, E_z]).T

    def field_func_TE_m(self, points, t):
        # c,a,b x,y,z
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        E_x = np.real(self.H_m * np.cos(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
        E_y = np.real(self.gamma * self.m * np.pi / self.a / (self.kc**2) * self.H_m * np.sin(self.m * np.pi * y / self.a) * np.cos(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),
        E_z = np.real(self.gamma * self.n * np.pi / self.b / (self.kc**2) * self.H_m * np.cos(self.m * np.pi * y / self.a) * np.sin(self.n * np.pi * z / self.b) * np.exp(1j * self.omega * t - self.gamma * x)),

        return np.vstack([E_x, E_y, E_z]).T

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.spines["bottom"].set_edgecolor("grey")
        self.ax.spines["left"].set_edgecolor("grey")
        self.ax.spines["top"].set_edgecolor("none")
        self.ax.spines["right"].set_edgecolor("none")
        self.ax.tick_params(axis="x", colors="grey", labelsize=12)
        self.ax.tick_params(axis="y", colors="grey", labelsize=12)
        self.ax.tick_params(axis="z", colors="grey", labelsize=12)
        self.ax.yaxis.get_offset_text().set(size=12)
        self.ax.xaxis.get_offset_text().set(size=12)
        self.ax.zaxis.get_offset_text().set(size=12)
        self.ax.tick_params(color="grey")
        self.ax.grid(False)
        # self.ax.grid(color='grey', alpha=0.4)
        # gird中间是镂空的
        self.ax.set_facecolor("none")

        # tansparent background
        self.ax.patch.set_alpha(0)
        self.fig.patch.set_alpha(0)
        # self.fig.tight_layout()
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.xaxis.set_major_formatter(formatter)

        super(MplCanvas, self).__init__(self.fig)



class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # self.setWindowTitle("上位机")
        # 定时刷新图片
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(10)  # 10ms刷新一次

        # self.update_thread = UpdateThread(self)
        # self.update_thread.start()

        self.init_ui()

    def add_label(self, text, wiget, text_size = None):
        layout = QHBoxLayout()
        label = QLabel(text)
        # wiget尽量大
        # label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if text_size:
            label.setFixedWidth(text_size)
        # label.setFixedWidth(50)
        layout.addWidget(label)
        layout.addWidget(wiget)
        layout.addStretch()
        return layout

    def update_quiver(self):
        self.data = self.arrow_array.field_func(self.arrow_start_points, self.t)
        data_length = length_of(self.data)

        # print(self.arrow_start_points.shape, self.data.shape, data_length.shape)

        mask = data_length > 0.01

        # 去除self.data中接近0的向量，并同步去除self.arrow_start_points中的点
        masked_arrow_start_points = self.arrow_start_points[mask]
        masked_data = self.data[mask]
        masked_data_length = data_length[mask]

        cmap = matplotlib.colormaps["coolwarm"]
        colors = cmap(masked_data_length)
        colors_repeat = np.repeat(colors, 2, axis=0)
        # 两个相连接
        colors = np.concatenate([colors, colors_repeat], axis=0)

        self.quiver = self.sc1.ax.quiver(
            masked_arrow_start_points[:, 0] - masked_data[:, 0]/2,
            masked_arrow_start_points[:, 1] - masked_data[:, 1]/2,
            masked_arrow_start_points[:, 2] - masked_data[:, 2]/2,
            masked_data[:, 0],
            masked_data[:, 1],
            masked_data[:, 2],
            color=colors,
        )

    def update_k(self):
        try :
            self.arrow_array.update_k()
        except:
            self.omega_spinbox.setValue(self.omega_spinbox.value() + 1)
            self.change_omega()
            

    def change_tetm(self):
        if self.option1_1.isChecked():
            self.arrow_array.TM = False
        else:
            self.arrow_array.TM = True
            if self.arrow_array.n == 0:
                self.n_spinbox.setValue(1)
                self.arrow_array.n = 1
            if self.arrow_array.m == 0:
                self.m_spinbox.setValue(1)
                self.arrow_array.m = 1
        
        self.arrow_array.set_field_func()
        self.arrow_array.update_Hm_Em()

    def change_field(self):
        if self.option2_1.isChecked():
            self.arrow_array.is_e = True
        else:
            self.arrow_array.is_e = False
        
        self.arrow_array.set_field_func()
        self.arrow_array.update_Hm_Em()

    def change_aspect(self):
        if self.all_checked.isChecked():
            self.arrow_start_points = np.mgrid[0:self.arrow_array.c+0.1:0.5, 0:self.arrow_array.a+0.1:0.5, 0:self.arrow_array.b+0.1:0.5].reshape(3, -1).T
        else:
            # 滑杆的比例
            scale = self.slider.value() / 100
            if self.option3_1.isChecked():
                self.arrow_start_points = np.mgrid[scale*self.arrow_array.c:scale*self.arrow_array.c+0.1:0.5, 0:self.arrow_array.a+0.1:0.5, 0:self.arrow_array.b+0.1:0.5].reshape(3, -1).T
            elif self.option3_2.isChecked():
                self.arrow_start_points = np.mgrid[0:self.arrow_array.c+0.1:0.5, scale*self.arrow_array.a:scale*self.arrow_array.a+0.1:0.5, 0:self.arrow_array.b+0.1:0.5].reshape(3, -1).T
            else:
                self.arrow_start_points = np.mgrid[0:self.arrow_array.c+0.1:0.5, 0:self.arrow_array.a+0.1:0.5, scale*self.arrow_array.b:scale*self.arrow_array.b+0.1:0.5].reshape(3, -1).T
            # self.sc1.draw()

    def change_m(self):
        if self.arrow_array.TM or (self.n_spinbox.value() == 0):
            if self.m_spinbox.value() == 0:
                self.m_spinbox.setValue(1)
                return

        self.arrow_array.m = self.m_spinbox.value()
        self.update_k()
        self.arrow_array.update_Hm_Em()

    def change_n(self):
        if self.arrow_array.TM or (self.m_spinbox.value() == 0):
            if self.n_spinbox.value() == 0:
                self.n_spinbox.setValue(1)
                return

        self.arrow_array.n = self.n_spinbox.value()
        self.update_k()
        self.arrow_array.update_Hm_Em()

    def change_abc(self):
        self.arrow_array.a = self.a_spinbox.value()
        self.arrow_array.b = self.b_spinbox.value()
        self.arrow_array.c = self.c_spinbox.value()
        self.update_k()
        self.change_aspect()
        self.arrow_array.update_Hm_Em()
        self.set_range()

    def change_speed(self):
        self.speed = self.speed_spinbox.value()

    def change_omega(self):
        self.arrow_array.omega = self.omega_spinbox.value() * 1e8
        self.arrow_array.delta_t = 2 * np.pi / self.arrow_array.omega
        self.update_k()
        self.arrow_array.update_Hm_Em()

    def all_checked_click(self):
        if self.all_checked.isChecked():
            # 将切片位置滑杆和切片方向单选按钮设为不可用
            self.slider.setEnabled(False)
            self.option3_1.setEnabled(False)
            self.option3_2.setEnabled(False)
            self.option3_3.setEnabled(False)

        else:
            self.slider.setEnabled(True)
            self.option3_1.setEnabled(True)
            self.option3_2.setEnabled(True)
            self.option3_3.setEnabled(True)

        self.change_aspect()

    def init_ui(self):
        # self.central_widget = QSplitter(Qt.Horizontal)
        # # 显示分界线并保留把手上的点
        # self.central_widget.setHandleWidth(3)
        # self.central_widget.setStyleSheet("QSplitter::handle{background-color: #40808080;}")

        self.central_widget = QWidget()
        self.central_widget_layout = QHBoxLayout()
        self.central_widget.setLayout(self.central_widget_layout)

        # self.central_widget_layout = QHBoxLayout()
        self.setCentralWidget(self.central_widget)
        # self.central_widget.setLayout(self.central_widget_layout)

        self.button_widget = QWidget()
        self.button_layout = QVBoxLayout()
        self.button_widget.setLayout(self.button_layout)
        # fixed size
        self.button_widget.setFixedWidth(240)
        # self.central_widget_layout.addLayout(self.button_layout)
        # self.central_widget.addWidget(self.button_widget)
        self.central_widget_layout.addWidget(self.button_widget)
        

        # self.button1 = QPushButton("按钮一1")
        # self.button1.setFixedWidth(150)
        # self.button1.clicked.connect(self.button1_click)
        # self.button_layout.addWidget(self.button1)

        # self.button2 = QPushButton("按钮二")
        # self.button2.setFixedWidth(150)
        # self.button2.clicked.connect(self.button2_click)
        # self.button_layout.addWidget(self.button2)

        # 单行输入框，有提示标签
        # self.input_layout = QHBoxLayout()
        # self.input_label = QLabel("输入框")
        # self.input_layout.addWidget(self.input_label)
        # self.input_label.setFixedWidth(50)
        # self.input_edit = QLineEdit()
        # self.input_layout.addWidget(self.input_edit)
        # self.input_edit.setFixedWidth(90)
        # # 靠左
        # self.input_layout.addStretch()
        # self.button_layout.addLayout(self.input_layout)



        # 二项选择框QRadioButton，选择TM或TE，e_field或m_field
        # 第一组单选按钮
        group1 = QGroupBox("模式")
        hbox1 = QHBoxLayout()
        
        self.option1_1 = QRadioButton("TE")
        self.option1_2 = QRadioButton("TM")
        
        # 创建按钮组并添加按钮
        self.button_group1 = QButtonGroup(self)
        self.button_group1.addButton(self.option1_1, 1)
        self.button_group1.addButton(self.option1_2, 2)
        
        # 默认选中第一个选项
        self.option1_1.setChecked(True)
        
        hbox1.addWidget(self.option1_1)
        hbox1.addWidget(self.option1_2)
        group1.setLayout(hbox1)
        
        # 第二组单选按钮
        group2 = QGroupBox("场")
        hbox2 = QHBoxLayout()
        
        self.option2_1 = QRadioButton("电场")
        self.option2_2 = QRadioButton("磁场")
        
        # 创建第二个按钮组
        self.button_group2 = QButtonGroup(self)
        self.button_group2.addButton(self.option2_1, 1)
        self.button_group2.addButton(self.option2_2, 2)
        
        # 默认选中第一个选项
        self.option2_1.setChecked(True)
        
        hbox2.addWidget(self.option2_1)
        hbox2.addWidget(self.option2_2)
        group2.setLayout(hbox2)


        # 第三组单选按钮
        group3 = QGroupBox("切片方向")
        hbox3 = QHBoxLayout()
        
        self.option3_1 = QRadioButton("x")
        self.option3_2 = QRadioButton("y")
        self.option3_3 = QRadioButton("z")

        self.button_group3 = QButtonGroup(self)
        self.button_group3.addButton(self.option3_1, 1)
        self.button_group3.addButton(self.option3_2, 2)
        self.button_group3.addButton(self.option3_3, 3)
        
        # 默认选中
        self.option3_3.setChecked(True)
        
        hbox3.addWidget(self.option3_1)
        hbox3.addWidget(self.option3_2)
        hbox3.addWidget(self.option3_3)
        group3.setLayout(hbox3)

        # 连接信号
        self.button_group1.buttonClicked.connect(self.change_tetm)
        self.button_group2.buttonClicked.connect(self.change_field)
        self.button_group3.buttonClicked.connect(self.change_aspect)
        
        # 添加所有组件到主布局
        self.button_layout.addWidget(group1)
        self.button_layout.addWidget(group2)
        self.button_layout.addWidget(group3)

        # 滑杆用于调节位置
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setSingleStep(1)
        self.slider.setFixedWidth(150)
        self.slider.valueChanged.connect(self.change_aspect)
        self.button_layout.addLayout(self.add_label("切片位置", self.slider, 70))
        
                
        # 显示所有按钮打勾
        self.all_checked = QCheckBox("显示全部")
        self.all_checked.stateChanged.connect(self.all_checked_click)
        self.button_layout.addWidget(self.all_checked)


        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(0, 3)
        self.speed_spinbox.setValue(2)
        self.speed_spinbox.setSingleStep(1)
        self.speed_spinbox.setFixedWidth(40)
        self.speed_spinbox.valueChanged.connect(self.change_speed)
        self.button_layout.addLayout(self.add_label("播放速度", self.speed_spinbox, 70))

        # 分割线
        split_line = QFrame(frameShape=QFrame.HLine)
        # 上下间距
        split_line.setFixedHeight(20)
        # 颜色
        split_line.setStyleSheet("color: #40808080;")
        self.button_layout.addWidget(split_line)

        # QSpinBox设置m和n与speed
        self.mn_layout = QHBoxLayout()  # 一行两个
        self.m_spinbox = QSpinBox()
        self.m_spinbox.setRange(0, 2)
        self.m_spinbox.setValue(1)
        self.m_spinbox.setSingleStep(1)
        self.m_spinbox.setFixedWidth(40)
        self.m_spinbox.valueChanged.connect(self.change_m)
        self.mn_layout.addLayout(self.add_label("m", self.m_spinbox, 15))

        self.mn_layout.addStretch()

        self.n_spinbox = QSpinBox()
        self.n_spinbox.setRange(0, 2)
        self.n_spinbox.setValue(0)
        self.n_spinbox.setSingleStep(1)
        self.n_spinbox.setFixedWidth(40)
        self.n_spinbox.valueChanged.connect(self.change_n)
        self.mn_layout.addLayout(self.add_label("n", self.n_spinbox, 15))

        self.mn_layout.addStretch()

        # 指定宽度的占位widget
        spacer = QWidget()
        spacer.setFixedWidth(60)
        self.mn_layout.addWidget(spacer)

        self.button_layout.addLayout(self.mn_layout)
    

        self.abc_layout = QHBoxLayout()  # 一行三个
        self.a_spinbox = QSpinBox()
        self.a_spinbox.setRange(2, 10)
        self.a_spinbox.setValue(4)
        self.a_spinbox.setSingleStep(1)
        self.a_spinbox.setFixedWidth(40)
        self.a_spinbox.valueChanged.connect(self.change_abc)
        self.abc_layout.addLayout(self.add_label("a", self.a_spinbox, 15))

        self.abc_layout.addStretch()

        self.b_spinbox = QSpinBox()
        self.b_spinbox.setRange(1, 5)
        self.b_spinbox.setValue(2)
        self.b_spinbox.setSingleStep(1)
        self.b_spinbox.setFixedWidth(40)
        self.b_spinbox.valueChanged.connect(self.change_abc)
        self.abc_layout.addLayout(self.add_label("b", self.b_spinbox, 15))

        self.abc_layout.addStretch()

        self.c_spinbox = QSpinBox()
        self.c_spinbox.setRange(8, 16)
        self.c_spinbox.setValue(10)
        self.c_spinbox.setSingleStep(1)
        self.c_spinbox.setFixedWidth(40)
        self.c_spinbox.valueChanged.connect(self.change_abc)
        self.abc_layout.addLayout(self.add_label("c", self.c_spinbox, 15))

        self.button_layout.addLayout(self.abc_layout)


        # 单行输入框，有提示标签
        self.omega_spinbox = QSpinBox()
        self.omega_spinbox.setRange(4, 15)
        self.omega_spinbox.setValue(5)
        self.omega_spinbox.setSingleStep(1)
        self.omega_spinbox.setFixedWidth(40)
        self.omega_spinbox.valueChanged.connect(self.change_omega)
        self.button_layout.addLayout(self.add_label("频率", self.omega_spinbox, 40))


        self.button_layout.addStretch()


        self.info_button = QPushButton("关于")
        # self.info_button.setFixedWidth(150)
        self.info_button.clicked.connect(self.info_button_click)
        self.button_layout.addWidget(self.info_button)



        # 分割竖线
        split_line = QFrame(frameShape=QFrame.VLine)
        # 左右间距
        split_line.setFixedWidth(5)
        # 颜色
        split_line.setStyleSheet("color: #40808080;")
        self.central_widget_layout.addWidget(split_line)





        # 新建一个三维的图
        self.sc1 = MplCanvas(self, width=3, height=4, dpi=100)
        self.sc1.setStyleSheet("""background-color: rgba(255, 255, 255, 0);""")
        self.sc1.setMinimumSize(500, 350)
        self.sc1.fig.set_size_inches(4, 2)
        # 尽量大的显示sc这个画布
        self.sc1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.central_widget_layout.addWidget(self.sc1)
        # self.central_widget.addWidget(self.sc1)

        self.arrow_start_points = []
        self.arrow_array = ArrowArray(m=1, n=0, e_field=True, TM=False, omega=5e8)

        
        self.set_range()

        self.t = 0
        self.speed = 2
        
        self.change_aspect()

        self.update_quiver()

        self.setWindowTitle("矩形波导场分布")


    def info_button_click(self):
        # QMessageBox.about(self, "关于", "本程序由微电子本科22级吴以恒制作，使用Python语言，基于PySide6和matplotlib库，使用了numpy库进行数值计算。\n"
        # "本程序的源代码已上传至GitHub，欢迎查看和二次开发："
        # "<a href='https://zhuanlan.zhihu.com/p/610696872'>https://zhuanlan.zhihu.com/p/610696872</a>\n"
        # "喜欢本程序的话，请给我一个star，谢谢！")

        msg_box = QMessageBox()
        msg_box.setWindowTitle("关于")
        msg_box.setTextFormat(Qt.TextFormat.RichText)  # 启用富文本格式
        msg_box.setText("本程序由微电子本科22级吴以恒制作，使用Python语言，基于PySide6和matplotlib库，并使用umpy库进行数值计算。<br><br>"
        "本程序的源代码已上传至GitHub，欢迎查看和二次开发："
        "<a href='https://github.com/wxhenry/Rectangular-waveguide-animation-demonstration'>https://github.com/wxhenry/Rectangular-waveguide-animation-demonstration</a><br><br>"
        "喜欢本程序的话，请给我一个star，谢谢！")
        msg_box.exec()



    def set_range(self):
        # 设置轴的范围
        self.sc1.ax.set_xlim(0, self.arrow_array.c)
        self.sc1.ax.set_ylim(0, self.arrow_array.a)
        self.sc1.ax.set_zlim(0, self.arrow_array.b)
        # 设置轴的ticks
        self.sc1.ax.set_xticks(np.arange(0, self.arrow_array.c + 1, 1))
        self.sc1.ax.set_yticks(np.arange(0, self.arrow_array.a + 1, 1))
        self.sc1.ax.set_zticks(np.arange(0, self.arrow_array.b + 1, 1))

        self.sc1.ax.set_box_aspect([self.arrow_array.c, self.arrow_array.a, self.arrow_array.b])


    def update_image(self):
        start_time = time.time()

        self.sc1.ax.relim()
        self.sc1.ax.autoscale_view()
        self.sc1.fig.canvas.draw()

        self.quiver.remove()
        self.update_quiver()

        print("\r更新图片耗时：", time.time() - start_time, end="")

        self.t += self.speed /10**10
        # time.sleep(0.04)
icon_data = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAElUlEQVR4nO2ZPW8cRRjH/8/s7r34Xuw4iVEixxbCCmWEjIRFikCDKIjFSwQfgIKKInyDfICkQ4hQRIhIdCCEhIyQkFIjRW4pEG4cUlhO7ODc+XbmeSjOc1nv3sveevf2QvZX3s7O7P83M8/O3QEFBQUFBQUFLyqU9wMMYvnKncqZU7MfiPDu/Z+u/ZrVOFMnYPnKncrp+eZHBHWdlLMqrCGCDTb61ubPH/+W9nhTI+BYcFKrAoGwBkBQjgdmDQg2mPStzR/TE5G7gMHBw2QjIjcBweAgtYqBwcOkK2LiAqIzDgj7CXpKR8TEBKy8+0u5WW1dezbjSYOHCYkYs1hmLiD5Uh+XZCIyE5DeUh+X8bZG6gLiV/WsiSciNQHRpT6pGR9FUARvMOubm6998jtuEHevnpDpDR5EABDc8ixAardK6tV737+9AwBu0i77F7dpDA4opwKnVIfj1cCsxRzq3sSPLSC/4jYOx4Mrt4LAYpdgy9gC+hU3nsrg1Ce4IJS7x0gB/fb4/yG4ZaCA56e49Vvqo4NbIgLW1m5WzeLFD5nNdYCei+KWJLglIuAf/fdb1R1zt9q8AOVUIOAUHjgtki/1QUQEsPZLTx//hdb+Fsq186g2l+CWGl0RkmyQk5N+cEu0BhAJQUFYo7W/hcN/H6Bcz0tEMHgDyi0jreCWIW8BApEDkZCI2SW4XtYisg9uiXEOCIk4eNDdGpmImFxwS8yDUPcUReRktDUmH9wyUoDRLRC5UI6HYyLCWyORiPyCW0YKENOB4QMopwTlVqFUGiLyD26JVQMAgE0HbHwox4snYnYZrlcPiZie4JYxvg1GRTjuDEh5vQyRYlk/j2pzGa5XgwjnHpzIAWAqwc8S/B5gRfhg/RgChuvV4Hg1gOiZCNZo7W3h8Mk2qnOvYPbc61BOfsHJ8SDsbwPy5S7m9+y1ZAKEYUwbrFsQNvDbu3C8GkqV+YgIZh+d1s5ReCCv4EYf3jaKbv9x982HwTbjbQFhGNMC6zZEuv/bgborwnQO0PIPjos4uo9Iwe7/SdALrgPBv1t72K9tPAEiMObp8eDhMEERnQNUmhfgluoTrW19g3/bP7hlhACCiEan/QgQA5DCyFkk6lb9CX5xShLcMkKAwC3PAeTAbz8C6/ZR/tz/VAZwsuCWiAAhcYgciGEAAiIFr3IKXrkJ//AJ/PZu7iLSCG6JCHDFuc/s/wBgnZTrChtAGADBq8zBKzdyE5Fm8F6fgy4sXfrsMoi+ALBOyumKsBWNFCBmsAgRVBqLcMsNCBuUZs5iYeW9YcMNf8hAcAYfBX/jRMF7fY9qEEeEPtxHJ1gjBKkIyDJ4b4y4DccRYXQL1cYi3HLzSMAZWVi5SnGHCwYXyNda4Zu0g/fGGveGeFtjr3vu96pjCZjEjEfGTHrjcBHUOweI8EgBk5zxyNgn7WCoCAwXkMeMh0nt/TVIRD8B0xDckvoLPCyCjY/SzGlZWLlKRO7UBLdkdoKxIkTMerl+zn3p4vsQ1tvM0xHckvkRbunSp5dL9cXPz778zp8lx/3q3pQELygoKCgoKCgoeNH5D1NualhX4mSLAAAAAElFTkSuQmCC"

if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle("fusion")
    pixmap = QPixmap()
    pixmap.loadFromData(base64.b64decode(icon_data))    
    app.setWindowIcon(QIcon(pixmap))
    font = QFont()
    font.setPointSize(12)
    font.setFamily("Microsoft YaHei")
    app.setFont(font)

    window = MainWindow()
    # window.setWindowIcon(QIcon(icon_path))
    window.show()
    sys.exit(app.exec())
