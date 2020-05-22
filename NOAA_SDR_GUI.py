# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PythonSDR_GUI_design.ui'
#
# Created: Mon Jun 17 20:43:02 2019
#      by: PyQt5 UI code generator 5.3.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(964, 686)
        MainWindow.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout.setObjectName("gridLayout")
        self.widget = QtWidgets.QWidget(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.splitter_h = QtWidgets.QSplitter(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.splitter_h.sizePolicy().hasHeightForWidth())
        self.splitter_h.setSizePolicy(sizePolicy)
        self.splitter_h.setToolTip("")
        self.splitter_h.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_h.setObjectName("splitter_h")
        self.splitter_v = QtWidgets.QSplitter(self.splitter_h)
        self.splitter_v.setToolTip("")
        self.splitter_v.setOrientation(QtCore.Qt.Vertical)
        self.splitter_v.setObjectName("splitter_v")
        self.fft_disp = QtWidgets.QWidget(self.splitter_v)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.fft_disp.sizePolicy().hasHeightForWidth())
        self.fft_disp.setSizePolicy(sizePolicy)
        self.fft_disp.setMinimumSize(QtCore.QSize(0, 64))
        self.fft_disp.setToolTip("")
        self.fft_disp.setObjectName("fft_disp")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.fft_disp)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.fft_disp_layout = QtWidgets.QHBoxLayout()
        self.fft_disp_layout.setContentsMargins(0, 0, 0, 0)
        self.fft_disp_layout.setObjectName("fft_disp_layout")
        self.gridLayout_6.addLayout(self.fft_disp_layout, 0, 0, 1, 1)
        self.waterfall_widget = QtWidgets.QWidget(self.splitter_v)
        self.waterfall_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.waterfall_widget.setObjectName("waterfall_widget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.waterfall_widget)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.waterfall_layout = QtWidgets.QHBoxLayout()
        self.waterfall_layout.setObjectName("waterfall_layout")
        self.gridLayout_4.addLayout(self.waterfall_layout, 0, 0, 1, 1)
        self.controls_widget = QtWidgets.QWidget(self.splitter_h)
        self.controls_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.controls_widget.setObjectName("controls_widget")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.controls_widget)
        self.gridLayout_9.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_15 = QtWidgets.QLabel(self.controls_widget)
        self.label_15.setTextFormat(QtCore.Qt.AutoText)
        self.label_15.setObjectName("label_15")
        self.gridLayout_9.addWidget(self.label_15, 2, 0, 1, 1)
        self.trans_slider = QtWidgets.QSlider(self.controls_widget)
        self.trans_slider.setMinimum(10)
        self.trans_slider.setMaximum(40)
        self.trans_slider.setSingleStep(300)
        self.trans_slider.setSliderPosition(20)
        self.trans_slider.setOrientation(QtCore.Qt.Horizontal)
        self.trans_slider.setObjectName("trans_slider")
        self.gridLayout_9.addWidget(self.trans_slider, 8, 2, 1, 1)
        self.satelite_combo = QtWidgets.QComboBox(self.controls_widget)
        self.satelite_combo.setObjectName("satelite_combo")
        self.gridLayout_9.addWidget(self.satelite_combo, 1, 2, 1, 1)
        self.bandwidth_combo = QtWidgets.QComboBox(self.controls_widget)
        self.bandwidth_combo.setObjectName("bandwidth_combo")
        self.gridLayout_9.addWidget(self.bandwidth_combo, 3, 2, 1, 1)
        self.line = QtWidgets.QFrame(self.controls_widget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_9.addWidget(self.line, 4, 0, 1, 3)
        self.label_2 = QtWidgets.QLabel(self.controls_widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout_9.addWidget(self.label_2, 8, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.controls_widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_9.addWidget(self.label_3, 10, 0, 1, 1)
        self.label_sat = QtWidgets.QLabel(self.controls_widget)
        self.label_sat.setObjectName("label_sat")
        self.gridLayout_9.addWidget(self.label_sat, 1, 0, 1, 1)
        self.sample_rate_combo = QtWidgets.QComboBox(self.controls_widget)
        self.sample_rate_combo.setObjectName("sample_rate_combo")
        self.gridLayout_9.addWidget(self.sample_rate_combo, 2, 2, 1, 1)
        self.cutoff_slider = QtWidgets.QSlider(self.controls_widget)
        self.cutoff_slider.setMinimum(20)
        self.cutoff_slider.setMaximum(60)
        self.cutoff_slider.setSingleStep(800)
        self.cutoff_slider.setProperty("value", 40)
        self.cutoff_slider.setSliderPosition(40)
        self.cutoff_slider.setOrientation(QtCore.Qt.Horizontal)
        self.cutoff_slider.setObjectName("cutoff_slider")
        self.gridLayout_9.addWidget(self.cutoff_slider, 7, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.controls_widget)
        self.label.setObjectName("label")
        self.gridLayout_9.addWidget(self.label, 7, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.controls_widget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_9.addWidget(self.line_2, 9, 0, 1, 3)
        self.run_stop_button = QtWidgets.QPushButton(self.controls_widget)
        self.run_stop_button.setCheckable(True)
        self.run_stop_button.setObjectName("run_stop_button")
        self.gridLayout_9.addWidget(self.run_stop_button, 16, 0, 1, 3)
        self.label_4 = QtWidgets.QLabel(self.controls_widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_9.addWidget(self.label_4, 11, 0, 1, 1)
        self.bandwidth_label = QtWidgets.QLabel(self.controls_widget)
        self.bandwidth_label.setObjectName("bandwidth_label")
        self.gridLayout_9.addWidget(self.bandwidth_label, 3, 0, 1, 1)
        self.lcdFreq = QtWidgets.QLCDNumber(self.controls_widget)
        self.lcdFreq.setDigitCount(6)
        self.lcdFreq.setSegmentStyle(QtWidgets.QLCDNumber.Filled)
        self.lcdFreq.setProperty("intValue", 0)
        self.lcdFreq.setObjectName("lcdFreq")
        self.gridLayout_9.addWidget(self.lcdFreq, 15, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem, 14, 0, 1, 3)
        self.imageLabel = QtWidgets.QLabel(self.controls_widget)
        self.imageLabel.setText("")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setObjectName("imageLabel")
        self.gridLayout_9.addWidget(self.imageLabel, 13, 0, 1, 3)
        self.horizontalSlider = QtWidgets.QSlider(self.controls_widget)
        self.horizontalSlider.setMinimum(1)
        self.horizontalSlider.setSliderPosition(20)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout_9.addWidget(self.horizontalSlider, 11, 1, 1, 2)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.controls_widget)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(100)
        self.horizontalSlider_2.setSliderPosition(100)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.gridLayout_9.addWidget(self.horizontalSlider_2, 10, 1, 1, 2)
        self.gridLayout_3.addWidget(self.splitter_h, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_10.setObjectName("gridLayout_10")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_10.addItem(spacerItem1, 0, 1, 1, 1)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab_2)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_vivo = QtWidgets.QScrollArea()
        self.tab_vivo.setWidgetResizable(True)
        self.tab_vivo.setObjectName("tab_vivo")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_vivo)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.image_sat = QtWidgets.QLabel(self.tab_vivo)
        self.image_sat.setText("")
        self.image_sat.setAlignment(QtCore.Qt.AlignTop)
        self.image_sat.setObjectName("image_sat")
        self.gridLayout_7.addWidget(self.image_sat, 0, 0, 1, 1)
        self.tab_vivo.setWidget(self.image_sat)
        self.tabWidget_2.addTab(self.tab_vivo, "")
        self.tab_sinc = QtWidgets.QScrollArea()
        self.tab_sinc.setWidgetResizable(True)
        self.tab_sinc.setObjectName("tab_sinc")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_sinc)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.image_sat_2 = QtWidgets.QLabel(self.tab_sinc)
        self.image_sat_2.setText("")
        self.image_sat_2.setAlignment(QtCore.Qt.AlignTop)
        self.image_sat_2.setObjectName("image_sat_2")
        self.gridLayout_5.addWidget(self.image_sat_2, 0, 0, 1, 1)
        self.tab_sinc.setWidget(self.image_sat_2)
        self.tabWidget_2.addTab(self.tab_sinc, "")
        self.tab_com = QtWidgets.QScrollArea()
        self.tab_com.setWidgetResizable(True)
        self.tab_com.setObjectName("tab_vivo")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.tab_com)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.image_sat_3 = QtWidgets.QLabel(self.tab_com)
        self.image_sat_3.setText("")
        self.image_sat_3.setAlignment(QtCore.Qt.AlignTop)
        self.image_sat_3.setObjectName("image_sat_3")
        self.gridLayout_8.addWidget(self.image_sat_3, 0, 0, 1, 1)
        self.tab_com.setWidget(self.image_sat_3)
        self.tabWidget_2.addTab(self.tab_com, "")
        self.gridLayout_10.addWidget(self.tabWidget_2, 0, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setObjectName("formLayout")
        self.button_file = QtWidgets.QPushButton(self.tab_2)
        self.button_file.setObjectName("button_file")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.button_file)
        self.button_rotate = QtWidgets.QPushButton(self.tab_2)
        self.button_rotate.setObjectName("button_rotate")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.button_rotate)
        self.button_proc = QtWidgets.QPushButton(self.tab_2)
        self.button_proc.setObjectName("button_proc")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.button_proc)
        self.gridLayout_10.addLayout(self.formLayout, 0, 2, 1, 2)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 964, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuOpciones = QtWidgets.QMenu(self.menuBar)
        self.menuOpciones.setObjectName("menuOpciones")
        self.menuOpciones_2 = QtWidgets.QMenu(self.menuBar)
        self.menuOpciones_2.setObjectName("menuOpciones_2")
        MainWindow.setMenuBar(self.menuBar)
        self.actionDesde_WAV = QtWidgets.QAction(MainWindow)
        self.actionDesde_WAV.setObjectName("actionDesde_WAV")
        self.actionDesde_WAV.setCheckable(True)
        self.actionDesde_RAW = QtWidgets.QAction(MainWindow)
        self.actionDesde_RAW.setObjectName("actionDesde_RAW")
        self.actionDesde_RAW.setCheckable(True)
        self.actionSalir = QtWidgets.QAction(MainWindow)
        self.actionSalir.setObjectName("actionSalir")
        self.actionEnviar_UDP = QtWidgets.QAction(MainWindow)
        self.actionEnviar_UDP.setCheckable(True)
        self.actionEnviar_UDP.setChecked(False)
        self.actionEnviar_UDP.setObjectName("actionEnviar_UDP")
        self.actionGuardar_WAV = QtWidgets.QAction(MainWindow)
        self.actionGuardar_WAV.setCheckable(True)
        self.actionGuardar_WAV.setObjectName("actionGuardar_WAV")
        self.actionGuardar_RAW = QtWidgets.QAction(MainWindow)
        self.actionGuardar_RAW.setCheckable(True)
        self.actionGuardar_RAW.setObjectName("actionGuardar_RAW")
        self.actionDecod_audio = QtWidgets.QAction(MainWindow)
        self.actionDecod_audio.setCheckable(True)
        self.actionDecod_audio.setObjectName("actionDecod_audio")
        self.actionFMCOMMS2 = QtWidgets.QAction(MainWindow)
        self.actionFMCOMMS2.setCheckable(True)
        self.actionFMCOMMS2.setObjectName("actionFMCOMMS2")
        self.actionRTL_SDR = QtWidgets.QAction(MainWindow)
        self.actionRTL_SDR.setCheckable(True)
        self.actionRTL_SDR.setChecked(True)
        self.actionRTL_SDR.setObjectName("actionRTL_SDR")
        self.actionHabilitar_Doppler = QtWidgets.QAction(MainWindow)
        self.actionHabilitar_Doppler.setCheckable(True)
        self.actionHabilitar_Doppler.setChecked(False)
        self.actionHabilitar_Doppler.setObjectName("actionHabilitar_Doppler")
        self.menuOpciones.addAction(self.actionDesde_WAV)
        self.menuOpciones.addAction(self.actionDesde_RAW)
        self.menuOpciones.addSeparator()
        self.menuOpciones.addAction(self.actionRTL_SDR)
        self.menuOpciones.addAction(self.actionFMCOMMS2)
        self.menuOpciones.addSeparator()
        self.menuOpciones.addAction(self.actionSalir)
        self.menuOpciones_2.addAction(self.actionEnviar_UDP)
        self.menuOpciones_2.addAction(self.actionGuardar_WAV)
        self.menuOpciones_2.addAction(self.actionGuardar_RAW)
        self.menuOpciones_2.addAction(self.actionDecod_audio)
        self.menuOpciones_2.addAction(self.actionHabilitar_Doppler)
        self.menuBar.addAction(self.menuOpciones.menuAction())
        self.menuBar.addAction(self.menuOpciones_2.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_15.setText(_translate("MainWindow", "Tasa de Muestreo"))
        self.label_2.setText(_translate("MainWindow", "Transición"))
        self.label_3.setText(_translate("MainWindow", "Ant"))
        self.label_sat.setText(_translate("MainWindow", "Satelites"))
        self.sample_rate_combo.setToolTip(_translate("MainWindow", "The rate at which data samples are produced"))
        self.label.setText(_translate("MainWindow", "Frec de Corte"))
        self.run_stop_button.setToolTip(_translate("MainWindow", "Comienza o detiene la reproducción"))
        self.run_stop_button.setText(_translate("MainWindow", "Comenzar"))
        self.label_4.setText(_translate("MainWindow", "Volumen"))
        self.bandwidth_label.setText(_translate("MainWindow", "Ancho de Banda Hz"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "Receptor de Satelites"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_vivo), _translate("MainWindow", "Imagen en vivo"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_sinc), _translate("MainWindow", "Imagen Sincronizada"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_com), _translate("MainWindow", "Imagen Compuesta"))
        self.button_file.setText(_translate("MainWindow", "Desde Archivo"))
        self.button_rotate.setText(_translate("MainWindow", "Rotar 180"))
        self.button_proc.setText(_translate("MainWindow", "Procesar"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Imagen"))
        self.menuOpciones.setTitle(_translate("MainWindow", "Fuente"))
        self.menuOpciones_2.setTitle(_translate("MainWindow", "Opciones"))
        self.actionDesde_WAV.setText(_translate("MainWindow", "Desde archivo WAV"))
        self.actionDesde_RAW.setText(_translate("MainWindow", "Desde archivo RAW"))
        self.actionSalir.setText(_translate("MainWindow", "Salir"))
        self.actionEnviar_UDP.setText(_translate("MainWindow", "Enviar vía UDP"))
        self.actionGuardar_WAV.setText(_translate("MainWindow", "Guardar WAV"))
        self.actionGuardar_RAW.setText(_translate("MainWindow", "Guardar RAW"))
        self.actionDecod_audio.setText(_translate("MainWindow", "Decodificar Audio"))
        self.actionHabilitar_Doppler.setText(_translate("MainWindow", "Habilitar correción de Doppler"))
        self.actionFMCOMMS2.setText(_translate("MainWindow", "FMCOMMS2"))
        self.actionRTL_SDR.setText(_translate("MainWindow", "RTL-SDR"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

