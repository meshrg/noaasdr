#!/usr/bin/env python
# -*- coding: utf-8 -*-

import StringIO
import sys
import os

from PyQt5 import Qt
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QApplication

from NOAA_SDR_GUI import Ui_MainWindow
import NOAA_APT
import FFTDisp
import Waterfall

import threading
import numpy as np
import scipy.signal
import datetime
from PIL import Image, ImageEnhance, ImageChops, ImageOps
from PIL.ImageQt import ImageQt


class NOAA_SDR(QMainWindow, Ui_MainWindow):
    def __init__(self, app):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)

        self.stop_thread = False
        self.app = app
        self.setupUi(self)
        self.setWindowTitle("NOAA APT SDR Decoding - GMC")
        self.imageLabel.setPixmap(QtGui.QPixmap("datos.jpg"))
        app.aboutToQuit.connect(self.app_quit)
        self.graphic_data = None
        self.config = self.get_default_config()
        self.full_rebuild_flag = True
        self.running = False
        self.enabled = False
        self.radio = NOAA_APT.NOAA_APT(self)
        self.decodedArray = None

        self.waterfall_widget = Waterfall.WaterfallWidget(self, self.config, self.waterfall_layout)
        self.fft_widget = FFTDisp.FFTDispWidget(self, self.config, self.fft_disp_layout)

        self.run_stop_button.clicked.connect(self.run_stop)
        self.button_file.clicked.connect(self.button_process_from_file)
        self.button_rotate.clicked.connect(self.button_rotate_click)
        self.button_proc.clicked.connect(self.button_image_process)
        self.actionEnviar_UDP.triggered.connect(self.enviar_udp_click)
        self.actionGuardar_WAV.triggered.connect(self.guardar_wav_click)
        self.actionGuardar_RAW.triggered.connect(self.guardar_raw_click)
        self.actionDecod_audio.triggered.connect(self.decode_audio_click)
        self.actionHabilitar_Doppler.triggered.connect(self.habilitar_doppler_click)
        noaa_list = [self.tr('NOAA 15'), self.tr('NOAA 18'), self.tr('NOAA 19')]
        self.satelite_combo.addItems(noaa_list)
        self.satelite_combo.activated.connect(self.satellite_change)
        sample_rate_list = [self.tr('960000'), self.tr('2000000')]
        self.sample_rate_combo.addItems(sample_rate_list)
        self.sample_rate_combo.setCurrentIndex(1)
        self.sample_rate_combo.activated.connect(self.sample_rate_change)
        bandwidth_list = [self.tr('42000'), self.tr('250000')]
        self.bandwidth_combo.addItems(bandwidth_list)
        self.bandwidth_combo.activated.connect(self.change_bandwidth)
        self.horizontalSlider.valueChanged.connect(self.volume_change)
        self.horizontalSlider_2.valueChanged.connect(self.lna_gain_change)
        self.cutoff_slider.valueChanged.connect(self.fir_cutoff_change)
        self.trans_slider.valueChanged.connect(self.fir_trans_change)
        self.actionDesde_WAV.triggered.connect(self.source_from_wav_change)
        self.actionDesde_RAW.triggered.connect(self.source_from_raw_change)
        self.actionRTL_SDR.triggered.connect(self.source_rtl_change)
        self.actionFMCOMMS2.triggered.connect(self.source_fmcomms2_change)

        QtCore.QTimer.singleShot(100, self.first_read_config)

    def enviar_udp_click(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionEnviar_UDP.setChecked(not self.actionEnviar_UDP.isChecked())
        else:
            if self.actionEnviar_UDP.isChecked():
                self.config['enableUDP'] = True
            else:
                self.config['enableUDP'] = False
            self.full_rebuild_flag = True

    def guardar_wav_click(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionGuardar_WAV.setChecked(not self.actionGuardar_WAV.isChecked())
        else:
            if self.actionGuardar_WAV.isChecked():
                self.config['file_path'] = str(QFileDialog.getExistingDirectory(self, 'Guardar')) + "/"
                if self.config['file_path'] == '/':
                    self.actionGuardar_WAV.setChecked(False)
                    return
                self.config['enableWAV'] = True
                self.full_rebuild_flag = True
            else:
                self.config['enableWAV'] = False
                self.full_rebuild_flag = True

    def guardar_raw_click(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionGuardar_RAW.setChecked(not self.actionGuardar_RAW.isChecked())
        else:
            if self.actionGuardar_RAW.isChecked():
                self.config['file_path'] = str(QFileDialog.getExistingDirectory(self, 'Guardar')) + "/"
                if self.config['file_path'] == '/':
                    self.actionGuardar_RAW.setChecked(False)
                    return
                self.config['enableRAW'] = True
                self.full_rebuild_flag = True
            else:
                self.config['enableRAW'] = False
                self.full_rebuild_flag = True

    def decode_audio_click(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionDecod_audio.setChecked(not self.actionDecod_audio.isChecked())
        else:
            if self.actionDecod_audio.isChecked():
                self.config['file_path'] = str(QFileDialog.getExistingDirectory(self, 'Guardar')) + "/"
                if self.config['file_path'] == '/':
                    self.actionDecod_audio.setChecked(False)
                    return
                self.config['enableDecode'] = True
                self.full_rebuild_flag = True
            else:
                self.config['enableDecode'] = False
                self.full_rebuild_flag = True

    def habilitar_doppler_click(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionHabilitar_Doppler.setChecked(not self.actionHabilitar_Doppler.isChecked())
        else:
            if self.actionHabilitar_Doppler.isChecked():
                self.message_dialog("Info", "Correción de Doppler habilitada. Debe ejecutar Gpredict y establecer el control de radio con Host: localhost y Puerto: 4532")
                self.config['enableDoppler'] = True
                self.full_rebuild_flag = True
            else:
                self.config['enableDoppler'] = False
                self.full_rebuild_flag = True

    def satellite_change(self):
        if self.satelite_combo.currentText() == 'NOAA 15':
            self.config['freq'] = 137.62e6
            self.update_freq(self.config['freq'])
            self.config['satellite'] = 'NOAA 15'
            self.radio.set_satellite('NOAA 15')
        elif self.satelite_combo.currentText() == 'NOAA 18':
            self.config['freq'] = 137.9125e6
            self.update_freq(self.config['freq'])
            self.config['satellite'] = 'NOAA 18'
            self.radio.set_satellite('NOAA 18')
        elif self.satelite_combo.currentText() == 'NOAA 19':
            self.config['freq'] = 137.1e6
            self.update_freq(self.config['freq'])
            self.config['satellite'] = 'NOAA 19'
            self.radio.set_satellite('NOAA 18')

    def volume_change(self):
        self.config['volume'] = float(self.horizontalSlider.value() / 100)
        if self.running:
            self.radio.set_volume(float(self.horizontalSlider.value()) / 100)

    def lna_gain_change(self):
        self.config['lna_gain'] = int(self.horizontalSlider_2.value())
        if self.running:
            self.radio.set_lna_gain(int(self.horizontalSlider_2.value()))

    def change_bandwidth(self):
        self.config['bandwidth'] = int(self.bandwidth_combo.currentText())
        self.radio.set_bandwidth(int(self.bandwidth_combo.currentText()))

    def fir_cutoff_change(self):
        self.config['fir_cutoff'] = int(self.cutoff_slider.value())
        self.radio.set_filter_cutoff(int(self.cutoff_slider.value()))

    def fir_trans_change(self):
        self.config['fir_trans'] = int(self.trans_slider.value())
        self.radio.set_filter_trans(int(self.trans_slider.value()))

    def sample_rate_change(self):
        self.config['sample_rate'] = int(self.sample_rate_combo.currentText())
        self.radio.set_sample_rate(int(self.sample_rate_combo.currentText()))

    def source_from_wav_change(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionDesde_WAV.setChecked(not self.actionDesde_WAV.isChecked())
        else:
            if self.actionDesde_WAV.isChecked():
                self.actionRTL_SDR.setChecked(False)
                self.actionFMCOMMS2.setChecked(False)
                self.actionDesde_RAW.setChecked(False)
                self.actionEnviar_UDP.setChecked(False)
                self.actionEnviar_UDP.setEnabled(False)
                self.actionGuardar_WAV.setChecked(False)
                self.actionGuardar_WAV.setEnabled(False)
                self.actionGuardar_RAW.setChecked(False)
                self.actionGuardar_RAW.setEnabled(False)
                self.actionDecod_audio.setChecked(False)
                self.actionDecod_audio.setEnabled(True)
                self.config['enableUDP'] = False
                self.config['enableWAV'] = False
                self.config['enableRAW'] = False
                self.config['enableDecode'] = False
                self.config['source'] = 'WAVFile'
                self.full_rebuild_flag = True
            else:
                self.actionDesde_WAV.setChecked(True)

    def source_from_raw_change(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionDesde_RAW.setChecked(not self.actionDesde_RAW.isChecked())
        else:
            if self.actionDesde_RAW.isChecked():
                self.actionRTL_SDR.setChecked(False)
                self.actionFMCOMMS2.setChecked(False)
                self.actionDesde_WAV.setChecked(False)
                self.actionEnviar_UDP.setChecked(False)
                self.actionEnviar_UDP.setEnabled(True)
                self.actionGuardar_WAV.setChecked(False)
                self.actionGuardar_WAV.setEnabled(True)
                self.actionGuardar_RAW.setChecked(False)
                self.actionGuardar_RAW.setEnabled(False)
                self.actionDecod_audio.setChecked(False)
                self.actionDecod_audio.setEnabled(True)
                self.config['enableUDP'] = False
                self.config['enableWAV'] = False
                self.config['enableRAW'] = False
                self.config['enableDecode'] = False
                self.config['source'] = 'RAWFile'
                self.full_rebuild_flag = True
            else:
                self.actionDesde_RAW.setChecked(True)

    def source_rtl_change(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionRTL_SDR.setChecked(not self.actionRTL_SDR.isChecked())
        else:
            if self.actionRTL_SDR.isChecked():
                self.actionFMCOMMS2.setChecked(False)
                self.actionDesde_WAV.setChecked(False)
                self.actionDesde_RAW.setChecked(False)
                self.actionGuardar_WAV.setEnabled(True)
                self.actionGuardar_WAV.setChecked(False)
                self.actionGuardar_RAW.setEnabled(True)
                self.actionGuardar_RAW.setChecked(False)
                self.config['enableUDP'] = False
                self.config['enableWAV'] = False
                self.config['enableRAW'] = False
                self.config['enableDecode'] = False
                self.config['source'] = 'RTL-SDR'
                self.full_rebuild_flag = True
            else:
                self.actionRTL_SDR.setChecked(True)

    def source_fmcomms2_change(self):
        if self.running:
            self.message_dialog("Info", "Primero deten la reproduccion")
            self.actionFMCOMMS2.setChecked(not self.actionFMCOMMS2.isChecked())
        else:
            if self.actionFMCOMMS2.isChecked():
                self.actionRTL_SDR.setChecked(False)
                self.actionDesde_WAV.setChecked(False)
                self.actionDesde_RAW.setChecked(False)
                self.actionGuardar_WAV.setEnabled(True)
                self.actionGuardar_WAV.setChecked(False)
                self.actionGuardar_RAW.setEnabled(True)
                self.actionGuardar_RAW.setChecked(False)
                self.config['enableUDP'] = False
                self.config['enableWAV'] = False
                self.config['enableRAW'] = False
                self.config['enableDecode'] = False
                self.config['source'] = 'FMCOMMS2'
                self.full_rebuild_flag = True
            else:
                self.actionFMCOMMS2.setChecked(True)

    def first_read_config(self):
        self.assign_freq(self.config['freq'])
        self.resize(1020, 600)
        self.float_to_splitter(0.65, self.splitter_v)
        self.float_to_splitter(0.80, self.splitter_h)
        self.enabled = True
        self.enable_disable_controls()

    def get_default_config(self):
        defaults = {
            'source': 'RTL-SDR',
            'file_path': '',
            'enableUDP': False,
            'enableWAV': False,
            'enableRAW': False,
            'enableDecode': False,
            'enableDoppler': False,
            'freq': 137.62e6,
            'satellite': 'NOAA 15',
            'sample_rate': 2000000,
            'bandwidth': 42000,
            'lna_gain': 100,
            'volume': 0.2,
            'fir_cutoff': 40,
            'fir_trans': 20,
            'dbscale_lo': -140,
            'dbscale_hi': 10,
            'fft_zoom': -1,
            'waterfall_bias': 150,
            'disp_trace_color': '#ffff00',
            'disp_text_color': '#80c0ff',
            'disp_vline_color': '#c00000',
        }
        return defaults

    def splitter_to_float(self, splitter):
        # creates normalized splitter position {0 ... 1}
        a, b = splitter.sizes()
        return float(a) / (a + b)

    def float_to_splitter(self, v, splitter):
        # requires normalized splitter position {0 ... 1}
        a, b = splitter.sizes()
        t = a + b
        aa = int(t * v)
        bb = t - aa
        splitter.setSizes([aa, bb])

    def assign_freq(self, f=None):
        if f is None:
            f = self.config['freq']
        self.lcdFreq.display(f / 1000)
        self.update_freq(f)

    def update_freq(self, f=None):
        if self.enabled:
            if f is None:
                f = self.config['freq']
                self.lcdFreq.display(f / 1000)
            else:
                self.config['freq'] = f
                self.lcdFreq.display(f / 1000)
        if self.radio.rtlsdr_source is not None:
            self.radio.set_center_freq(f)

    def run_stop(self):
        self.running = not self.running
        if self.running:
            self.radio.stop()
            self.radio.wait()
            self.radio.disconnect_all()
            if self.full_rebuild_flag:
                self.radio = NOAA_APT.NOAA_APT(self)
                self.full_rebuild_flag = False
            if self.config['source'] != 'WAVFile' and self.config['source'] != 'RAWFile':
                self.update_freq()
            else:
                self.lcdFreq.display("F1LE")
            if self.radio.initialize_radio():
                self.radio.start()

                if self.config['enableDecode']:
                    self.button_file.setEnabled(False)
                    self.button_proc.setEnabled(False)
                    self.tab_com.setEnabled(False)
                    self.tab_sinc.setEnabled(False)
                    self.stop_thread = False
                    self.t = threading.Thread(target=self.worker)
                    self.t.daemon = True
                    self.t.start()
                if self.config['source'] == 'WAVFile':
                    self.enable_disable_controls()
                    self.satelite_combo.setEnabled(False)
                elif self.config['source'] == 'RAWFile':
                    self.enable_disable_controls()
                    self.satelite_combo.setEnabled(False)
                    self.cutoff_slider.setEnabled(True)
                    self.trans_slider.setEnabled(True)
                else:
                    self.enable_disable_controls(True)
                    self.satelite_combo.setEnabled(True)
                self.run_stop_button.setText("Detener")
            else:
                self.running = False
        else:
            self.radio.stop()
            self.stop_thread = True
            self.t.join()
            self.radio.wait()
            self.radio.disconnect_all()
            self.run_stop_button.setText("Comenzar")
            self.satelite_combo.setEnabled(True)
            self.button_file.setEnabled(False)
            self.button_proc.setEnabled(False)
            self.tab_com.setEnabled(False)
            self.tab_sinc.setEnabled(False)

    def message_dialog(self, title, message):
        mb = QMessageBox(QMessageBox.Warning, title, message, QMessageBox.Ok)
        mb.exec_()

    def draw_fft_disp(self):
        if self.graphic_data is not None:
            self.fft_widget.accept_data(self.graphic_data)
            self.graphic_data = None

    def enable_disable_controls(self, enabled=False):
        if enabled:
            self.sample_rate_combo.setEnabled(True)
            self.bandwidth_combo.setEnabled(True)
            self.cutoff_slider.setEnabled(True)
            self.trans_slider.setEnabled(True)
            self.horizontalSlider_2.setEnabled(True)
        else:
            self.sample_rate_combo.setEnabled(False)
            self.bandwidth_combo.setEnabled(False)
            self.cutoff_slider.setEnabled(False)
            self.trans_slider.setEnabled(False)
            self.horizontalSlider_2.setEnabled(False)

    def app_quit(self):
        self.running = False
        self.enabled = False
        self.radio.stop()
        self.radio.wait()
        self.radio.disconnect_all()
        Qt.QApplication.quit()

    def button_process_from_file(self):
        filename = QFileDialog.getOpenFileName(self, 'Abrir imagen en DAT', '', "DAT files (*.dat)")
        self.image_sat.clear()
        self.image_sat_2.clear()
        self.image_sat_3.clear()
        try:
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            with open(str(filename[0]), 'rb') as file_object:
                data = np.fromfile(file_object, dtype=np.uint8)

                data = data[0:len(data) - (len(data) % 2080)]
                filtered = scipy.signal.medfilt(np.abs(data), 5)
                digitized = self._digitize(filtered)
                self.decodedArray = digitized
                matrix = digitized.reshape(len(digitized) // 2080, 2080)
                image = Image.fromarray(matrix)
                ratio = image.width/image.height
                image = image.resize((self.image_sat.width(), self.image_sat.width()/ratio), Image.BICUBIC)

                qim = ImageQt(image)
                pix = QtGui.QPixmap.fromImage(qim)
                self.image_sat.setPixmap(pix)

        except Exception as e:
            print(e)
        QApplication.restoreOverrideCursor()

    def button_rotate_click(self):
        if self.tabWidget_2.currentIndex() == 0:
            img = self.image_sat.pixmap().toImage()
            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.ReadWrite)
            img.save(buffer, "PNG")
            strio = StringIO.StringIO()
            strio.write(buffer.data())
            buffer.close()
            strio.seek(0)
            pil_im = Image.open(strio)
            pil_im = pil_im.rotate(180)

            self.image_sat.clear()
            qim = ImageQt(pil_im)
            pix = QtGui.QPixmap.fromImage(qim)
            self.image_sat.setPixmap(pix)

        elif self.tabWidget_2.currentIndex() == 1:
            img = self.image_sat_2.pixmap().toImage()
            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.ReadWrite)
            img.save(buffer, "PNG")
            strio = StringIO.StringIO()
            strio.write(buffer.data())
            buffer.close()
            strio.seek(0)
            pil_im = Image.open(strio)
            pil_im = pil_im.rotate(180)

            self.image_sat_2.clear()
            qim = ImageQt(pil_im)
            pix = QtGui.QPixmap.fromImage(qim)
            self.image_sat_2.setPixmap(pix)

        elif self.tabWidget_2.currentIndex() == 2:
            img = self.image_sat_3.pixmap().toImage()
            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.ReadWrite)
            img.save(buffer, "PNG")
            strio = StringIO.StringIO()
            strio.write(buffer.data())
            buffer.close()
            strio.seek(0)
            pil_im = Image.open(strio)
            pil_im = pil_im.rotate(180)

            self.image_sat_3.clear()
            qim = ImageQt(pil_im)
            pix = QtGui.QPixmap.fromImage(qim)
            self.image_sat_3.setPixmap(pix)

    def button_image_process(self):
        try:
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            if self.decodedArray is not None:
                if self.image_sat_2.pixmap() is None:
                    matrix = self._reshape(self.decodedArray)
                    image = Image.fromarray(matrix)
                    ratio = image.width / image.height
                    image = image.resize((self.image_sat.width(), self.image_sat.width() / ratio), Image.BICUBIC)

                    qim = ImageQt(image)
                    pix = QtGui.QPixmap.fromImage(qim)
                    self.image_sat_2.setPixmap(pix)

                    width, height = image.size
                    left_image = image.crop(((0, 0, width / 2, height)))
                    right_image = image.crop(((width / 2, 0, width, height)))
                    composite = Image.merge('RGB', (
                    ImageEnhance.Brightness(left_image).enhance(2), ImageEnhance.Brightness(left_image).enhance(3),
                    ImageEnhance.Brightness(right_image).enhance(1.5)))

                    qim = ImageQt(composite)
                    pix = QtGui.QPixmap.fromImage(qim)
                    self.image_sat_3.setPixmap(pix)

        except Exception as e:
            print(e)
        QApplication.restoreOverrideCursor()
        self.tabWidget_2.setCurrentIndex(2)

    def worker(self):
        while True:

            try:
                with open(self.radio.decodeFilename, 'rb') as file_object:
                    data = np.fromfile(file_object, dtype=np.uint8)

                    data = data[0:len(data)-(len(data)%2080)]
                    filtered = scipy.signal.medfilt(np.abs(data), 5)
                    digitized = self._digitize(filtered)
                    self.decodedArray = digitized
                    matrix = digitized.reshape(len(digitized)//2080, 2080)
                    image = Image.fromarray(matrix)
                    ratio = image.width / image.height
                    image = image.resize((self.image_sat.width(), self.image_sat.width() / ratio), Image.BICUBIC)

                    qim = ImageQt(image)
                    pix = QtGui.QPixmap.fromImage(qim)
                    self.image_sat.setPixmap(pix)

            except Exception as e:
                print(e)
                continue

            if self.stop_thread:
                break
        return

    def _digitize(self, signal, plow=0.5, phigh=99.5):
        '''
        Convert signal to numbers between 0 and 255.

        FROM APT-DECODER (github.com/zacstewart/apt-decoder)
        '''
        (low, high) = np.percentile(signal, (plow, phigh))
        delta = high - low
        data = np.round(255 * (signal - low) / delta)
        data[data < 0] = 0
        data[data > 255] = 255
        return data.astype(np.uint8)

    def _reshape(self, signal):
        '''
        Find sync frames and reshape the 1D signal into a 2D image.

        Finds the sync A frame by looking at the maximum values of the cross
        correlation between the signal and a hardcoded sync A frame.

        The expected distance between sync A frames is 2080 samples, but with
        small variations because of Doppler effect.

        FROM APT-DECODER (github.com/zacstewart/apt-decoder)
        '''
        # sync frame to find: seven impulses and some black pixels (some lines
        # have something like 8 black pixels and then white ones)
        syncA = [0, 128, 255, 128] * 7 + [0] * 7

        # list of maximum correlations found: (index, value)
        peaks = [(0, 0)]

        # minimum distance between peaks
        mindistance = 2000

        # need to shift the values down to get meaningful correlation values
        signalshifted = [x - 128 for x in signal]
        syncA = [x - 128 for x in syncA]
        for i in range(len(signal) - len(syncA)):
            corr = np.dot(syncA, signalshifted[i: i + len(syncA)])

            # if previous peak is too far, keep it and add this value to the
            # list as a new peak
            if i - peaks[-1][0] > mindistance:
                peaks.append((i, corr))

            # else if this value is bigger than the previous maximum, set this
            # one
            elif corr > peaks[-1][1]:
                peaks[-1] = (i, corr)

        # create image matrix starting each line on the peaks found
        matrix = []
        for i in range(len(peaks) - 1):
            matrix.append(signal[peaks[i][0]: peaks[i][0] + 2080])

        return np.array(matrix)


if __name__ == "__main__":
    pd = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(pd)
    app = Qt.QApplication(sys.argv)
    window = NOAA_SDR(app)
    window.show()
    sys.exit(app.exec_())
