#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: CIDTE-Receptor APT NOAA-SAT
# Author: Gareth Montenegro Chaidez
# Description: Receptor Satelital para NOAA
# Generated: Sat Jun 15 12:12:52 2019
##################################################

from datetime import datetime
from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import logpwrfft

import osmosdr
import time
import numpy as np
import pmt
import gpredict

from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QFileDialog


class DrawGraphics(QtCore.QObject):
    draw = QtCore.pyqtSignal()


class MyVectorSink(gr.sync_block):
    def __init__(self, main, sz):
        self.main = main
        self.sz = sz

        gr.sync_block.__init__(
            self,
            name="My Vector sink",
            in_sig=[(np.float32, self.sz)],
            out_sig=None,
        )
        # event-related
        self.drawgr = DrawGraphics()
        self.drawgr.draw.connect(self.main.draw_fft_disp)

    def work(self, input_items, output_items):
        if self.main.graphic_data is None:
            data = np.fft.fftshift(input_items)
            self.main.graphic_data = data[0][0].tolist()
            self.drawgr.draw.emit()
        return len(input_items)


class NOAA_APT(gr.top_block, QWidget):

    def __init__(self, main):
        self.main = main
        gr.top_block.__init__(self, "CIDTE-Receptor APT NOAA-SAT")
        QWidget.__init__(self)

        ##################################################
        # Variables
        ##################################################

        self.enableUDP = self.main.config['enableUDP']
        self.enableWAV = self.main.config['enableWAV']
        self.enableRAW = self.main.config['enableRAW']
        self.enableDecode = self.main.config['enableDecode']
        self.enableDoppler = self.main.config['enableDoppler']
        self.source = self.main.config['source']
        self.fft_size = 4096
        self.frame_rate = 60
        self.average = 0.50
        self.logpwrfft = None
        self.center_freq = self.main.config['freq']
        self.volume = self.main.config['volume']
        self.udp_port = 10027
        self.udp_ip_address = "192.168.1.13"
        self.satellite = self.main.config['satellite']
        self.sample_rate = self.main.config['sample_rate']
        self.lna_gain = self.main.config['lna_gain']
        self.filter_trans = self.main.config['fir_trans']
        self.filter_cutoff = self.main.config['fir_cutoff']
        self.doppler_freq = self.center_freq
        self.dir_path = self.main.config['file_path']
        self.bandwidth = self.main.config['bandwidth']
        self.decodeFilename = self.dir_path + self.satellite + "_" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".dat"

        self.rtlsdr_source = None
        self.currently_configured_device = None

    def initialize_radio(self):
        if self.configure_source():
            self.build_blocks()
            self.connect_blocks()
            self.main.full_rebuild_flag = False
        else:
            print("Ocurrio un Error inicializando la fuente")
            return False
        return True

    def configure_source(self):
        if self.source == 'WAVFile':
            filename = QFileDialog.getOpenFileName(self, 'Abrir WAV', '', "WAV files (*.wav)")
            if filename[0] == '':
                return False
            self.blocks_wavfile_source = blocks.wavfile_source(str(filename[0]), False)
            self.rational_resampler_upsample_wav = filter.rational_resampler_fff(
                interpolation=4,
                decimation=1,
                taps=None,
                fractional_bw=None,
            )
        elif self.source == 'RAWFile':
            filename = QFileDialog.getOpenFileName(self, 'Abrir RAW', '', "RAW files (*.raw)")
            if filename[0] == '':
                return False
            self.blocks_file_source_raw = blocks.file_source(gr.sizeof_gr_complex * 1, str(filename[0]), False)
            #self.blocks_file_source_raw.set_begin_tag(pmt.PMT_NIL)
            self.blocks_throttle_raw = blocks.throttle(gr.sizeof_gr_complex * 1, self.sample_rate, True)
        elif self.source == 'RTL-SDR':
            if self.rtlsdr_source is None or 'rtl' != self.currently_configured_device:
                self.rtlsdr_source = osmosdr.source(args="numchan=" + str(1) + " " + '')
                self.currently_configured_device = 'rtl'
            gain_names = self.rtlsdr_source.get_gain_names()
            if len(gain_names) == 0:
                self.main.run_stop_button.setEnabled(False)
                return False
            else:
                self.main.run_stop_button.setEnabled(True)
                self.rtlsdr_source.set_time_now(osmosdr.time_spec_t(time.time()), osmosdr.ALL_MBOARDS)
                self.rtlsdr_source.set_sample_rate(self.sample_rate)
                self.rtlsdr_source.set_center_freq(self.center_freq, 0)
                self.rtlsdr_source.set_freq_corr(0, 0)
                self.rtlsdr_source.set_dc_offset_mode(2, 0)
                self.rtlsdr_source.set_iq_balance_mode(2, 0)
                self.rtlsdr_source.set_gain_mode(False, 0)
                self.rtlsdr_source.set_gain(self.lna_gain, 0)
                self.rtlsdr_source.set_if_gain(self.lna_gain, 0)
                self.rtlsdr_source.set_bb_gain(self.lna_gain, 0)
                self.rtlsdr_source.set_antenna('', 0)
                self.rtlsdr_source.set_bandwidth(self.bandwidth, 0)
                self.main.run_stop_button.setEnabled(True)
        elif self.source == 'FMCOMMS2':
            self.source = ''

        return True

    def build_blocks(self):
        if self.source != 'WAVFile':
            if self.enableDoppler:
                self.gpredict_doppler = gpredict.doppler(self.set_doppler_freq, "localhost", 4532, False)
            self.rational_resampler_downsample_to_44100 = filter.rational_resampler_fff(
                interpolation=44100,
                decimation=400000,
                taps=None,
                fractional_bw=None,
            )

            self.freq_xlating_fir_filter = filter.freq_xlating_fir_filter_ccc(1, (
                firdes.low_pass(1, self.sample_rate, self.filter_cutoff * 1000, self.filter_trans * 1000, firdes.WIN_HAMMING, 6.76)),
                                                                                    0, self.sample_rate)
            self.blocks_multiply_signal_doppler = blocks.multiply_vcc(1)
            self.blocks_multiply_const_volume = blocks.multiply_const_vff((self.volume,))
            self.audio_sink = audio.sink(44100, '', True)
            self.analog_wfm_rcv = analog.wfm_rcv(
                quad_rate=self.sample_rate,
                audio_decimation=5,
            )
            self.analog_sig_source_doppler = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                                             self.center_freq - self.doppler_freq, 1, 0)

            self.logpwrfft = logpwrfft.logpwrfft_c(
                sample_rate=self.sample_rate,
                fft_size=self.fft_size,
                ref_scale=2,
                frame_rate=self.frame_rate,
                avg_alpha=self.average,
                average=(self.average != 1),
            )

            self.fft_vector_sink = MyVectorSink(self.main, self.fft_size)

            if self.enableWAV or self.enableDecode:
                self.rational_resampler_downsample_to_11025 = filter.rational_resampler_fff(
                    interpolation=1,
                    decimation=4,
                    taps=None,
                    fractional_bw=None,
                )
                self.blocks_multiply_const_wav_volume = blocks.multiply_const_vff((.6,))

            if self.enableWAV:
                self.blocks_wavfile_sink = blocks.wavfile_sink(self.decodeFilename, 1, 11025, 16)

            if self.enableUDP:
                self.rational_resampler_udp = filter.rational_resampler_ccc(
                    interpolation=96000,
                    decimation=2000000,
                    taps=None,
                    fractional_bw=None,
                )
                self.blocks_udp_sink = blocks.udp_sink(gr.sizeof_gr_complex * 1, self.udp_ip_address, self.udp_port, 1472, True)

            if self.enableRAW:
                self.blocks_file_sink_raw = blocks.file_sink(gr.sizeof_gr_complex * 1,
                                                           self.dir_path + self.satellite + "_" + datetime.now().strftime(
                                                               "%Y.%m.%d.%H.%M.%S") + ".raw", False)
                self.blocks_file_sink_raw.set_unbuffered(False)

            if self.enableDecode:
                self.rational_resampler_upsample_for_hilbert = filter.rational_resampler_fff(
                    interpolation=16640,
                    decimation=11025,
                    taps=None,
                    fractional_bw=None,
                )
                self.band_pass_filter = filter.interp_fir_filter_fff(1, firdes.band_pass(1, 11025, 500, 4.2e3, 200,
                                                                                           firdes.WIN_HAMMING, 6.76))
                self.hilbert_fc = filter.hilbert_fc(65, firdes.WIN_HAMMING, 6.76)
                self.blocks_complex_to_mag = blocks.complex_to_mag(1)
                self.rational_resampler_downsample_to_4600 = filter.rational_resampler_fff(
                    interpolation=1,
                    decimation=4,
                    taps=None,
                    fractional_bw=None,
                )
                self.blocks_multiply_const_to_image = blocks.multiply_const_vff((255,))
                self.blocks_float_to_uchar = blocks.float_to_uchar()
                self.blocks_file_sink_decoded = blocks.file_sink(gr.sizeof_char * 1, self.decodeFilename, False)
                self.blocks_file_sink_decoded.set_unbuffered(False)
        else:
            self.audio_sink = audio.sink(44100, '', True)
            self.logpwrfft = logpwrfft.logpwrfft_f(
                sample_rate=11025,
                fft_size=4096,
                ref_scale=2,
                frame_rate=60,
                avg_alpha=self.average,
                average=(self.average != 1),
            )
            self.fft_vector_sink = MyVectorSink(self.main, self.fft_size)
            self.blocks_multiply_const_volume = blocks.multiply_const_vff((self.volume,))

            if self.enableDecode:
                self.rational_resampler_upsample_for_hilbert = filter.rational_resampler_fff(
                    interpolation=16640,
                    decimation=11025,
                    taps=None,
                    fractional_bw=None,
                )
                self.band_pass_filter = filter.interp_fir_filter_fff(1, firdes.band_pass(1, 11025, 500, 4.2e3, 200,
                                                                                           firdes.WIN_HAMMING, 6.76))
                self.hilbert_fc = filter.hilbert_fc(65, firdes.WIN_HAMMING, 6.76)
                self.blocks_complex_to_mag = blocks.complex_to_mag(1)
                self.rational_resampler_downsample_to_4600 = filter.rational_resampler_fff(
                    interpolation=1,
                    decimation=4,
                    taps=None,
                    fractional_bw=None,
                )
                self.blocks_multiply_const_to_image = blocks.multiply_const_vff((255,))
                self.blocks_float_to_uchar = blocks.float_to_uchar()
                self.blocks_file_sink_decoded = blocks.file_sink(gr.sizeof_char * 1, self.decodeFilename, False)
                self.blocks_file_sink_decoded.set_unbuffered(False)

    def connect_blocks(self):
        if self.source != 'WAVFile':
            self.connect((self.logpwrfft, 0), (self.fft_vector_sink, 0))
            self.connect((self.analog_sig_source_doppler, 0), (self.blocks_multiply_signal_doppler, 1))
            self.connect((self.analog_wfm_rcv, 0), (self.rational_resampler_downsample_to_44100, 0))
            self.connect((self.blocks_multiply_const_volume, 0), (self.audio_sink, 0))
            self.connect((self.blocks_multiply_signal_doppler, 0), (self.freq_xlating_fir_filter, 0))
            self.connect((self.freq_xlating_fir_filter, 0), (self.analog_wfm_rcv, 0))
            self.connect((self.rational_resampler_downsample_to_44100, 0), (self.blocks_multiply_const_volume, 0))

            if self.enableWAV or self.enableDecode:
                self.connect((self.rational_resampler_downsample_to_44100, 0), (self.rational_resampler_downsample_to_11025, 0))
                self.connect((self.rational_resampler_downsample_to_11025, 0), (self.blocks_multiply_const_wav_volume, 0))

            if self.enableWAV:
                self.connect((self.blocks_multiply_const_wav_volume, 0), (self.blocks_wavfile_sink, 0))

            if self.enableRAW:
                if self.source == 'RTL-SDR':
                    self.connect((self.rtlsdr_source, 0), (self.blocks_file_sink_raw, 0))

            if self.enableUDP:
                if self.source == 'RTL-SDR':
                    self.connect((self.rtlsdr_source, 0), (self.rational_resampler_udp, 0))
                elif self.source == 'RAWFile':
                    self.connect((self.blocks_throttle_raw, 0), (self.rational_resampler_udp, 0))
                self.connect((self.rational_resampler_udp, 0), (self.blocks_udp_sink, 0))

            if self.enableDecode:
                self.connect((self.blocks_multiply_const_wav_volume, 0), (self.band_pass_filter, 0))
                self.connect((self.band_pass_filter, 0), (self.rational_resampler_upsample_for_hilbert, 0))
                self.connect((self.blocks_complex_to_mag, 0), (self.rational_resampler_downsample_to_4600, 0))
                self.connect((self.blocks_float_to_uchar, 0), (self.blocks_file_sink_decoded, 0))
                self.connect((self.blocks_multiply_const_to_image, 0), (self.blocks_float_to_uchar, 0))
                self.connect((self.hilbert_fc, 0), (self.blocks_complex_to_mag, 0))
                self.connect((self.rational_resampler_upsample_for_hilbert, 0), (self.hilbert_fc, 0))
                self.connect((self.rational_resampler_downsample_to_4600, 0), (self.blocks_multiply_const_to_image, 0))
        else:
            self.connect((self.blocks_wavfile_source, 0), (self.rational_resampler_upsample_wav, 0))
            self.connect((self.blocks_wavfile_source, 0), (self.logpwrfft, 0))
            self.connect((self.logpwrfft, 0), (self.fft_vector_sink, 0))
            self.connect((self.rational_resampler_upsample_wav, 0), (self.blocks_multiply_const_volume, 0))
            self.connect((self.blocks_multiply_const_volume, 0), (self.audio_sink, 0))

            if self.enableDecode:
                self.connect((self.blocks_wavfile_source, 0), (self.band_pass_filter, 0))
                self.connect((self.band_pass_filter, 0), (self.rational_resampler_upsample_for_hilbert, 0))
                self.connect((self.blocks_complex_to_mag, 0), (self.rational_resampler_downsample_to_4600, 0))
                self.connect((self.blocks_float_to_uchar, 0), (self.blocks_file_sink_decoded, 0))
                self.connect((self.blocks_multiply_const_to_image, 0), (self.blocks_float_to_uchar, 0))
                self.connect((self.hilbert_fc, 0), (self.blocks_complex_to_mag, 0))
                self.connect((self.rational_resampler_upsample_for_hilbert, 0), (self.hilbert_fc, 0))
                self.connect((self.rational_resampler_downsample_to_4600, 0), (self.blocks_multiply_const_to_image, 0))

        if self.source == 'RTL-SDR':
            self.connect((self.rtlsdr_source, 0), (self.logpwrfft, 0))
            self.connect((self.rtlsdr_source, 0), (self.blocks_multiply_signal_doppler, 0))
        elif self.source == 'FMCOMMS2':
            self.source = ''
        elif self.source == 'RAWFile':
            self.connect((self.blocks_file_source_raw, 0), (self.blocks_throttle_raw, 0))
            self.connect((self.blocks_throttle_raw, 0), (self.logpwrfft, 0))
            self.connect((self.blocks_throttle_raw, 0), (self.blocks_multiply_signal_doppler, 0))

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.set_doppler_freq(self.center_freq)

        if self.source == 'RTL-SDR':
            self.rtlsdr_source.set_center_freq(self.center_freq, 0)
        elif self.source == 'FMCOMMS2':
            self.source = ''

        self.analog_sig_source_doppler.set_frequency(self.center_freq-self.doppler_freq)

    def get_volume(self):
        return self.volume

    def set_volume(self, volume):
        self.volume = volume
        self.blocks_multiply_const_volume.set_k((self.volume, ))

    def get_udp_port(self):
        return self.udp_port

    def set_udp_port(self, udp_port):
        self.udp_port = udp_port

    def get_udp_ip_address(self):
        return self.udp_ip_address

    def set_udp_ip_address(self, udp_ip_address):
        self.udp_ip_address = udp_ip_address

    def get_satellite(self):
        return self.satellite

    def set_satellite(self, satellite):
        self.satellite = satellite
        self.decodeFilename = self.dir_path + self.satellite + "_" + datetime.now().strftime(
            "%Y.%m.%d.%H.%M.%S") + ".dat"
        if self.enableWAV:
            self.blocks_wavfile_sink.open(self.dir_path + self.satellite + "_" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".wav")
        if self.enableDecode:
            self.blocks_file_sink_decoded.open(self.decodeFilename)
        if self.enableRAW:
            self.blocks_file_sink_raw.open(self.dir_path + self.satellite + "_" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".raw")

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

        if self.source == 'RAWFile':
            return
        if self.source == 'RTL-SDR':
            self.rtlsdr_source.set_sample_rate(self.sample_rate)
        elif self.source == 'FMCOMMS2':
            return

        self.freq_xlating_fir_filter.set_taps((firdes.low_pass(1, self.sample_rate, self.filter_cutoff*1000, self.filter_trans*1000, firdes.WIN_HAMMING, 6.76)))
        self.analog_sig_source_doppler.set_sampling_freq(self.sample_rate)

    def get_lna_gain(self):
        return self.lna_gain

    def set_lna_gain(self, lna_gain):
        self.lna_gain = lna_gain

        if self.source == 'RTL-SDR':
            self.rtlsdr_source.set_gain(self.lna_gain, 0)
            self.rtlsdr_source.set_if_gain(self.lna_gain, 0)
            self.rtlsdr_source.set_bb_gain(self.lna_gain, 0)
        elif self.source == 'FMCOMMS2':
            return

    def get_filter_trans(self):
        return self.filter_trans

    def set_filter_trans(self, filter_trans):
        self.filter_trans = filter_trans
        self.freq_xlating_fir_filter.set_taps((firdes.low_pass(1, self.sample_rate, self.filter_cutoff*1000, self.filter_trans*1000, firdes.WIN_HAMMING, 6.76)))

    def get_filter_cutoff(self):
        return self.filter_cutoff

    def set_filter_cutoff(self, filter_cutoff):
        self.filter_cutoff = filter_cutoff
        self.freq_xlating_fir_filter.set_taps((firdes.low_pass(1, self.sample_rate, self.filter_cutoff*1000, self.filter_trans*1000, firdes.WIN_HAMMING, 6.76)))

    def get_doppler_freq(self):
        return self.doppler_freq

    def set_doppler_freq(self, doppler_freq):
        self.doppler_freq = doppler_freq
        self.analog_sig_source_doppler.set_frequency(self.center_freq-self.doppler_freq)

    def get_dir_path(self):
        return self.dir_path

    def set_dir_path(self, dir_path):
        self.dir_path = dir_path
        self.decodeFilename = self.dir_path + self.satellite + "_" + datetime.now().strftime(
            "%Y.%m.%d.%H.%M.%S") + ".dat"
        if self.enableWAV:
            self.blocks_wavfile_sink.open(self.dir_path + self.satellite + "_" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".wav")
        if self.enableDecode:
            self.blocks_file_sink_decoded.open(self.decodeFilename)
        if self.enableRAW:
            self.blocks_file_sink_raw.open(self.dir_path + self.satellite + "_" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + ".raw")

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth

        if self.source == 'RTL-SDR':
            self.rtlsdr_source.set_bandwidth(self.bandwidth, 0)
        elif self.source == 'FMCOMMS2':
            return

    def compute_offset_f(self, front_end=True):
        fir_offset_f = self.limit_offset_range(0, 44100 / 2)
        if front_end:
            return fir_offset_f - self.filter_cutoff
        else:
            return -(fir_offset_f + self.filter_cutoff)

    def limit_offset_range(self, a, b):
        f = abs(a)
        sign = (-1, 1)[a >= 0]
        f = (f, b)[f > b]
        return f * sign