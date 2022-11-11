from warnings import warn
from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import filtfilt,firwin
from scipy.interpolate import interp1d 

class C_signal :
    """
    Methods
    -------
        
    Who give a float :
    - `std()`
    - `max()`
    - `min()`
    - `val_at_nearest_t(t:)`
    - `t_at_max()`
    - `t_at_min()`
    - `duration()`

    Who plot something :
    - `plot()`
    - `plot_fft()`
    - `plot_study_domain()`

    Who return a signal :
    - `recut()`
    - `copy()`
    """


    def __init__(           self,
                            data : np.ndarray | list,
                            fs : float,
                            unit : str = '',
                            name : str = ''):
        """
        Args :
        ------
        `data` : Array or list of values
        `fs` : Sampling frequency (Hz)  
        `unit` : unit of the signal
        `name` : name of the signal
        """
        match type(data):
            case np.ndarray :
                self.data = data 
            case _ :
                if data is None : 
                    warn("Data = None")
                else : self.data = np.array(data)

        self.fs = fs
        self.unit = unit
        self.name = name

    def mean(self)->float : return np.mean(self.data)
    def std(self)-> float : return np.std(self.data)
    def max(self)-> float : return np.max(self.data)
    def min(self)-> float : return np.min(self.data)
    def val_at_nearest_t(self,t: float)-> float : return self.data[int(t*self.fs)]
    def t_at_max(self)-> float : return np.argmax(self.data) / self.fs
    def t_at_min(self)-> float : return np.argmin(self.data) / self.fs
    def duration(self)-> float : return len(self.data)/self.fs

    # def plot() : pass
    def plot_fft(self,n=None):
        """
        Args 
        -----
        - `n`: Size of the signal after zero padding. It's need `n`>= len(data)
        """
        if n is None : n = len(self.data)

        if n < len(self.data):
            print("n is smaller than self.data, we use n=len(self.data) instead")
            n=len(self.data)

        # FFT avec 0 padding sur N
        plt.figure()
        yf = np.fft.fft(self.data,n)
        xf = np.linspace(0,self.fs/2,n//2-1)
        # xf = np.fft.fftfreq(n,d=1/fs)[:n//2-1]
        yf_real = 2/np.size(self.data)*np.abs(yf[0:n//2-1])
        # multiplication par 2 car puissances divisée entre freqs positifs et négatifs
        # division par la taille de max_spectro pour prendre en compte le pas 
        plt.plot(xf,yf_real)


    def plot_ADD_box_on_recut() : pass
    def plot_ADD_box_on_fft_recut() : pass
    def plot_ADD_mean_on_recut() : pass
    
    def recut(self) : pass

    def copy(self):
        return C_signal(data = self.data.copy(),
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name)















    # def mean_on_recut(  self,
    #                     t_start=0,
    #                     t_end=None,
    #                     plot=False,
    #                     close=True,
    #                     plot_recut=False):

    #     if t_end is None : t_end = self.give_duration()
    #     if t_end < 0 : t_end += self.give_duration()
    #     sig_recut = self.re_cut(t_start,t_end)
    #     mean = np.mean(sig_recut.data)

    #     if plot and not plot_recut :
    #         self.plot(close=close)
    #         plt.plot([t_start,t_end],[mean]*2,'--r',label=f"{self.name} moyen (={mean:.2f} {self.unit})")
    #         plt.legend()

    #     if plot and plot_recut : 
    #         sig_recut.plot(close=close)
    #         plt.plot([0,t_end - t_start],[mean]*2,'--r',label=f"{self.name} moyen (={mean:.2f} {self.unit})")
    #         plt.legend()
    #     return mean



    # def derivate(self,order=1):
    #     if order == 0 : return self.copy()
    #     if order != 1 :
    #         rep_0 = self.derivate(order = order-1)
    #         return rep_0.derivate()
    #     data_n = self.data.copy()
    #     data_n_1 = np.roll(self.data, 1)
    #     data_d = (data_n - data_n_1) * self.fs
    #     rep = self.copy()
    #     rep.name += " derivated"
    #     rep.data = data_d[1:]
    #     return rep


    # def re_cut( self,
    #             t1:float|None = None,
    #             t2:float|None = None    ):
    #     """
    #     Cette fonction rend un copie recoupée du signal

    #     args
    #     ----
    #     `t1` = temps de début (en s)  

    #     `t2` = temps de fin (en s)

    #     return
    #     ------
    #     `signal_recut` = a copy of the original signal but recuted
    #     """
    #     rep = self.copy()
    #     rep.name += " re_cut"
        
    #     if t1 is None: t1=0
        
    #     if t2 is None: rep.data = rep.data[ int(t1*self.fs) : ]
    #     else : rep.data = rep.data[ int(t1*self.fs) : int(t2*self.fs) ]
        
    #     return rep 


    # def plot(self,      color=None,
    #                     close=True,
    #                     draw_points=False,
    #                     axis_label=True,
    #                     linewidth=None,
    #                     add_to_title='',
    #                     new_unit : tuple[str,float] = None):


    def plot(   self,
                new_unit : tuple[str,float] = None,
                add_to_title='',
                new_figure = True,
                **kwargs):

        # TODO create a unit class
        
        if new_figure : plt.figure()
        unit = self.unit 
        data = self.data
        if new_unit is not None :
            unit = new_unit[0]
            data = data*new_unit[1]

        plt.grid(True)
        N = len(data)
        X=np.linspace(start=0 , stop=N/self.fs, num=N)

        plt.plot(X, data,label = f"{self.name}",**kwargs)
        plt.xlabel("time (s)")
        plt.ylabel(f"Amplitude ({unit})")
        plt.title(f"{self.name}"+add_to_title)

    # def plot_FFT(self,      color=None,
    #                         label=None,
    #                         add_50Hz=False,
    #                         add_freqs : list[float]|None = None,
    #                         new_figure = True,
    #                         with_0_Hz = False,
    #                         add_3dB = False):
    #     '''
    #     Args 
    #     ------
    #     `color` = The color used to draw the signal

    #     `label` = Not used for the moment

    #     `add_50Hz` = Draw 50 Hz and harmonics 
        
    #     `add_freqs` = Liste de fréquence à ajouter au graphique
        
    #     `new_figure` = Crée une nouvelle figure avant l'affichage
        
    #     `with_0Hz` = if False, it subtract the mean of the signal to the signal itself, before doing the FFT
    #     '''
    #     if new_figure : plt.figure()
    #     if add_freqs is None : add_freqs = []
        
    #     self.data
    #     self.fs

    #     Te = 1/self.fs                      # periode d'echantillonage
    #     N = len(self.data)                  # Nombre de point de données

    #     if with_0_Hz : yf = fft(self.data)[:N//2]          # Pour ne garder que la moitié
    #     else : yf = fft(self.data-self.data.mean())[:N//2]         
    #     yf = abs(yf)                        # Passage au module
    #     yf = yf/max(yf)                     # Normalisation
    #     yf_db = 20*np.log10(yf)             # Convertion en dB
    #     yf_db = [-100 if e==-np.inf else e for e in yf_db]
    #     xf = fftfreq(N, Te)[:N//2]          # Génération de l'echelle des fréquences

    #     if add_50Hz :
    #         v_min = min(yf_db)
    #         plt.plot([50,50,100,100,150,150,200,200],[v_min,-3,-3,v_min]*2,linestyle='-',color='orange',label='50Hz et harmoniques')

    #     min_db = np.min(yf_db)
    #     for freq in add_freqs : 
    #         print(f"min_db = {min_db}")
    #         plt.plot([freq,freq],[min_db,0],'--',color='#f82',label=f"{freq} Hz")
    #         plt.annotate(f" {freq} Hz",(freq,-70),color='#f82')

    #     plt.plot(xf,yf_db,label=f"FFT de {self.name}",color=color)
    #     if add_3dB : plt.plot([0,self.fs/2],[-3,-3],'--r',label="-3 dB")
    #     plt.grid()
    #     plt.legend(loc="lower left")
    #     plt.title(f"FFT du signal {self.name}")
    #     plt.ylabel('Amplitude normalisé (dB)')
    #     plt.xlabel('fréquence (Hz)')


    # def study_domain(self,start,end):
    #     """Cette fonction ajoute un rectangle autour du domaine d'étude choisi sur un graphique pré-existant
        
    #     args
    #     ----
    #     `start` = temps de début du domaine d'étude (en s)
        
    #     `end`   = temps de fin du domaine d'étude (en s)"""
    #     etude = (start,end)
    #     M,m = self.max_min(250000,-250000) ; A  = M-m
    #     max_e,min_e = self.re_cut(start,end).max_min(250000,-250000)
    #     max_e += A/10 ; min_e -= A/10
    #     x_etude = [etude[0],etude[0],etude[1],etude[1],etude[0]]
    #     y_etude = [min_e,max_e,max_e,min_e,min_e]
    #     self.plot()
    #     plt.plot(x_etude,y_etude,'--',color='black',label="domaine d'étude")
    #     plt.legend()



class Filter :    
    def FIR(            signal : C_signal | list[C_signal],
                        order : int,
                        cutoff : float | list[float],
                        window : str = 'hamming',
                        type_filtre : Literal['low_pass','high_pass'] = 'low_pass') -> C_signal | list[C_signal]:
        if order == 0:
            return signal

        if type(signal) == C_signal : 
            if type_filtre == 'low_pass':
                pass_zero = True 
            elif type_filtre == 'high_pass':
                pass_zero = False


            num_coef = firwin(
                numtaps = order+1,
                cutoff = cutoff,
                window = window,
                pass_zero = pass_zero,
                fs = signal.fs)
            y = filtfilt(num_coef, 1, signal.data)
            signal_f = C_signal(y, signal.fs, signal.unit, f"{signal.name} - ({window},{order},{cutoff}Hz)")
            return signal_f
        
        res = []
        for sig in signal :
            sig_f = Filter.FIR(
                signal = sig,
                order = order,
                cutoff = cutoff,
                window = window,
                type_filtre = type_filtre)

            res.append(sig_f)
        return res




class f_sig :
    f_I1, f_I2, f_I3 = None, None, None
    init_ok = False
    def init_f_current():
        I_mes_1 = np.array([0,19.70, 49.42, 121.86, 163.97, 334.00, 785.80, 1537.30, 2013.60]) * 1e-6
        I_mes_2 = np.array([0,26.84, 50.10, 100.89, 165.22, 335.85, 788.40, 1536.00, 2015.50]) * 1e-6
        I_mes_3 = np.array([0,18.67, 45.90, 116.30, 163.44, 335.26, 784.50, 1526.10, 1968.50]) * 1e-6
        V_ADC_1 = np.array([0,19.02, 33.83, 70.00,  91.10,  176.50, 401.80, 776.40,  1015.00]) * 1e-3
        V_ADC_2 = np.array([0,22.30, 34.47, 59.83,  92.00,  177.30, 403.35, 777.10,  1019.26]) * 1e-3
        V_ADC_3 = np.array([0,18.32, 32.10, 67.00,  91.10,  177.19, 401.48, 772.35,  990.93 ]) * 1e-3

        f_sig.f_I1 = interp1d(V_ADC_1,I_mes_1,kind="quadratic")
        f_sig.f_I2 = interp1d(V_ADC_2,I_mes_2,kind="quadratic")
        f_sig.f_I3 = interp1d(V_ADC_3,I_mes_3,kind="quadratic")
        f_sig.init_ok = True


    def f_current(sig_V : C_signal,num : int):
        '''
        Cette fonction convertie la tension lue par l'ADC en courant traversant le fil

        Args : 
        ------
        `sig_V` (en Volts)
        `num` = le numéro du générateur utilisé
        Returns :
        ---------
        `i` (en Ampères)
        '''
        if not f_sig.init_ok : f_sig.init_f_current()
        if sig_V.unit not in ['V','v','Volts','volts'] : 
            print("Attention l'unité de sig_V.unit est peut-être fausse") 

        sig_I = C_signal(data=None, fs=sig_V.fs, unit='A', name=f"U_I{num} converted to I{num}")
        match num :
            case 1 :
                sig_I.data = f_sig.f_I1(sig_V.data)
            case 2 :
                sig_I.data = f_sig.f_I2(sig_V.data)
            case 3 :
                sig_I.data = f_sig.f_I3(sig_V.data)
            case _ :
                warn("Le numéro du générateur de courant doit être 1,2 ou 3")
        return sig_I
        





def bar_graph(l_y,l_label,l_ticks,l_color=None,close=True):
        if close : plt.close('all')
        N=len(l_y)
        N2=len(l_y[0])
        if l_color is None: l_color=[None]*N
        barWidth = 1/(N+1)

        for i,(y,label,color) in enumerate(zip(l_y,l_label,l_color)) :
                x = np.arange(N2) + i*barWidth
                plt.bar(x,y,label=label,width=barWidth,color=color,edgecolor='black')

        plt.xticks([n + barWidth*(N-1)/2 for n in range(N2)],l_ticks)
        plt.legend()





