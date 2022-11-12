

# To import this file :
# --------------------------


# import sys

# my_path = r"C:\Users\alomb\OneDrive\new_racine\Documents\_Pro\CODE\Data_analysis"
# if my_path not in sys.path : sys.path.append(my_path); print("path added")

# from f_analyse import * 





from warnings import warn
from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import filtfilt,firwin
from scipy.interpolate import interp1d 



class T_Signal :
    """
    Methods
    -------
        
    Who give a float :
    - `std`
    - `max`
    - `min`
    - `val_at_nearest_t(t:)`
    - `t_at_max`
    - `t_at_min`
    - `duration`

    Who plot something :
    - `plot`
    - `plot_ADD_box_on_recut`
    - `plot_ADD_times`
    - `plot_ADD_values`

    Who return a signal :
    - `recut`
    - `copy`
    - `empty_copy`

    - `fft`
    """
    

    def __init__(           self,
                            data : np.ndarray | list = None,
                            fs : float = 1,
                            unit : str = '',
                            name : str = '',
                            t0 : float = 0): #TODO make it fonctional
        """
        Args :
        ------
        `data` : Array or list of values
        `fs` : Sampling frequency (Hz)  
        `unit` : unit of the signal
        `name` : name of the signal
        """

        if data is None : self.data = None
        elif type(data)==np.ndarray : self.data = data 
        elif type(data)==list : self.data = np.array(data)
        else : warn("Wrong data type")

        self.fs = fs
        self.unit = unit
        self.name = name
        self.t0 = t0

    def mean(self)->float : return np.mean(self.data)
    def std(self)-> float : return np.std(self.data)
    def max(self)-> float : return np.max(self.data)
    def min(self)-> float : return np.min(self.data)
    def val_at_nearest_t(self,t: float)-> float : return self.data[int(t*self.fs)]
    def t_at_max(self)-> float : return np.argmax(self.data) / self.fs
    def t_at_min(self)-> float : return np.argmin(self.data) / self.fs
    def duration(self)-> float : return len(self.data)/self.fs
    def t_max(self)-> float : return self.t0 + self.duration()
    def t_min(self)-> float : return self.t0



    def plot_fft(self,  n: int=None, 
                        n_factor: float=None,
                        choose_next_power_of_2 = True,
                        print_choices = False,
                        dB_mode = False,
                        **kwargs):
        """
        Args 
        -----
        `n`: int (optional)
            Size of the signal after zero padding. It's need `n`>= len(data)
        `n_factor` : float (optional)
            If not None, `n`=len(data)*`n_factor`
        `choose_next_power_of_2` : bool
            If true use the newxt power of 2 istead of `n`
        `print_choices` : bool
        `dB_mode` : bool
            Impact the output signal

        Return
        ------
        `xf` : Array
        `yf_real` : Array
            It is impacted by `dB_mode`
        """
        # TODO only retrun the fft signal, and make the plot throug it
        # TODO use apodization windows (hamming,...)

        if n is None : n = len(self.data)
        if n_factor is not None : n= len(self.data) * n_factor

        if n < len(self.data):
            print("n is smaller than self.data, we use n=len(self.data) instead")
            n=len(self.data)

        if choose_next_power_of_2 :
            n = int(2**np.ceil(np.log(n) / np.log(2)))

        if print_choices :
            print(f"FFT : n={n}, len(data)={len(self.data)}")

        # FFT avec 0 padding sur N
        plt.figure()
        yf = np.fft.fft(self.data,n)
        xf = np.linspace(0,self.fs/2,n//2-1)
        # xf = np.fft.fftfreq(n,d=1/fs)[:n//2-1]
        yf_real = 2/np.size(self.data)*np.abs(yf[0:n//2-1])
        if dB_mode : yf_real = 20*np.log10(yf_real)
        # multiplication par 2 car puissances divisée entre freqs positifs et négatifs
        # division par la taille de max_spectro pour prendre en compte le pas 
        plt.plot(xf,yf_real)
        plt.title(self.name + " FFT")
        if dB_mode : 
            plt.ylabel("Amplitude(dB)")
            plt.xlabel("frequency(Hz)")

        # TODO make C_signal compatible with fft signal ?
        # fft_sig = self.empty_copy()
        # fft_sig.name += ' fft'
        # fft_sig.data = yf_real
        # fft_sig.fs = 1/(xf[1]-xf[0])
        # fft_sig.unit = 'dB' if dB_mode else 'FFT_unit'

        return xf, yf_real




    def plot_ADD_box_on_recut(self,t1: float,t2: float,**kwargs) : 
        yh = self.recut(t1,t2).max()
        yl = self.recut(t1,t2).min()
        e = yh-yl
        yh += e * 0.1 # to set the upper margin
        yl -= e * 0.1 # to set the lower margin
        X = [t1,t2,t2,t1,t1]
        Y = [yh,yh,yl,yl,yh]
        plt.plot(X,Y,'--',**kwargs)


    def plot_ADD_mean(self,**kwargs):
        X = [self.t_max(), self.t_min()]
        Y = [self.mean()]*2
        plt.plot(X,Y,'--')

    def plot_ADD_times(self, times:list|np.ndarray, **kwargs):
        yh = self.max()
        yl = self.min()
        e = yh-yl
        yh += e * 0.1 # to set the upper margin
        yl -= e * 0.1 # to set the lower margin
        
        for time in times :
            X = [time]*2
            Y = [yl,yh]
            plt.plot(X,Y,'--',label=f"t={times}s",**kwargs)

    def plot_ADD_values(self,   l_values:list[float],
                                l_labels:list[str]=None,
                                l_colors:list[str]=None,
                                **kwargs):
        X = [self.t_max(), self.t_min()]
        if l_labels is None : l_labels = [None]*len(l_values)
        if l_colors is None : l_colors = [None]*len(l_values)
        
        for value,label,color in zip(l_values,l_labels,l_colors):
            Y = [value]*2
            plt.plot(X,Y,'--',label=label,color=color,**kwargs)




    def index_at(self,t: float):
        """
        Args
        ------
        `t` : flaot
            Time for which the nearest index is caluclated
        """
        return int(t*self.fs)

    def recut(self,t1: float=0,t2: float=None) :
        """
        Args
        -----
        `t1` : float
            Beginning of the recuted signal
        `t2` : float (optional)
            End of the recuted signal, use None to keep the signal until the end
        """
        if t2 is None : t2 = self.duration()
        i1,i2 = self.index_at(t1), self.index_at(t2)
        rep = self.empty_copy()
        rep.data = self.data[i1:i2]
        rep.name += ' recuted'
        return rep

    def empty_copy(self):
        """Copy everything exept data"""
        return T_Signal(data = None,
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name)

    def copy(self):
        return T_Signal(data = self.data.copy(),
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name)

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
        if unit is None : plt.ylabel('Amplitude')
        else : plt.ylabel(f"Amplitude ({unit})")
        plt.title(f"{self.name}"+add_to_title)

    def plot_FFT_depreceated(self,      color=None, 
                            label=None,
                            add_50Hz=False,
                            add_freqs : list[float]|None = None,
                            new_figure = True,
                            with_0_Hz = False,
                            add_3dB = False):
        '''
        Args 
        ------
        `color` = The color used to draw the signal

        `label` = Not used for the moment

        `add_50Hz` = Draw 50 Hz and harmonics 
        
        `add_freqs` = Liste de fréquence à ajouter au graphique
        
        `new_figure` = Crée une nouvelle figure avant l'affichage
        
        `with_0Hz` = if False, it subtract the mean of the signal to the signal itself, before doing the FFT
        '''
        if new_figure : plt.figure()
        if add_freqs is None : add_freqs = []
        
        self.data
        self.fs

        Te = 1/self.fs                      # periode d'echantillonage
        N = len(self.data)                  # Nombre de point de données

        if with_0_Hz : yf = fft(self.data)[:N//2]          # Pour ne garder que la moitié
        else : yf = fft(self.data-self.data.mean())[:N//2]         
        yf = abs(yf)                        # Passage au module
        yf = yf/max(yf)                     # Normalisation
        yf_db = 20*np.log10(yf)             # Convertion en dB
        yf_db = [-100 if e==-np.inf else e for e in yf_db]
        xf = fftfreq(N, Te)[:N//2]          # Génération de l'echelle des fréquences

        if add_50Hz :
            v_min = min(yf_db)
            plt.plot([50,50,100,100,150,150,200,200],[v_min,-3,-3,v_min]*2,linestyle='-',color='orange',label='50Hz et harmoniques')

        min_db = np.min(yf_db)
        for freq in add_freqs : 
            print(f"min_db = {min_db}")
            plt.plot([freq,freq],[min_db,0],'--',color='#f82',label=f"{freq} Hz")
            plt.annotate(f" {freq} Hz",(freq,-70),color='#f82')

        plt.plot(xf,yf_db,label=f"FFT de {self.name}",color=color)
        if add_3dB : plt.plot([0,self.fs/2],[-3,-3],'--r',label="-3 dB")
        plt.grid()
        plt.legend(loc="lower left")
        plt.title(f"FFT du signal {self.name}")
        plt.ylabel('Amplitude normalisé (dB)')
        plt.xlabel('fréquence (Hz)')




class F_signal(T_Signal):
    """
    Methods
    -------
        
    Who give a float :
    o- `std()`
    o- `mean()`
    o- `max()`
    o- `min()`
    o- `val_at_nearest_t` => `val_at_nearest_f`
    o- `t_at_max` => `f_at_max`
    o- `t_at_min` => `f_at_min`
    o- `duration` => `f_range`

    Who plot something :
    o- `plot()`
    o- `plot_ADD_box_on_recut`
    - `plot_ADD_freqs`
    - `plot_ADD_values`
    - ``

    Who return a T_signal :
    - `recut()`
    - `copy()`
    """
    def __init__(self, data: np.ndarray | list = None, fs: float = 1, unit: str = '', name: str = ''):
        """
        Args :
        ------
        `data` : Array or list of values
        `fs` : << Sampling frequency >> calculated by 1/(f1-f0)
        `unit` : unit of the signal
        `name` : name of the signal
        """
        super().__init__(data, fs, unit, name)

        # def f_at_max(self,**kwrags): return self.t_at_max(**kwrags)
        self.f_at_max = self.t_at_max
        self.f_at_min = self.t_at_min
        self.val_at_nearest_f = self.val_at_nearest_t
        self.f_range = self.duration

    def copy(self): pass

    def plot(self, new_unit: tuple[str, float] = None, add_to_title='', new_figure=True, **kwargs):
        super().plot(new_unit, add_to_title, new_figure, **kwargs)
        plt.xlabel("Frequency (Hz)")
        




class FFT_full_sig :
    """
    Methods
    -------
        
    Who give a float :
    - Modulus functions
        - `modul_at_nearest_f`
        - `modul_max`
        - `modul_min`
        - `modul_f_at_max`
        - `modul_f_at_min`
        - `modul_f_at` 
    
    - Phases functions
        - `phase_at_nearest_f`
        - `phase_max`
        - `phase_min`
        - `phase_f_at_max`
        - `phase_f_at_min`
        - `phase_f_at`
    - ``

    Who return an Array :
    - `modul_f_at`
    - `phase_f_at`
    

    Who plot something :
    - `plot`
    - `plot_modul`
    - `plot_phase`
    - On plot
        - `plot_ADD_freqs`
        - `plot_ADD_box_on_recut`

    - On plot_modul
        - `plot_ADDmodul_box_on_recut`
        - `plot_ADDmodul_freqs`
        - `plot_ADDmodul_moduls`
    - On plot_phase
        - `plot_ADDphase_box_on_recut`
        - `plot_ADDphase_freqs`
        - `plot_ADDphase_phases`

    Who return a T_signal :
    - `ifft`
    - ``
    """

    def __init__(self,  data: np.ndarray | list = None, 
                        fs: float = 1, 
                        unit: str = '', 
                        name: str = ''):
        """
        Args :
        ------
        `data` : Array or list of complexes values
        `fs` : Sampling frequency (Hz)  
        `unit` : unit of the signal
        `name` : name of the signal
        """
        if data is None : self.data = None
        elif type(data)==np.ndarray : self.data = data 
        elif type(data)==list : self.data = np.array(data)
        else : warn("Wrong data type")

        self.fs = fs
        self.unit = unit
        self.name = name

    def plot(self): pass
    def plot_ADD_freqs(self):pass
    def plot_ADD_box_on_recut(self):pass

    def plot_modul(self): pass
    def plot_ADDmodul_box_on_recut(self):pass
    def plot_ADDmodul_freqs(self):pass
    def plot_ADDmodul_moduls(self):pass
    
    def plot_phase(self): pass
    def plot_ADDphase_box_on_recut(self):pass
    def plot_ADDphase_freqs(self):pass
    def plot_ADDphase_phases(self):pass

    def modul_at_nearest_f(self,f):pass
    def modul_max(self):pass
    def modul_min(self):pass
    def modul_f_at_max(self):pass
    def modul_f_at_min(self):pass
    def modul_f_at(self):pass
    
    def phase_at_nearest_f(self,f):pass
    def phase_max(self):pass
    def phase_min(self):pass
    def phase_f_at_max(self):pass
    def phase_f_at_min(self):pass
    def phase_f_at(self):pass
    
    def f_range(self):pass
    def ifft(self): pass
    def copy(self): pass
    def empty_copy(self): pass

























































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


    def f_current(sig_V : T_Signal,num : int):
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

        sig_I = T_Signal(data=None, fs=sig_V.fs, unit='A', name=f"U_I{num} converted to I{num}")
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






class Filter :    
    def FIR(            signal : T_Signal | list[T_Signal],
                        order : int,
                        cutoff : float | list[float],
                        window : str = 'hamming',
                        type_filtre : Literal['low_pass','high_pass'] = 'low_pass') -> T_Signal | list[T_Signal]:
        if order == 0:
            return signal

        if type(signal) == T_Signal : 
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
            signal_f = T_Signal(y, signal.fs, signal.unit, f"{signal.name} - ({window},{order},{cutoff}Hz)")
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
