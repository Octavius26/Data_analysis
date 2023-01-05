

# To import this file :
# --------------------------


# import sys

# my_path = r"C:\Users\alomb\OneDrive\new_racine\Documents\_Pro\CODE\Data_analysis"
# if my_path not in sys.path : sys.path.append(my_path); print("path added")

# from f_analyse import * 





from warnings import warn
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

import scipy.fft as spf


from scipy.fft import fft, fftfreq
from scipy.signal import filtfilt,firwin
from scipy.interpolate import interp1d 



class C_Signal :
    """
    Who return an amplitude value
    - `mean`
    - `std`
    - `max`
    - `min`
    - `val_at_nearset_t`

    Who return a time value
    - `t_at_max`
    - `t_at_min`
    - `duration`
    - `t_max`
    - `t_min`
    
    Who plot something
    - `plot`

    - `plot_ADD_t_at_max`
    - `plot_ADD_t_at_min`
    - `plot_ADD_t`  =========================== to implement
    - `plot_ADD_times` ======================== to delete
    - `plot_ADD_box_on_recut`
    - `plot_ADD_mean`
    - `plot_ADD_val` ========================== to implement
    - `plot_ADD_values` ======================= to delete

    Who return a signal
    - `recut`

    - `copy`
    - `empty_copy`
    
    Others
    - `index_at`
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
        `t0` : The beginning of the signal
        """

        if data is None : self.data = None
        elif type(data)==np.ndarray : self.data = data 
        elif type(data)==list : self.data = np.array(data)
        else : warn("Wrong data type")

        self.fs = fs
        self.unit = unit
        self.name = name
        self.t0 = t0
        if self.data is not None : self.N = len(self.data)

    def mean(self)->float : return np.mean(self.data)
    def std(self)-> float : return np.std(self.data)
    def max(self)-> float : return np.max(self.data)
    def min(self)-> float : return np.min(self.data)
    def val_at_nearest_t(self,t: float)-> float : return self.data[int(t*self.fs)]

    def t_at_max(self)-> float : return np.argmax(self.data) / self.fs
    def t_at_min(self)-> float : return np.argmin(self.data) / self.fs
    def duration(self)-> float : return self.N/self.fs
    def t_max(self)-> float : return self.t0 + self.duration()
    def t_min(self)-> float : return self.t0

    @property
    def N(self): return len(self.data)

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
        X=np.linspace(start=self.t_min() , stop=self.t_max(), num=N)
        plt.plot(X, data,label = f"{self.name}",**kwargs)
        plt.xlabel("time (s)")
        if unit is None : plt.ylabel('Amplitude')
        else : plt.ylabel(f"Amplitude ({unit})")
        plt.title(f"{self.name} {add_to_title}")

    def plot_ADD_t_at_max(self,**kwargs): 
        self.plot_ADD_times([self.t_at_max()],**kwargs)
        
    def plot_ADD_t_at_min(self,**kwargs): 
        self.plot_ADD_times([self.t_at_min()],**kwargs)

    def plot_ADD_box_on_recut(self,
                                t1: float=None,
                                t2: float=None,
                                extra_margin_y: float=0,
                                extra_margin_x: float=0,
                                **kwargs)->None: 
        """
        Add a box on a previous plot

        Args 
        -----
        `t1`: float (default = signal's start time)
            Start time of the box
        `t2`: float (default = signal's end time)
            End time of the box
        `extra_margin_x` : float (default = 0)
        `extra_margin_y` : float (default = 0)
        """                               
        if t1 is None : t1 = self.t_min
        if t2 is None : t2 = self.t_max
        yh = self.recut(t1,t2).max()
        yl = self.recut(t1,t2).min()
        e = yh-yl
        yh += e * 0.1 + extra_margin_y # to set the upper margin
        yl -= e * 0.1 + extra_margin_y# to set the lower margin
        t1 -= extra_margin_x
        t2 -= extra_margin_x
        X = [t1,t2,t2,t1,t1]
        Y = [yh,yh,yl,yl,yh]
        plt.plot(X,Y,'--',**kwargs)

    def plot_ADD_mean(self,**kwargs):
        X = [self.t_max(), self.t_min()]
        Y = [self.mean()]*2
        plt.plot(X,Y,'--',label=f"mean = {self.mean()}")

    def plot_ADD_times(self, l_times:list|np.ndarray, **kwargs):
        yh = self.max()
        yl = self.min()
        e = yh-yl
        yh += e * 0.1 # to set the upper margin
        yl -= e * 0.1 # to set the lower margin
        
        for time in l_times :
            X = [time]*2
            Y = [yl,yh]
            plt.plot(X,Y,'--',label=f"t={time}s",**kwargs)

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
        return int((t-self.t0)*self.fs)

    def recut(self,t1: float=0,t2: float=None) :
        """
        Args
        -----
        `t1` : float
            Beginning of the recuted signal
        `t2` : float (optional)
            End of the recuted signal, use None to keep the signal until the end
        """
        if t1 < self.t_min : print("Warning : t1 < t_min , we used t1 = t_min istead"); t1 = self.t_min
        if self.t_max < t2 : print("Warning : t_max < t2 , we used t2 = t_max istead"); t2 = self.t_max

        if t2 is None : t2 = self.t_max
        i1,i2 = self.index_at(t1), self.index_at(t2)
        rep = self.empty_copy()
        rep.data = self.data[i1:i2]
        rep.name += ' recuted'
        return rep

    def empty_copy(self):
        """Copy everything exept data (including t0)"""
        return T_Signal(data = None,
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name,
                        t0 = self.t_min)

    def copy(self):
        return T_Signal(data = self.data.copy(),
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name,
                        t0 = self.t_min)



class T_signal:
    """
    Who return an amplitude value
    - `mean`
    - `std`
    - `max`
    - `min`
    - `val_at_nearset_t`

    Who return a time value
    - `t_at_max`
    - `t_at_min`
    - `duration`
    - `t_max`
    - `t_min`
    
    Who plot something
    - `plot`

    - `plot_ADD_t_at_max`
    - `plot_ADD_t_at_min`
    - `plot_ADD_t`  =========================== to implement
    - `plot_ADD_times` ======================== to delete
    - `plot_ADD_box_on_recut`
    - `plot_ADD_mean`
    - `plot_ADD_val` ========================== to implement
    - `plot_ADD_values` ======================= to delete

    Who return a signal
    - `recut`

    - `copy`
    - `empty_copy`
    
    Others
    - `index_at`
    """

    def __init__(   self,
                    data : np.ndarray | list = None,
                    fs : float = 1,
                    unit : str = '',
                    name : str = '',
                    t0 : float = 0): #TODO make it fonctional ?
        
        self.SIG = C_Signal(data=data,
                            fs=fs,
                            unit=unit,
                            name=name,
                            t0=t0)

        self.recut = self.SIG.recut
        self.mean = self.SIG.mean
        self.std = self.SIG.std
        self.max = self.SIG.max
        self.min = self.SIG.min
        self.val_at_nearest_t = self.SIG.val_at_nearest_t
        self.t_at_max = self.SIG.t_at_max
        self.t_at_min = self.SIG.t_at_min
        self.duration = self.SIG.duration
        self.t_max = self.SIG.t_max
        self.t_min = self.SIG.t_min
        self.N = self.SIG.N
        self.plot_ADD_t_at_max = self.SIG.plot_ADD_t_at_max
        self.plot_ADD_t_at_min = self.SIG.plot_ADD_t_at_min
        self.fft = self.SIG.fft
        self.plot_ADD_t_at_max = self.SIG.plot_ADD_t_at_max 
        self.plot_ADD_t_at_min = self.SIG.plot_ADD_t_at_min
        self.plot_ADD_box_on_recut = self.SIG.plot_ADD_box_on_recut
        self.plot_ADD_times = self.SIG.plot_ADD_times
        self.plot_ADD_time = self.SIG.plot_ADD_time # TO implement
        self.plot_ADD_values = self.SIG.plot_ADD_values
        self.plot_ADD_mean = self.SIG.mean
        self.index_at = self.SIG.index_at
        self.recut = self.SIG.recut
        self.empty_copy = self.SIG.empty_copy
        self.copy = self.SIG.copy
        self.plot = self.SIG.plot
        









class F_signal(T_Signal):
    """
    Methods
    -------
        
    Who give a float :
    - `std()`
    - `mean()`
    - `max()`
    - `min()`
    - `val_at_nearest_t` => `val_at_nearest_f`
    - `t_at_max` => `f_at_max`
    - `t_at_min` => `f_at_min`
    - `duration` => `f_range`

    Who plot something :
    - `plot()`
    - `plot_ADD_box_on_recut` #TODO redifne plot_ADD_box_on_recut
    - `plot_ADD_freqs`
    - `plot_ADD_values`

    Who return a T_signal :
    - `recut()`
    - `copy()`
    - `empty_copy()`
    """
    def __init__(self, data: np.ndarray | list = None, fs: float = 1, unit: str = '', name: str = '',f0 = 0):
        """
        Args :
        ------
        `data` : Array or list of values
        `fs` : << Sampling frequency >> calculated by 1/(f1-f0)
        `unit` : unit of the signal
        `name` : name of the signal
        """
        super().__init__(data, fs, unit, name, f0)

        # def f_at_max(self,**kwrags): return self.t_at_max(**kwrags)
        self.f_at_max = self.t_at_max
        self.f_at_min = self.t_at_min
        self.val_at_nearest_f = self.val_at_nearest_t
        self.f_range = self.duration
    
    @property
    def f_min(self)->float: return self.t_min
    @property
    def f_max(self)->float: return self.t_max

    def empty_copy(self):
        """Copy everything exept data (including f0)"""
        return F_signal(data = None,
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name,
                        f0 = self.f_min)

    def copy(self):
        return F_signal(data = self.data.copy(),
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name,
                        f0 = self.f_min)

    def plot(self, new_unit: tuple[str, float] = None, add_to_title='', new_figure=True, **kwargs):
        super().plot(new_unit, add_to_title, new_figure, **kwargs)
        plt.xlabel("Frequency (Hz)")
        
    def plot_ADD_freqs(self, l_freqs: list | np.ndarray, **kwargs):
        yh = self.max()
        yl = self.min()
        e = yh-yl
        yh += e * 0.1 # to set the upper margin
        yl -= e * 0.1 # to set the lower margin
        
        for time in l_freqs :
            X = [time]*2
            Y = [yl,yh]
            plt.plot(X,Y,'--',label=f"f={time} Hz",**kwargs)

    # def plot_ADD_f_at_max(self,**kwargs):
    #     return super().plot_ADD_t_at_max(**kwargs)

    # def plot_ADD_f_at_min(self,**kwargs):
    #     return super().plot_ADD_t_at_min(**kwargs)


    def plot_ADD_f_at_max(self,**kwargs): 
        self.plot_ADD_freqs([self.f_at_max()],**kwargs)
        
    def plot_ADD_f_at_min(self,**kwargs): 
        self.plot_ADD_freqs([self.f_at_min()],**kwargs)






class FFT_modul(F_signal):
    def __init__(self, data: np.ndarray | list = None, fs: float = 1, unit: str = '', name: str = '', f0=0):
        super().__init__(data, fs, unit, name, f0)

    def plot_ADD_moduls(self, l_modul: list[float], l_labels: list[str] = None, l_colors: list[str] = None, **kwargs):
        return super().plot_ADD_values(l_modul, l_labels, l_colors, **kwargs)






class FFT_phase(F_signal):
    def __init__(self, data: np.ndarray | list = None, fs: float = 1, unit: str = '', name: str = '', f0=0):
        super().__init__(data, fs, unit, name, f0)

    def plot_ADD_phases(self, l_phases: list[float], l_labels: list[str] = None, l_colors: list[str] = None, **kwargs):
        return super().plot_ADD_values(l_phases, l_labels, l_colors, **kwargs)


















class FFT_signal :
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
                        name: str = '',
                        f0: float = 0,
                        nb_zero: int = 0,
                        window: np.ndarray | list = None ):
        """
        Args :
        ------
        `data` : Array or list of complexes values
        `fs` : Sampling frequency (Hz)  
        `unit` : unit of the signal
        `name` : name of the signal
        `f0` : beginning of the signal
        `nb_zero` : the number of zeros added for 0 padding
        `window` : the appodization window used #TODO exploiter la fenêtre d'appodisation 
        """
        if data is None : self.data = None
        elif type(data)==np.ndarray : self.data = data 
        elif type(data)==list : self.data = np.array(data)
        else : warn("Wrong data type")

        self.fs = fs
        self.unit = unit
        self.name = name
        self.f0 = f0
        self.nb_zero = nb_zero
        self.window = window
        if self.data is not None : self.N = len(self.data)

        self.__compute_modul_phase_from_data()
        self.__init_functions()
    
    def __compute_modul_phase_from_data(self):
            
        # FFT avec 0 padding sur n
        # yf = np.fft.fft(self.data,n)
        # xf = np.linspace(0,self.fs/2,n//2-1)
        # xf = np.fft.fftfreq(n,d=1/fs)[:n//2-1]

        N = self.N
        n =  (len(self.data)-1)*2 if N%2 != 0 else len(self.data)*2-1
        yf_modul = 2/(n-self.nb_zero) * np.abs(self.data)
        # yf_modul = 2/np.size(self.data)*np.abs(self.data[0:n//2-1])
        # multiplication par 2 car puissances divisée entre freqs positifs et négatifs
        # division par la taille de max_spectro pour prendre en compte le pas 
        yf_phase = np.angle(self.data)

        self.Modul = FFT_modul( data = yf_modul,
                                fs = self.fs,
                                unit = '?',
                                name = f"{self.name} (FFT_modul)",
                                f0 = self.f0)

        self.Phase = FFT_phase( data = yf_phase,
                                fs = self.fs,
                                unit = '?',
                                name = f"{self.name} (FFT_phase)",
                                f0 = self.f0)

    def __init_functions(self):
        self.plot_modul = self.Modul.plot
        self.plot_ADDmodul_box_on_recut = self.Modul.plot_ADD_box_on_recut
        self.plot_ADDmodul_freqs = self.Modul.plot_ADD_freqs
        self.plot_ADDmodul_moduls = self.Modul.plot_ADD_moduls
        self.plot_ADDmodul_f_at_max = self.Modul.plot_ADD_f_at_max
        self.plot_ADDmodul_f_at_min = self.Modul.plot_ADD_f_at_min

        self.plot_phase = self.Phase.plot
        self.plot_ADDphase_box_on_recut = self.Phase.plot_ADD_box_on_recut
        self.plot_ADDphase_freqs = self.Phase.plot_ADD_freqs
        self.plot_ADDphase_phases = self.Phase.plot_ADD_phases
        self.plot_ADDphase_f_at_max = self.Phase.plot_ADD_f_at_max
        self.plot_ADDphase_f_at_min = self.Phase.plot_ADD_f_at_min
            
        self.modul_at_nearest_f = self.Modul.val_at_nearest_f
        self.modul_max = self.Modul.max
        self.modul_min = self.Modul.min
        self.modul_f_at_max = self.Modul.f_at_max
        self.modul_f_at_min = self.Modul.f_at_min
        # self.modul_f_at = self.Modul.f_at   # TODO To define modul_f_at

        self.phase_at_nearest_f = self.Phase.val_at_nearest_f
        self.phase_max = self.Phase.max
        self.phase_min = self.Phase.min
        self.phase_f_at_max = self.Phase.f_at_max
        self.phase_f_at_min = self.Phase.f_at_min
        # self.phase_f_at = self.Phase.f_at   # TODO To define phase_f_at



    def f_range(self)-> float : return self.N/self.fs

    def plot(self,new_figure=True,figsize : tuple[int] =None,hspace=1,**kwargs):
        """
        Args :
        ------
        `new_figure` : (default = True)
        `figsize` : (default = None) 
        `hspace` : (default = 1)
            space between the two subplot
        """
        plt.figure(figsize=figsize)
        plt.subplot(211)
        self.Modul.plot(new_figure=False,**kwargs)
        plt.subplot(212)
        self.Phase.plot(new_figure=False,**kwargs)
        plt.subplots_adjust(hspace=hspace)

    def plot_ADD_freqs(self):pass
    def plot_ADD_box_on_recut(self):pass

    def copy(self):
        return FFT_signal(  data = self.data.copy(),
                            fs = self.fs,
                            unit = self.unit,
                            name = self.name,
                            f0 = self.f0)

    def empty_copy(self): 
        return FFT_signal(  data = None,
                            fs = self.fs,
                            unit = self.unit,
                            name = self.name,
                            f0 = self.f0)













































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
