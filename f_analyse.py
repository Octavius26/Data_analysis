

# To import this file :
# --------------------------


# import sys

# my_path = r"C:\Users\alomb\OneDrive\new_racine\Documents\_Pro\CODE\Data_analysis"
# if my_path not in sys.path : sys.path.append(my_path); print("path added")

# from f_analyse import * 


from __future__ import annotations


from warnings import warn
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

import scipy.fft as spf


from scipy.fft import fft, fftfreq
from scipy.signal import filtfilt,firwin
from scipy.interpolate import interp1d 



class C_signal :
    """
    Who return an amplitude value
    - `mean`
    - `std`
    - `max`
    - `min`
    - `val_at_nearset_t`

    Who return a time value
    - `t_at_max`
        Ne rend qu'une valeur (la première rencontrée ?) #TODO check it
    - `t_at_min`
        Ne rend qu'une valeur (la première rencontrée ?) #TODO check it
    - `duration`
    - `t_max`
    - `t_min`
    
    - `t_at_val` ========= #TODO to implement
        - rend un tableau des valeurs de t telle que f(t)=val
        - comme la fonction est discrète, on peut ne pas tomber sur la valeur exacte
            - utiliser une interpolation linéaire ?
            - rendre toutes les valeurs à avec statifaisant à peu près la condition ?
    
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

    - `fft`
    
    Others
    - `index_at`
    - `N`
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
    def N(self): return 0 if self.data is None else len(self.data)
        
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

        
    def plot_ADD_t_at_min(self,**kwargs): 
        self.plot_ADD_times([self.t_at_min()],**kwargs)

    def plot_ADD_box(self,
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


    def plot_ADD_val(self,val,**kwargs):
        X = [self.t_max(), self.t_min()]
        Y = [val,val]
        plt.plot(X,Y,'--',label=f"{val} {self.unit}",**kwargs)


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
        if t2 is None : t2 = self.t_max()

        if t1 < self.t_min() : print("Warning : t1 < t_min , we used t1 = t_min istead"); t1 = self.t_min
        if self.t_max() < t2 : print("Warning : t_max < t2 , we used t2 = t_max istead"); t2 = self.t_max

        i1,i2 = self.index_at(t1), self.index_at(t2)
        rep = self.empty_copy()
        rep.data = self.data[i1:i2]
        rep.name += ' recuted'
        return rep

    def empty_copy(self):
        """Copy everything exept data (including t0)"""
        return C_signal(data = None,
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name,
                        t0 = self.t_min())

    def copy(self):
        return C_signal(data = self.data.copy(),
                        fs = self.fs,
                        unit = self.unit,
                        name = self.name,
                        t0 = self.t_min())

    def fft(self,   n: int=None,
                    n_factor: float=None,
                    choose_next_power_of_2 = True,
                    print_choices = False,
                    **kwargs):
        """
        Args 
        -----
        `n`: int (default = len of the data)
            number of point used to compute the fft (must be greater than the number of point in the signal)
        `n_factor` : float (optional)
            If not None, `n`=len(data)*`n_factor`
        `choose_next_power_of_2` : bool (default = True)
            If true use the newxt power of 2 istead of `n`
        `print_choices` : bool

        Return
        ------
        `xf` : Array
        `yf_real` : Array
        """
        # TODO use apodization windows (hamming,...)

        if n is None : n = self.N
        if n_factor is not None : n = int(self.N * n_factor)

        if n < self.N:
            print("n is smaller than self.data, we use n=len(self.data) instead")
            n=self.N

        if choose_next_power_of_2 :
            n = int(2**np.ceil(np.log(n) / np.log(2)))

        if print_choices :
            print(f"FFT : n={n}, len(data)={self.N}")

        # FFT avec 0 padding sur n
        yf = spf.rfft(self.data,n)
        xf = spf.rfftfreq(n,1/self.fs)
        return FFT_signal(  data = yf,
                            fs = 1/(xf[1]-xf[0]),
                            unit = '',
                            name = f"{self.name} FFT",
                            f0 = 0,
                            nb_zero = n - self.N,
                            window = None)

    def fill_like(sig1:C_signal,sig2:C_signal,infos=True):
        #TODO Make it work !!!
        """
        Make to signals compatible in terms of t0 and duration
        
        Args
        -----
        `sig1` : C_signal
        `sig2` : C_signal

        Return
        -----
        `sig1_` : C_signal
            (a copy of sig1)
        `sig2_` : C_signal
            (a copy of sig2)
        """

        if sig1.fs != sig2.fs : 
            raise ValueError("fill_like with differents fs")
        
        sig1__ = sig1.copy()
        sig2__ = sig2.copy()

        N_diff_A = (sig1__.t0 - sig2__.t0) * sig1__.fs

        if N_diff_A > 0 :
            # sig2__.t0 < sig1__.t0
            sig1__.data = np.pad(sig1__.data,(N_diff_A,0))
            sig1__.t0 = sig2__.t0
            if infos : print("fill due to different t0")
        elif N_diff_A < 0 :
            # sig1__.t0 < sig2__.t0
            sig2__.data = np.pad(sig2__.data,(-N_diff_A,0))
            sig2__.t0 = sig1__.t0
            if infos : print("fill due to different t0")

        N_diff_B = sig1__.N - sig2__.N

        if N_diff_B > 0 : 
            sig2__.data = np.pad(sig2__.data,(0,N_diff_B))
            if infos : print("fill due to different duration")
        elif N_diff_B < 0 :
            sig1__.data = np.pad(sig1__.data,(0,-N_diff_B))
            if infos : print("fill due to different duration")

        return sig1__ , sig2__


    def __add__(self,val):
        if isinstance(val, (float, int)):
            new_sig = self.copy()
            new_sig.data += val
            new_sig.name += f" + {val}"

        elif isinstance(val,C_signal) :
            if self.fs != val.fs :
                raise NotImplementedError("Sum with differents fs")
            
            new_sig,sig2 = C_signal.fill_like(self,val)

            new_sig.data += sig2.data
            new_sig.name = f"({new_sig.name} + {sig2.name})"

        else : raise NotImplementedError(f"You cannot sum C_signal and {type(val)} objects")
        return new_sig

    __radd__ = __add__

    def __mul__(self,val):
        if isinstance(val,(int,float)):
            new_sig = self.copy()
            new_sig.data *= val
            new_sig.name += f" * {val}"


        elif isinstance(val,C_signal):
            if self.fs != val.fs :
                raise NotImplementedError("operation with differents fs")
            new_sig,sig2 = C_signal.fill_like(self,val)
            

            new_sig.data *= sig2.data
            new_sig.name += f" * {sig2.name}"

        else : raise NotImplementedError(f"You cannot multiply C_signal and {type(val)} objects")
        return new_sig

    __rmul__ = __mul__



    def __truediv__(self,val):
        if isinstance(val,(int,float)):
            new_sig = self.copy()
            new_sig.data /= val
            new_sig.name += f" / {val}"


        elif isinstance(val,C_signal):
            if self.fs != val.fs :
                raise NotImplementedError("operation with differents fs")
            new_sig,sig2 = C_signal.fill_like(self,val)
            

            new_sig.data /= sig2.data
            new_sig.name += f" / {sig2.name}"

        else : raise NotImplementedError(f"You cannot divide C_signal and {type(val)} objects")
        return new_sig
    
    def __rtruediv__(self,val):
        if isinstance(val,(int,float)):
            new_sig = self.copy()
            new_sig.data = val / new_sig.data
            new_sig.name = f"{val} / {new_sig.name}"


        elif isinstance(val,C_signal):
            if self.fs != val.fs :
                raise NotImplementedError("operation with differents fs")
            new_sig,sig2 = C_signal.fill_like(self,val)
            

            sig2.data /= new_sig.data
            new_sig.name = f"{sig2.name} / {new_sig.name}"

        else : raise NotImplementedError(f"You cannot divide {type(val)} and C_signal objects")
        return new_sig
    
    def __sub__(self,val):
        if isinstance(val, (float, int)):
            new_sig = self.copy()
            new_sig.data -= val
            new_sig.name += f" - {val}"

        elif isinstance(val,C_signal) :
            if self.fs != val.fs :
                raise NotImplementedError("operation with differents fs")
            
            new_sig,sig2 = C_signal.fill_like(self,val)

            new_sig.data += sig2.data
            new_sig.name = f"({new_sig.name} - {sig2.name})"

        else : raise NotImplementedError(f"You cannot substract C_signal and {type(val)} objects")
        return new_sig
    
    def __rsub__(self,val):
        if isinstance(val, (float, int)):
            new_sig = self.empty_copy()
            new_sig.data = val - self.data 
            new_sig.name = f"{val} - {self.name}"

        elif isinstance(val,C_signal) :
            if self.fs != val.fs :
                raise NotImplementedError("operation with differents fs")
            
            new_sig,sig2 = C_signal.fill_like(self,val)

            sig2.data -= new_sig.data
            new_sig.name = f"({sig2.name} - {new_sig.name})"

        else : raise NotImplementedError(f"You cannot substract C_signal and {type(val)} objects")
        return new_sig

    def __neg__(self):
        new_sig = self.copy()
        new_sig.data.__mul__ *= -1
        new_sig.name = f"- {self.name}"
        return new_sig




    def resample(self,new_fs:float,fs_factor:float=None):
        """
        Args
        -----
        
        `new_fs` : float
        `fs_factor` : float
            if not None, `new_fs` = `fs` * `fs_factor`
        """
        # TODO to implement
        raise NotImplementedError()
    
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

    - `fft`
    
    Others
    - `index_at`
    - `N`
    """

    def __init__(   self,
                    data : np.ndarray | list = None,
                    fs : float = 1,
                    unit : str = '',
                    name : str = '',
                    t0 : float = 0, #TODO make it fonctional ?
                    from_sig : C_signal = None):
        """
        Args
        ----
        `data`: array | list
            time value for the vertical line
        fs : float (default = 1)
            Sampling frequency
        unit : str (default = '')
        name : str (default = '')
        t0 : float (default = 0)        
        """
        if from_sig is None: 
            self.SIG = C_signal(data=data,
                                fs=fs,
                                unit=unit,
                                name=name,
                                t0=t0)

        else: 
            self.SIG = from_sig

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

        self.plot = self.SIG.plot

        self.plot_ADD_t_at_min = self.SIG.plot_ADD_t_at_min
        # self.plot_ADD_times = self.SIG.plot_ADD_times
        self.plot_ADD_box = self.SIG.plot_ADD_box
        self.plot_ADD_mean = self.SIG.mean
        self.plot_ADD_val = self.SIG.plot_ADD_val #TODO implement it
        # self.plot_ADD_values = self.SIG.plot_ADD_values


        # TODO to replace : we need to retun a T_signal
        # self.copy = self.SIG.copy
        # self.empty_copy = self.SIG.empty_copy

        self.fft = self.SIG.fft

        self.index_at = self.SIG.index_at
        self.N = self.SIG.N

    def recut(self,t1: float=0,t2: float=None) :
        """
        Args
        -----
        `t1` : float
            Beginning of the recuted signal
        `t2` : float (optional)
            End of the recuted signal, use None to keep the signal until the end
        """
        return T_signal(from_sig = self.SIG.recut(t1=t1,t2=t2))
        
    def copy(self):
        return T_signal(from_sig=self.SIG.copy())
    
    def empty_copy(self):
        return T_signal(from_sig=self.SIG.empty_copy())

    def __add__(self,val):
        if isinstance(val,(float,int)):
            print("verifier le type") # TODO 
            return self.SIG + val
        elif isinstance(val,T_signal):
            # TODO ajouter un test de l'unité utilisée
            print("verifier le type") # TODO 
            return self.SIG + val.SIG
        else : raise NotImplementedError(f"You sum T_signal and {type(val)} objects")
        

    __radd__ = __add__


    def __mul__(self,val):
        if isinstance(val,(float,int)):
            print("verifier le type") # TODO 
            return self.SIG * val
        elif isinstance(val,T_signal):
            # TODO ajouter un test de l'unité utilisée
            print("verifier le type") # TODO 
            return self.SIG * val.SIG
        else : raise NotImplementedError(f"You can't multiply T_signal and {type(val)} objects")

    __rmul__ = __mul__

    def __truediv__(self, val):
        if isinstance(val,(int,float)):
            return self.SIG / val
        elif isinstance(val,T_signal):
            return self.SIG / val.SIG
        else : raise NotImplementedError(f"You can't divide T_signal and {type(val)} objects")

    def __rtruediv__(self,val):
        if isinstance(val,(int,float)):
            return val / self.SIG
        elif isinstance(val,T_signal):
            return val.SIG / self.SIG 
        else : raise NotImplementedError(f"You can't divide {type(val)} and T_signal objects")



    def __rsub__(self,val):
        if isinstance(val,(int,float)):
            return val - self.SIG
        elif isinstance(val,T_signal):
            return val.SIG - self.SIG 
        else : raise NotImplementedError(f"You can't substract {type(val)} and T_signal objects")

    def __sub__(self,val):
        if isinstance(val,(int,float)):
            return self.SIG - val
        elif isinstance(val,T_signal):
            return self.SIG - val.SIG
        else : raise NotImplementedError(f"You can't substract {type(val)} and T_signal objects")


    @property
    def data(self): return self.SIG.data
    @data.setter
    def data(self,new_data): self.SIG.data = new_data

    @property
    def name(self): return self.SIG.name
    @name.setter
    def name(self,new_name): self.SIG.name = new_name

    @property
    def t0(self): return self.SIG.t0
    @t0.setter
    def t0(self,new_t0): self.SIG.t0 = new_t0

    @property
    def fs(self): return self.SIG.fs
    @t0.setter
    def fs(self,new_fs): self.SIG.fs = new_fs

    def plot_ADD_t(self,t:float,**kwargs):
        """ Plot a vertical line

        Args 
        -----
        `t`: float
            time value for the vertical line
        """
        yh = self.max()
        yl = self.min()
        e = yh-yl
        yh += e * 0.1 # to set the upper margin
        yl -= e * 0.1 # to set the lower margin
        
        X = [t]*2
        Y = [yl,yh]
        plt.plot(X,Y,'--',label=f"t={t}s",**kwargs)


    def plot_ADD_t_at_max(self,**kwargs): 
        """Plot a vertical line where the signal is maximal"""
        self.plot_ADD_t(self.t_at_max(),**kwargs)


    def fft_new(self,   
                n: int=None,
                n_factor: float=None,
                choose_next_power_of_2 = True,
                print_choices = False,
                **kwargs):
        """
        Args 
        -----
        `n`: int (default = len of the data)
            number of point used to compute the fft (must be greater than the number of point in the signal)
        `n_factor` : float (default = not used)
            If not None, `n`=len(data)*`n_factor`
        `choose_next_power_of_2` : bool (default = True)
            If true use the newxt power of 2 istead of `n`
        `print_choices` : bool (default = False)

        Return
        ------
        `xf` : Array
        `yf_real` : Array
        """
        # TODO use apodization windows (hamming,...)

        if n is None : n = self.N
        if n_factor is not None : n = int(self.N * n_factor)

        if n < self.N:
            print("n is smaller than self.data, we use n=len(self.data) instead")
            n=self.N

        if choose_next_power_of_2 :
            n = int(2**np.ceil(np.log(n) / np.log(2)))

        if print_choices :
            print(f"FFT : n={n}, len(data)={self.N}")

        # FFT avec 0 padding sur n
        yf = spf.rfft(self.data,n)
        xf = spf.rfftfreq(n,1/self.fs)
        return FFT_signal(  data = yf,
                            fs = 1/(xf[1]-xf[0]),
                            unit = '',
                            name = f"{self.name} FFT",
                            f0 = 0,
                            nb_zero = n - self.N,
                            window = None)



class F_signal:
    """
    Who return an amplitude value
    - `mean`
    - `std`
    - `max`
    - `min`
    - `val_at_nearset_t`

    Who return a time value
    - `f_at_max`
    - `f_at_min`
    - `f_range`
    - `f_max`
    - `f_min`
    
    Who plot something
    - `plot`

    - `plot_ADD_f_at_max`
    - `plot_ADD_f_at_min`
    - `plot_ADD_f`  =========================== to implement
    - `plot_ADD_frequences` ======================== to delete
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
    - `N`
    """

    def __init__(   self,
                    data : np.ndarray | list = None,
                    fs : float = 1,
                    unit : str = '',
                    name : str = ''): #TODO make it fonctional ?
        """
        Args :
        ------
        `data` : Array or list of values
        `fs` : << Sampling frequency >> calculated by 1/(f1-f0)
        `unit` : unit of the signal
        `name` : name of the signal
        """
        
        self.SIG = C_signal(data=data,
                            fs=fs,
                            unit=unit,
                            name=name,
                            t0=0)

        self.mean = self.SIG.mean
        self.std = self.SIG.std
        self.max = self.SIG.max
        self.min = self.SIG.min
        self.val_at_nearest_f = self.SIG.val_at_nearest_t

        self.f_at_max = self.SIG.t_at_max
        self.f_at_min = self.SIG.t_at_min
        self.f_range = self.SIG.duration #TODO check it
        self.f_max = self.SIG.t_max
        self.f_min = self.SIG.t_min

        # self.plot = self.SIG.plot

        # self.plot_ADD_f_at_max = self.SIG.plot_ADD_t_at_max
        # self.plot_ADD_f_at_min = self.SIG.plot_ADD_t_at_min
        # self.plot_ADD_f = self.SIG.plot_ADD_t #TODO implement it
        # self.plot_ADD_frequences = self.SIG.plot_ADD_times
        self.plot_ADD_box_on_recut = self.SIG.plot_ADD_box
        self.plot_ADD_mean = self.SIG.mean
        self.plot_ADD_val = self.SIG.plot_ADD_val #TODO implement it
        # self.plot_ADD_values = self.SIG.plot_ADD_values

        self.recut = self.SIG.recut

        # TODO to replace : we need to retun a F_signal
        # self.copy = self.SIG.copy
        # self.empty_copy = self.SIG.empty_copy
        
        self.index_at = self.SIG.index_at
        self.N = self.SIG.N
        
    def plot(   self,
                new_unit : tuple[str,float] = None,
                add_to_title='',
                new_figure = True,
                **kwargs):

        # TODO create a unit class ?
        
        if new_figure : plt.figure()
        unit = self.unit 
        data = self.data
        if new_unit is not None :
            unit = new_unit[0]
            data = data*new_unit[1]

        plt.grid(True)
        N = len(data)
        X=np.linspace(start=self.f_min() , stop=self.f_max(), num=N)
        plt.plot(X, data,label = f"{self.name}",**kwargs)
        plt.xlabel("frequency (Hz)")
        if unit is None : plt.ylabel('Amplitude')
        else : plt.ylabel(f"Amplitude ({unit})")
        plt.title(f"{self.name} {add_to_title}")

    def plot_ADD_f(self,f:float,**kwargs):
        """ Plot a vertical line

        Args 
        -----
        `t`: float
            time value for the vertical line
        """
        yh = self.max()
        yl = self.min()
        e = yh-yl
        yh += e * 0.1 # to set the upper margin
        yl -= e * 0.1 # to set the lower margin
        
        X = [f]*2
        Y = [yl,yh]
        plt.plot(X,Y,'--',label=f"f={f}Hz",**kwargs)

    def plot_ADD_f_at_max(self):
        self.plot_ADD_f(self.f_at_max())

    def plot_ADD_f_at_min(self):
        self.plot_ADD_f(self.f_at_min())


    @property
    def data(self): return self.SIG.data
    @data.setter
    def data(self,new_data): self.SIG.data = new_data

    @property
    def unit(self): return self.SIG.unit
    @unit.setter
    def unit(self,new_unit): self.SIG.unit = new_unit

    @property
    def name(self): return self.SIG.name
    @name.setter
    def name(self,new_name): self.SIG.name = new_name
























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
        - `plotM_ADD_box`
        - `plotM_ADD_freqs`
        - `plotM_ADD_moduls`
    - On plot_phase
        - `plotP_ADD_box`
        - `plotP_ADD_freqs`
        - `plotP_ADD_phases`

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
        `fs` : << Sampling frequency >> calculated by 1/(f1-f0)
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

        self.__compute_modul_phase_from_data()
        self.__init_functions()
    
    def __compute_modul_phase_from_data(self):
            
        # FFT avec 0 padding sur n
        # yf = np.fft.fft(self.data,n)
        # xf = np.linspace(0,self.fs/2,n//2-1)
        # xf = np.fft.fftfreq(n,d=1/fs)[:n//2-1]

        N = self.N
        n = (len(self.data)-1)*2 if N%2 != 0 else len(self.data)*2-1
        yf_modul = 2/(n-self.nb_zero) * np.abs(self.data)
        # yf_modul = 2/np.size(self.data)*np.abs(self.data[0:n//2-1])
        # multiplication par 2 car puissances divisée entre freqs positifs et négatifs
        # division par la taille de max_spectro pour prendre en compte le pas 
        yf_phase = np.angle(self.data)

        self.Modul = F_signal( data = yf_modul,
                                fs = self.fs,
                                unit = '?',
                                name = f"{self.name} (FFT_modul)")

        self.Phase = F_signal( data = yf_phase,
                                fs = self.fs,
                                unit = 'rad ?',
                                name = f"{self.name} (FFT_phase)")

    def __init_functions(self):

        self.plotM = self.Modul.plot
        self.plotM_ADD_box = self.Modul.plot_ADD_box_on_recut
        self.plotM_ADD_f = self.Modul.plot_ADD_f
        
        self.plotM_ADD_val = self.Modul.plot_ADD_val
        self.plotM_ADD_f_at_max = self.Modul.plot_ADD_f_at_max
        self.plotM_ADD_f_at_min = self.Modul.plot_ADD_f_at_min

        self.plotP = self.Phase.plot
        self.plotP_ADD_box = self.Phase.plot_ADD_box_on_recut
        self.plotP_ADD_f = self.Phase.plot_ADD_f
        self.plotP_ADD_phases = self.Phase.plot_ADD_val
        self.plotP_ADD_f_at_max = self.Phase.plot_ADD_f_at_max
        self.plotP_ADD_f_at_min = self.Phase.plot_ADD_f_at_min
            
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



    def f_max(self)-> float : return self.N/self.fs

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

    def plot_ADD_freq(self):
        #TODO test it
        plt.subplot(211)
        self.plotM_ADD_f
        plt.subplot(212)
        self.plotP_ADD_f


    def plot_ADD_box(self):
        #TODO test it
        plt.subplot(211)
        self.plotM_ADD_box
        plt.subplot(212)
        self.plotP_ADD_box

    def copy(self):
        return FFT_signal(  data = self.data.copy(),
                            fs = self.fs,
                            unit = self.unit,
                            name = self.name,
                            f0 = self.f0)

    def empty_copy(self): 
        """Warning : This function conserve the f0 value"""
        return FFT_signal(  data = None,
                            fs = self.fs,
                            unit = self.unit,
                            name = self.name,
                            f0 = self.f0)

    @property
    def N(self): return 0 if self.data is None else len(self.data)


    def ifft(self)-> T_signal:
        t_data = spf.irfft(self.data)
        new_sig = T_signal( data = t_data,
                            fs = self.f_max() / 2, #TODO change it
                            unit = '?',
                            name = f"{self.name}_from_FFT",
                            t0 = 0)
        return new_sig








#   def fft(self,   n: int=None,
#                     n_factor: float=None,
#                     choose_next_power_of_2 = True,
#                     print_choices = False,
#                     **kwargs):
#         """
#         Args 
#         -----
#         `n`: int (default = len of the data)
#             number of point used to compute the fft (must be greater than the number of point in the signal)
#         `n_factor` : float (optional)
#             If not None, `n`=len(data)*`n_factor`
#         `choose_next_power_of_2` : bool (default = True)
#             If true use the newxt power of 2 istead of `n`
#         `print_choices` : bool

#         Return
#         ------
#         `xf` : Array
#         `yf_real` : Array
#         """
#         # TODO use apodization windows (hamming,...)

#         if n is None : n = self.N
#         if n_factor is not None : n = int(self.N * n_factor)

#         if n < self.N:
#             print("n is smaller than self.data, we use n=len(self.data) instead")
#             n=self.N

#         if choose_next_power_of_2 :
#             n = int(2**np.ceil(np.log(n) / np.log(2)))

#         if print_choices :
#             print(f"FFT : n={n}, len(data)={self.N}")

#         # FFT avec 0 padding sur n
#         yf = spf.rfft(self.data,n)
#         xf = spf.rfftfreq(n,1/self.fs)
#         return FFT_signal(  data = yf,
#                             fs = 1/(xf[1]-xf[0]),
#                             unit = '',
#                             name = f"{self.name} FFT",
#                             f0 = 0,
#                             nb_zero = n - self.N,
#                             window = None)






























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


    def f_current(sig_V : T_signal,num : int):
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

        sig_I = T_signal(data=None, fs=sig_V.fs, unit='A', name=f"U_I{num} converted to I{num}")
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
    def FIR(            signal : T_signal | list[T_signal],
                        order : int,
                        cutoff : float | list[float],
                        window : str = 'hamming',
                        type_filtre : Literal['low_pass','high_pass'] = 'low_pass') -> T_signal | list[T_signal]:
        if order == 0:
            return signal

        if type(signal) == T_signal : 
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
            signal_f = T_signal(y, signal.fs, signal.unit, f"{signal.name} - ({window},{order},{cutoff}Hz)")
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














    # def plot_ADD_times(self, l_times:list|np.ndarray, **kwargs):
    #     yh = self.max()
    #     yl = self.min()
    #     e = yh-yl
    #     yh += e * 0.1 # to set the upper margin
    #     yl -= e * 0.1 # to set the lower margin
        
    #     for time in l_times :
    #         X = [time]*2
    #         Y = [yl,yh]
    #         plt.plot(X,Y,'--',label=f"t={time}s",**kwargs)

    

    # def plot_ADD_values(self,   l_values:list[float],
    #                             l_labels:list[str]=None,
    #                             l_colors:list[str]=None,
    #                             **kwargs):
    #     X = [self.t_max(), self.t_min()]
    #     if l_labels is None : l_labels = [None]*len(l_values)
    #     if l_colors is None : l_colors = [None]*len(l_values)

    #     for value,label,color in zip(l_values,l_labels,l_colors):
    #         Y = [value]*2
    #         plt.plot(X,Y,'--',label=label,color=color,**kwargs)