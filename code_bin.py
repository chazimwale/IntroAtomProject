class OldLaserLinewidth:

# This function is specific to the Oscilloscope used in LG.03
    
    def __init__(self,filename):
        self.filename = filename
        # self.df = None
    
    def oscilloscope_data(self, plot = True, figsize = (12,8)):

        # Reading data with pandas, ensuring numeric conversion
        df = pd.read_csv(self.filename,skiprows=11,names = ['TIME','CH1'])
        
        # Validate datafram structure
        if 'TIME' not in df.columns or 'CH1' not in df.columns:
            print("Invalid CSV structure")
            return np.array([]),np.array([])
        
        # Convert to numeric and force errors to NaN
        time_series = pd.to_numeric(df['TIME'], errors = 'coerce')
        volts_series = pd.to_numeric(df['CH1'], errors = 'coerce')
        
        # Checking for NaN points
        if time_series.isna().all() or volts_series.isna().all():
            print("Numeric conversion failed for all data")
            return np.array([]), np.array([])
        
        time = time_series.values
        volts = volts_series.values
       
        # Remove any NaN values present
        valid_mask = ~(np.isnan(time) | np.isnan(volts))
        time = time[valid_mask]
        volts = volts[valid_mask]

        volts_raw = volts
        volts_clean = volts_raw - np.mean(volts_raw)

        # Final Validation
        if len(time) == 0 or len(volts) == 0:
            print("No valid data found after filtering")
            return np.array([]),np.array([])
        
        assert len(time) == len(volts)
        
        # -------- DETERMINING BEAT FREQUENCY ------------ 

        # FFT Initialisation
        dt = time[1] - time[0]
        N = len(volts_clean)
    
        # FFT Implementation for positive frequencies only
        fft_clean = np.fft.fft(volts_clean)
        freqs = np.fft.fftfreq(N, dt)
        
        mask = freqs > 0
        freqs_mhz = freqs[mask] * 1e-6
        spectrum = np.abs(fft_clean[mask])
        spectrum /= np.max(spectrum)
        
        # Identify main beat peak
        peak_index = np.argmax(spectrum)
        peak_freq = freqs_mhz[peak_index] #  *This* is the beat frequency

        #-----------------------------------------------
        
        if plot:

            plt.figure(figsize= figsize)
            plt.grid(True)

            title_size = 30
            lab_size = legend_size = 25
            xtick_size = ytick_size = 25

            plt.plot(time * 1e6, volts_raw * 1e3, 'b-', label='Raw Signal', linewidth=2)
            plt.plot(time * 1e6, volts_clean * 1e3,  'r-', label='Clean Signal', linewidth=2)
            plt.title(fr'Oscilloscope data for $\Delta f$ = {peak_freq:.3f} MHz', size = title_size)
            plt.xlabel(r'time, $\mu$s', size = lab_size)
            plt.xticks(size = xtick_size)
            plt.ylabel(r'Voltage, mV', size = lab_size)
            plt.yticks(size = ytick_size)
            plt.legend(fontsize =legend_size)

        return {
            "time_data": time, 
            "raw_voltage_data": volts_raw, 
            "clean_voltage_data": volts_clean
        }
    
    def spectrum_data_old(self, plot = True):
        osci_data = self.oscilloscope_data(plot = False)
        time = osci_data['time_data']
        volts_raw = osci_data['raw_voltage_data']
        volts_clean = osci_data['clean_voltage_data']

        # Sampling interval (dt) and rate.
        dt = time[1] - time[0]
        rate = 1/dt

        N,M = len(volts_raw), len(volts_clean)
        assert N == M

        # Raw data (before processing)
        fft_raw_result = np.fft.fft(volts_raw)
        freqs_raw = np.fft.fftfreq(N,dt)
        
        # Create mask for positive frequencies only
        mask_raw = freqs_raw > 0
        freqs_raw_data = freqs_raw[mask_raw]
        spectrum_raw = np.abs(fft_raw_result[mask_raw])

        # Cleaned data (after processing)
        fft_clean_result = np.fft.fft(volts_clean)
        freqs_clean = np.fft.fftfreq(M,dt)
        
        # Create mask for positive frequencies only
        mask_clean = freqs_clean > 0
        freqs_clean_data = freqs_clean[mask_clean]
        spectrum_clean = np.abs(fft_clean_result[mask_clean])

        # Identify main beat peak

        freqs_mhz_raw = freqs_raw[mask_raw] * 1e-6
        peak_index_raw = np.argmax(spectrum_raw)
        peak_freq_raw = freqs_mhz_raw[peak_index_raw] #  *This* is the beat frequency raw

        freqs_mhz_clean = freqs_clean[mask_clean] * 1e-6
        peak_index_clean = np.argmax(spectrum_clean)
        peak_freq_clean = freqs_mhz_clean[peak_index_clean] #  *This* is the beat frequency clean

        if plot:

            # Create figure and subplots with shared y-axis if desired
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
            
            xlab_size = ylab_size = legend_size = 23
            xtick_size = ytick_size = 20

            # Plot 1: Raw signal FFT (positive frequencies only)
            ax1.plot(freqs_raw_data*1e-6, spectrum_raw, color='b', label='Raw Signal FFT', linewidth=2)
            ax1.set_ylabel(r'Amplitude', size=ylab_size)
            ax1.tick_params(axis='y', labelsize=ytick_size)
            ax1.tick_params(axis='x', labelsize=xtick_size)
            ax1.legend(fontsize=legend_size)
            ax1.set_yscale('log')
            ax1.grid(True)

            # Plot 2: Cleaned signal FFT (positive frequencies only)
            ax2.plot(freqs_clean_data*1e-6, spectrum_clean, color='r', label='Processed Signal FFT', linewidth=2)
            ax2.set_xlabel(r'Frequency, MHz', size=ylab_size)
            ax2.set_ylabel(r'Amplitude', size=ylab_size)
            ax2.tick_params(axis='both', labelsize=xtick_size)
            ax2.legend(fontsize=legend_size)
            ax2.set_yscale('log')
            ax2.grid(True)

            plt.tight_layout()
            plt.show()
    
        return {
            "Raw Frequency Data": freqs_raw_data,
            "Raw Spectrum Data": spectrum_raw,
            "Raw Beat Frequency" : peak_freq_raw, 
            "Clean Frequency Data": freqs_clean_data,
            "Clean Spectrum Data": spectrum_clean,
            "Clean Beat Frequency" : peak_freq_clean,
        }


    def spectrum_data(self, plot=True):
        # Efficient data loading
        data_dict = self.oscilloscope_data(plot=False)
        time = data_dict["time_data"]
        volts_raw = data_dict["raw_voltage_data"]
        volts_clean = data_dict["clean_voltage_data"]

        dt = time[1] - time[0]
        N = len(volts_raw)

        # FFT calculations
        fft_raw = np.fft.fft(volts_raw)
        fft_clean = np.fft.fft(volts_clean)
        freqs = np.fft.fftfreq(N, dt)
        
        # Positive frequencies only
        mask = freqs > 0
        freqs_mhz = freqs[mask] * 1e-6
        
        spectrum_raw = np.abs(fft_raw[mask])
        spectrum_clean = np.abs(fft_clean[mask])
        
        # Normalize both for fair comparison
        spectrum_raw /= np.max(spectrum_raw)
        spectrum_clean /= np.max(spectrum_clean)
        
        # Find peaks
        peak_idx_raw = np.argmax(spectrum_raw)
        peak_idx_clean = np.argmax(spectrum_clean)
        peak_freq_raw = freqs_mhz[peak_idx_raw]
        peak_freq_clean = freqs_mhz[peak_idx_clean]

        if plot:
            self._plot_spectrum_comparison(freqs_mhz, spectrum_raw, spectrum_clean, 
                                        peak_freq_raw, peak_freq_clean)
        
        return {
            "frequency_data_MHz": freqs_mhz,
            "raw_spectrum": spectrum_raw,
            "clean_spectrum": spectrum_clean,
            "raw_beat_frequency_MHz": peak_freq_raw,
            "clean_beat_frequency_MHz": peak_freq_clean,
        }

    def _plot_spectrum_comparison(self, freqs_mhz, spectrum_raw, spectrum_clean, 
                                peak_freq_raw, peak_freq_clean):
        """Helper function for spectrum comparison plotting"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        
        # Common styling
        lab_size = 20
        tick_size = 16
        legend_size = 14
        
        # Plot 1: Raw spectrum (log scale)
        ax1.plot(freqs_mhz, spectrum_raw, 'b-', linewidth=2, label='Raw Signal')
        ax1.axvline(peak_freq_raw, color='blue', linestyle='--', alpha=0.7, 
                    label=f'Peak: {peak_freq_raw:.3f} MHz')
        ax1.set_ylabel('Amplitude (norm)', size=lab_size)
        ax1.set_yscale('log')
        ax1.legend(fontsize=legend_size)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=tick_size)
        ax1.set_title('Raw Signal Spectrum - Dominated by DC Spike', size=16)
        
        # Plot 2: Clean spectrum (log scale)
        ax2.plot(freqs_mhz, spectrum_clean, 'r-', linewidth=2, label='Clean Signal')
        ax2.axvline(peak_freq_clean, color='red', linestyle='--', alpha=0.7,
                    label=f'Peak: {peak_freq_clean:.3f} MHz')
        ax2.set_ylabel('Amplitude (norm)', size=lab_size)
        ax2.set_yscale('log')
        ax2.legend(fontsize=legend_size)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=tick_size)
        ax2.set_title('Clean Signal Spectrum - Clear Beat Frequency', size=16)
        
        # Plot 3: Direct comparison (linear scale, zoomed)
        ax3.plot(freqs_mhz, spectrum_raw, 'b-', linewidth=2, alpha=0.7, label='Raw')
        ax3.plot(freqs_mhz, spectrum_clean, 'r-', linewidth=2, label='Clean')
        ax3.set_xlabel('Frequency (MHz)', size=lab_size)
        ax3.set_ylabel('Amplitude (norm)', size=lab_size)
        ax3.legend(fontsize=legend_size)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=tick_size)
        ax3.set_title('Direct Comparison (Zoomed to Beat Frequency)', size=16)
        
        # Zoom to region around beat frequency
        zoom_range = 5  # MHz
        ax3.set_xlim([peak_freq_clean - zoom_range, peak_freq_clean + zoom_range])
        ax3.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.show()
    def gauss_func(self, x, A, x0, sigma, offset):
        """Gaussian for curve fitting."""
        return A * np.exp(-((x - x0)**2) / (2 * sigma**2)) + offset
    
    def lorentz_func(self, x, A, x0, gamma,offset):
        return A / (1 + ((x - x0)/(gamma/2))**2) + offset

    def voigt_func(self,x, A, x0, sigma, gamma,offset):

        z = ((x - x0) + 1j* gamma) / (sigma * np.sqrt(2))

        return A * np.real(sp.wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + offset

    def voigt_linewidth(self, plot = True):

        # Loading Oscilloscope Data
        time, volts_raw, volts_clean = self.oscilloscope_data()
        dt = time[1] - time[0]
        N = len(volts_clean)
    
        
        # # --- Apply Hanning window ---
        window = np.hanning(N)
        volts_windowed = volts_clean * window
        
        # # Normalize for energy correction (so window doesn’t affect amplitude scaling)
        correction_factor = np.sqrt(np.mean(window**2))
        volts_windowed /= correction_factor
        
        # --- FFT ---
        fft_clean = np.fft.fft(volts_windowed)
        freqs = np.fft.fftfreq(N, dt)
        
        # Keep positive frequencies only
        mask = freqs > 0
        freqs_mhz = freqs[mask] * 1e-6
        spectrum = np.abs(fft_clean[mask])

        spectrum /= np.max(spectrum)
        
        # Identify main beat peak
        peak_index = np.argmax(spectrum)
        peak_freq = freqs_mhz[peak_index]
        
        # Define region for fitting (2 MHz window around peak)
        window_mhz = 2
        region_mask = (freqs_mhz > peak_freq - window_mhz) & (freqs_mhz < peak_freq + window_mhz)
        x_fit = freqs_mhz[region_mask]
        x_smooth = np.linspace(x_fit[0], x_fit[-1], 1000)  # 500 points for a smooth curve
        y_fit = spectrum[region_mask]
        
        # Normalisation for numerical stability
        y_fit /= np.max(y_fit)

        # Voigt Fit Initial Guesses

        A_guess = np.max(y_fit)
        x0_guess = peak_freq
        sigma_guess = 0.5
        gamma_guess = 0.5
        offset_guess = np.min(y_fit)


        # Bounds for the Voigt fit recommended, this stops 
        bounds = ([0, peak_freq - 0.05, 0.01, 0.01, 0],
                  [2, peak_freq + 0.05, 5.0, 5.0, 0.2])

        try:
            popt, pcov = curve_fit(self.voigt_func, x_fit, y_fit,
                                p0=[A_guess, x0_guess, sigma_guess, gamma_guess, offset_guess],
                                bounds = bounds, maxfev = 100000)
        
        except RuntimeError:
            print(f"Voigt fit did not converge for {self.filename}")
            return None
        
        # Extract Parameters
        A_fit, x0_fit, sigma_fit, gamma_fit, offset_fit = popt

        fwhm_gauss = 2 * np.sqrt(2*np.log(2)) * sigma_fit
        fwhm_lorentz = 2 * gamma_fit
        fwhm_voigt= 0.5343 * fwhm_lorentz + np.sqrt(0.2166 * fwhm_lorentz**2 + fwhm_gauss)

        perr = np.sqrt(np.diag(pcov))
        A_err, x0_err, sigma_err, gamma_err, offset_err = perr

        # Plotting
        if plot:
            plt.figure(figsize=(12, 8))
            labsize = 25
            ticksize = 20
            titlesize = 35
            lw = 2.0


            plt.plot(freqs_mhz, spectrum, 'y-', label='FFT Spectrum', linewidth = lw)
            plt.plot(x_smooth, self.voigt_func(x_smooth, *popt), 'b', 
                    label=(f'Voigt Fit\nA = {A_fit:.4f} ± {A_err:.4f}\n'
                            f'Center = {x0_fit:.4f} ± {x0_err:.4f} MHz\n'
                            f'FWHM = {fwhm_voigt:.4f} ± {2*np.sqrt(2*np.log(2))*sigma_err:.4f} MHz'), 
                    linewidth=lw)
            plt.xlabel("Frequency (MHz)", size = labsize)
            plt.xticks(size = ticksize)
            plt.ylabel("Amplitude", size = labsize)
            plt.yticks(size = ticksize)
            plt.xlim([peak_freq - window_mhz/4, peak_freq + window_mhz/4])
            plt.title(f"Laser Linewidth for f = {peak_freq:.4f} MHz", size = titlesize)
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return {
        "center_frequency_MHz": x0_fit,
        "center_error_MHz": x0_err,
        "sigma_MHz": sigma_fit,
        "sigma_error_MHz": sigma_err,
        "FWHM_MHz": fwhm_voigt,
        "offset": offset_fit,
        "offset_error": offset_err,
        "fit_params": popt,
        "fit_covariance": pcov,
        # "FWHM_error_MHz": 2 * np.sqrt(2*np.log(2)) * sigma_err,
    }

    def gauss_linewidth(self, plot=True):
        
        # Loading Oscilloscope Data
        time, volts_raw, volts_clean = self.oscilloscope_data()
        dt = time[1] - time[0]
        N = len(volts_clean)
    
        
        # # --- Apply Hanning window ---
        window = np.hanning(N)
        volts_windowed = volts_clean * window
        
        # # Normalize for energy correction (so window doesn’t affect amplitude scaling)
        correction_factor = np.sqrt(np.mean(window**2))
        volts_windowed /= correction_factor
        
        # --- FFT ---
        fft_clean = np.fft.fft(volts_windowed)
        freqs = np.fft.fftfreq(N, dt)
        
        # Keep positive frequencies only
        mask = freqs > 0
        freqs_mhz = freqs[mask] * 1e-6
        spectrum = np.abs(fft_clean[mask])

        spectrum /= np.max(spectrum)
        
        # Identify main beat peak
        peak_index = np.argmax(spectrum)
        peak_freq = freqs_mhz[peak_index]
        
        # Define region for fitting (2 MHz window around peak)
        window_mhz = 2
        region_mask = (freqs_mhz > peak_freq - window_mhz) & (freqs_mhz < peak_freq + window_mhz)
        x_fit = freqs_mhz[region_mask]
        x_smooth = np.linspace(x_fit[0], x_fit[-1], 1000)  # 500 points for a smooth curve
        y_fit = spectrum[region_mask]
        
        # Normalisation for numerical stability
        y_fit /= np.max(y_fit)
        
        # --- Initial Gaussian Guesses---
        A_guess = np.max(y_fit)
        x0_guess = peak_freq
        sigma_guess = 2.0  # MHz
        offset_guess = np.min(y_fit)
        
        try:
            popt, pcov = curve_fit(self.gauss_func, x_fit, y_fit,
                                p0=[A_guess, x0_guess, sigma_guess, offset_guess],
                                maxfev=10000)
        except RuntimeError:
            print(f"Gaussian fit did not converge for {self.filename}.")
            return None
        
        # Extract parameters
        A_fit, x0_fit, sigma_fit, offset_fit = popt
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_fit  # MHz

        # Errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        A_err, x0_err, sigma_err, offset_err = perr
            
        # Plotting
        if plot:
            plt.figure(figsize=(12, 8))
            labsize = 25
            ticksize = 20
            titlesize = 35
            lw = 2.0


            plt.plot(freqs_mhz, spectrum, 'y-', label='FFT Spectrum', linewidth = lw)
            plt.plot(x_smooth, self.gauss_func(x_smooth, *popt), 'b', 
                    label=(f'Gaussian Fit\nA = {A_fit:.4f} ± {A_err:.4f}\n'
                            f'Center = {x0_fit:.4f} ± {x0_err:.4f} MHz\n'
                            f'FWHM = {fwhm:.4f} ± {2*np.sqrt(2*np.log(2))*sigma_err:.4f} MHz'), 
                    linewidth=lw)
            plt.xlabel("Frequency (MHz)", size = labsize)
            plt.xticks(size = ticksize)
            plt.ylabel("Amplitude", size = labsize)
            plt.yticks(size = ticksize)
            plt.xlim([peak_freq - window_mhz/4, peak_freq + window_mhz/4])
            plt.title(f"Laser Linewidth for f = {peak_freq:.4f} MHz", size = titlesize)
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return {
        "center_frequency_MHz": x0_fit,
        "center_error_MHz": x0_err,
        "sigma_MHz": sigma_fit,
        "sigma_error_MHz": sigma_err,
        "FWHM_MHz": fwhm,
        "FWHM_error_MHz": 2 * np.sqrt(2*np.log(2)) * sigma_err,
        "offset": offset_fit,
        "offset_error": offset_err,
        "fit_params": popt,
        "fit_covariance": pcov
    }
    
    def lorentz_linewidth(self, plot = True):
        # Loading Oscilloscope Data
        time, volts_raw, volts_clean = self.oscilloscope_data()
        dt = time[1] - time[0]
        N = len(volts_clean)
    
        
        # # --- Apply Hanning window ---
        window = np.hanning(N)
        volts_windowed = volts_clean * window
        
        # # Normalize for energy correction (so window doesn’t affect amplitude scaling)
        correction_factor = np.sqrt(np.mean(window**2))
        volts_windowed /= correction_factor
        
        # --- FFT ---
        fft_clean = np.fft.fft(volts_windowed)
        freqs = np.fft.fftfreq(N, dt)
        
        # Keep positive frequencies only
        mask = freqs > 0
        freqs_mhz = freqs[mask] * 1e-6
        spectrum = np.abs(fft_clean[mask])

        spectrum /= np.max(spectrum)
        
        # Identify main beat peak
        peak_index = np.argmax(spectrum)
        peak_freq = freqs_mhz[peak_index]
        
        # Define region for fitting (2 MHz window around peak)
        window_mhz = 2
        region_mask = (freqs_mhz > peak_freq - window_mhz) & (freqs_mhz < peak_freq + window_mhz)
        x_fit = freqs_mhz[region_mask]
        x_smooth = np.linspace(x_fit[0], x_fit[-1], 1000)  # 500 points for a smooth curve
        y_fit = spectrum[region_mask]
        
        # Normalisation for numerical stability
        y_fit /= np.max(y_fit)

        # --- Initial Lorentz Guesses---
        A_guess = np.max(y_fit)
        x0_guess = peak_freq
        gamma_guess = 0.5  # MHz
        offset_guess = np.min(y_fit)
        
        try:
            popt, pcov = curve_fit(self.lorentz_func, x_fit, y_fit,
                                p0=[A_guess, x0_guess, gamma_guess, offset_guess],
                                maxfev=10000)
        except RuntimeError:
            print(f"Lorentzian fit did not converge for {self.filename}.")
            return None
        
        # Extract parameters
        A_fit, x0_fit, gamma_fit, offset_fit = popt
        fwhm = gamma_fit  # MHz

        # Errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        A_err, x0_err, gamma_err, offset_err = perr
            
        # Plotting
        if plot:
            plt.figure(figsize=(12, 8))
            labsize = 25
            ticksize = 20
            titlesize = 35
            lw = 2.0


            plt.plot(freqs_mhz, spectrum, 'y-', label='FFT Spectrum', linewidth = lw)
            plt.plot(x_smooth, self.lorentz_func(x_smooth, *popt), 'b', 
                    label=(f'Lorentzian Fit\nA = {A_fit:.4f} ± {A_err:.4f}\n'
                            f'Center = {x0_fit:.4f} ± {x0_err:.4f} MHz\n'
                            f'FWHM = {fwhm:.4f} ± {gamma_fit*gamma_err:.4f} MHz'), 
                    linewidth=lw)
            plt.xlabel("Frequency (MHz)", size = labsize)
            plt.xticks(size = ticksize)
            plt.ylabel("Amplitude", size = labsize)
            plt.yticks(size = ticksize)
            plt.xlim([peak_freq - window_mhz/4, peak_freq + window_mhz/4])
            plt.title(f"Laser Linewidth for f = {peak_freq:.4f} MHz", size = titlesize)
            plt.legend()
            plt.grid(False)
            plt.show()
        
        return {
        "center_frequency_MHz": x0_fit,
        "center_error_MHz": x0_err,
        "gamma_MHz": gamma_fit,
        "sigma_error_MHz": gamma_err,
        "FWHM_MHz": fwhm,
        "FWHM_error_MHz": gamma_fit * gamma_err,
        "offset": offset_fit,
        "offset_error": offset_err,
        "fit_params": popt,
        "fit_covariance": pcov
    }

    def fit_performance(self, x_data, y_data, popt, func):

        y_pred = func(x_data, *popt)
        ss_res = np.sum((y_data - y_pred)**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res/ss_tot)

        rmse = np.sqrt(np.mean((y_data - y_pred)**2))

        return {
        'r_squared': r_squared,
        'rmse': rmse,
        'max_residual': np.max(np.abs(y_data - y_pred))
    }

        

    def voigt_fit(self):
        pass


    def freq_comparison(self):

        time, volts = LaserLinewidth(self.filename).oscilloscope_data()

        dt = time[1]-time[0]
        rate = 1/dt

        # Signal Pre-Processing
        volts_clean = volts - np.mean(volts)

    def analyse_linewidth(self):

        # Load data
        time,volts,_ = LaserLinewidth(self.filename).oscilloscope_data()

        dt = time[1] - time[0]
        sampling_rate = 1.0/dt

        # Signal pre-processing
        volts_clean = volts - np.mean(volts)

        fft_raw = np.fft.fft(volts)
        freqs = np.fft.fftfreq(len(volts),dt)

        fft_clean = np.fft.fft(volts_clean)
        freq_clean = np.fft.fftfreq(len(volts_clean),dt)

        
        plt.figure(figsize=(12, 10))

        # Plot 1: Time domain comparison
        plt.subplot(3, 1, 1)
        plt.plot(time, volts, 'b-', label='Raw: DC + AC', linewidth=2)
        plt.plot(time, volts_clean, 'r-', label='Clean: AC only', linewidth=1)
        plt.ylabel('Voltage (mV)')
        plt.title('Time Domain: Before and After DC Removal')
        plt.legend()
        plt.grid(True)

        # Plot 2: FFT of raw signal (problem!)
        plt.subplot(3, 1, 2)
        fft_magnitude_raw = np.abs(fft_raw)[:len(freqs)//2]
        freqs_plot = freqs[:len(freqs)//2]  # Convert to MHz
        plt.plot(freqs_plot, fft_magnitude_raw, 'b-')
        plt.axvline(x=6, color='red', linestyle='--', label='6 MHz beat frequency')
        plt.ylabel('FFT Magnitude')
        plt.title('FFT of RAW Signal - DC Spike Dominates Everything')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale to see the problem

        # Plot 3: FFT of clean signal (solution!)
        plt.subplot(3, 1, 3)
        fft_magnitude_clean = np.abs(fft_clean)[:len(freqs)//2]
        plt.plot(freqs_plot, fft_magnitude_clean, 'r-')
        plt.axvline(x=6, color='red', linestyle='--', label='6 MHz beat frequency')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('FFT Magnitude')
        plt.title('FFT of CLEAN Signal - Beat Peak Clearly Visible')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        plt.tight_layout()
        plt.show()



        

        # window = np.hanning(len(volts_clean))
        # volts_windowed = volts_clean * window

        # fft_result = np.fft.fft(volts_clean)
        # frequencies = np.fft.fftfreq(len(volts_windowed),dt)

        # n = len(fft_result)
        # single_sided_freq = frequencies[:n//2]


        return freqs



filename_2mhz = 'MOT Thermometry Data/2 mhz_000.csv'
filename_4mhz = 'MOT Thermometry Data/4 mhz_000.csv'
filename_5mhz = 'MOT Thermometry Data/2 mhz_000.csv'
filename_6mhz = 'MOT Thermometry Data/6 mhz_000.csv'
filename_8mhz = 'MOT Thermometry Data/8 mhz_000.csv'
filename_10mhz = 'MOT Thermometry Data/10 mhz_000.csv'

class AtomSpectroscopy:

    def __init__(self, filename):
        self.filename = filename
    
    def oscilloscope_data(self, plot=True, figsize=(12, 8), find_peaks=True, peak_prominence=0.1):
        # Reading data with pandas, ensuring numeric conversion
        df = pd.read_csv(self.filename, skiprows=11, names=['TIME', 'CH1'])
        
        # Validate dataframe structure
        if 'TIME' not in df.columns or 'CH1' not in df.columns:
            print("Invalid CSV structure")
            return np.array([]), np.array([])
        
        # Convert to numeric and force errors to NaN
        time_series = pd.to_numeric(df['TIME'], errors='coerce')
        volts_series = pd.to_numeric(df['CH1'], errors='coerce')
        
        # Checking for NaN points
        if time_series.isna().all() or volts_series.isna().all():
            print("Numeric conversion failed for all data")
            return np.array([]), np.array([])
        
        time = time_series.values
        volts = volts_series.values
       
        # Remove any NaN values present
        valid_mask = ~(np.isnan(time) | np.isnan(volts))
        time = time[valid_mask]
        volts = volts[valid_mask]

        volts_raw = volts
        volts_clean = volts_raw - np.mean(volts_raw)

        # Final Validation
        if len(time) == 0 or len(volts) == 0:
            print("No valid data found after filtering")
            return np.array([]), np.array([])
        
        assert len(time) == len(volts)
        
        # Find peaks in the data
        peaks_info = {}
        if find_peaks:
            from scipy.signal import find_peaks
            
            # Find peaks (both positive and negative)
            positive_peaks, _ = find_peaks(volts_clean, prominence=peak_prominence)
            negative_peaks, _ = find_peaks(-volts_clean, prominence=peak_prominence)
            
            peaks_info = {
                'positive_peaks': positive_peaks,
                'negative_peaks': negative_peaks,
                'peak_times': time[positive_peaks],
                'peak_voltages': volts_clean[positive_peaks],
                'dip_times': time[negative_peaks],
                'dip_voltages': volts_clean[negative_peaks]
            }
        
        if plot:
            plt.figure(figsize=figsize)
            plt.grid(True, alpha=0.3)

            title_size = 16
            lab_size = 14
            legend_size = 12
            xtick_size = ytick_size = 12

            # Plot the main signal
            plt.plot(time * 1e3, volts_clean * 1e3, 'b-', label='Polarization Signal', linewidth=1.5, alpha=0.8)
            
            # Mark peaks if found
            if find_peaks and len(peaks_info['positive_peaks']) > 0:
                plt.plot(peaks_info['peak_times'] * 1e3, peaks_info['peak_voltages'] * 1e3, 
                        'ro', markersize=6, label=f'Peaks ({len(peaks_info["positive_peaks"])} found)', zorder=5)
            
            if find_peaks and len(peaks_info['negative_peaks']) > 0:
                plt.plot(peaks_info['dip_times'] * 1e3, peaks_info['dip_voltages'] * 1e3, 
                        'go', markersize=6, label=f'Dips ({len(peaks_info["negative_peaks"])} found)', zorder=5)

            plt.title('Polarization Spectroscopy Signal', size=title_size)
            plt.xlabel('Time (ms)', size=lab_size)
            plt.ylabel('Voltage (mV)', size=lab_size)
            plt.xticks(size=xtick_size)
            plt.yticks(size=ytick_size)
            plt.legend(fontsize=legend_size)
            
            # Add some statistics to the plot
            if find_peaks:
                stats_text = f"Total peaks: {len(peaks_info['positive_peaks'])}\n"
                stats_text += f"Total dips: {len(peaks_info['negative_peaks'])}"
                plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                           verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.show()

        return {
            "time_data": time, 
            "voltage_data": volts_clean,
            "peaks_info": peaks_info
        }
    
    def enhanced_pol_spectroscopy(self, plot=True, figsize=(15, 10), smooth_data=True, window_size=51):
        """
        Enhanced polarization spectroscopy analysis with multiple visualization options
        """
        data = self.oscilloscope_data(plot=False, find_peaks=True)
        time = data['time_data']
        voltage = data['voltage_data']
        peaks_info = data['peaks_info']
        
        if smooth_data:
            from scipy.signal import savgol_filter
            voltage_smooth = savgol_filter(voltage, window_size, 3)
        else:
            voltage_smooth = voltage
        
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Plot 1: Raw signal with peaks
            # axes[0,0].plot(time * 1e3, voltage * 1e3, 'b-', alpha=0.7, linewidth=1, label='Raw Signal')
            if smooth_data:
                axes[0,0].plot(time * 1e3, voltage_smooth * 1e3, 'r-', linewidth=1.5, label='Smoothed')
            
            if len(peaks_info['positive_peaks']) > 0:
                axes[0,0].plot(peaks_info['peak_times'] * 1e3, peaks_info['peak_voltages'] * 1e3, 
                              'ro', markersize=5, label='Peaks')
            if len(peaks_info['negative_peaks']) > 0:
                axes[0,0].plot(peaks_info['dip_times'] * 1e3, peaks_info['dip_voltages'] * 1e3, 
                              'go', markersize=5, label='Dips')
            
            axes[0,0].set_xlabel('Time (ms)')
            axes[0,0].set_ylabel('Voltage (mV)')
            axes[0,0].set_title('Polarization Signal with Peaks')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Derivative (shows slope changes)
            derivative = np.gradient(voltage_smooth, time)
            axes[0,1].plot(time * 1e3, derivative * 1e3, 'purple', linewidth=1.5)
            axes[0,1].set_xlabel('Time (ms)')
            axes[0,1].set_ylabel('dV/dt (mV/ms)')
            axes[0,1].set_title('Signal Derivative')
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Histogram of peak voltages
            if len(peaks_info['peak_voltages']) > 0:
                axes[1,0].hist(peaks_info['peak_voltages'] * 1e3, bins=20, alpha=0.7, color='red', edgecolor='black')
                axes[1,0].set_xlabel('Peak Voltage (mV)')
                axes[1,0].set_ylabel('Count')
                axes[1,0].set_title('Peak Voltage Distribution')
                axes[1,0].grid(True, alpha=0.3)
            
            # Plot 4: Peak statistics
            axes[1,1].axis('off')
            if len(peaks_info['positive_peaks']) > 0:
                stats_text = f"POLARIZATION SPECTROSCOPY ANALYSIS\n"
                stats_text += f"Total peaks: {len(peaks_info['positive_peaks'])}\n"
                stats_text += f"Total dips: {len(peaks_info['negative_peaks'])}\n"
                stats_text += f"Max peak: {np.max(peaks_info['peak_voltages'])*1e3:.2f} mV\n"
                stats_text += f"Min dip: {np.min(peaks_info['dip_voltages'])*1e3:.2f} mV\n"
                stats_text += f"Peak-to-peak: {np.ptp(voltage)*1e3:.2f} mV\n"
                stats_text += f"Signal RMS: {np.std(voltage)*1e3:.2f} mV"
                
                axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                              fontsize=12, verticalalignment='top', 
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
        
        return {
            "time_data": time,
            "voltage_data": voltage,
            "voltage_smooth": voltage_smooth if smooth_data else voltage,
            "peaks_info": peaks_info,
            "signal_stats": {
                "rms_voltage": np.std(voltage),
                "peak_to_peak": np.ptp(voltage),
                "mean_voltage": np.mean(voltage),
                "num_peaks": len(peaks_info['positive_peaks']),
                "num_dips": len(peaks_info['negative_peaks'])
            }
        }
    
    def sa_spectroscopy(self, plot=True):
        pass
    
    def pol_spectroscopy(self, plot=True):
        # You can make this call the enhanced version
        return self.enhanced_pol_spectroscopy(plot=plot)


# Usage
pol_spec = 'MOT Thermometry Data/pol spec_000.csv'
atom_spec = AtomSpectroscopy(pol_spec)

# Basic view with peaks
data_basic = atom_spec.oscilloscope_data(plot=True, find_peaks=True, peak_prominence=0.50)

# Enhanced analysis
data_enhanced = atom_spec.enhanced_pol_spectroscopy(plot=True, smooth_data=True)

class LaserMeasurement:

    def __init__(self, filename):
        self.filename = filename

    def oscilloscope_data(self, plot=True, figsize=(12,8)):
        # Reading CSV data
        df = pd.read_csv(self.filename, skiprows=11, names=['TIME','CH1'])
        
        # Validate columns
        if 'TIME' not in df.columns or 'CH1' not in df.columns:
            print("Invalid CSV structure")
            return np.array([]), np.array([])

        # Numeric conversion
        time_series = pd.to_numeric(df['TIME'], errors='coerce')
        volts_series = pd.to_numeric(df['CH1'], errors='coerce')

        # Remove NaNs
        valid_mask = ~(np.isnan(time_series) | np.isnan(volts_series))
        time = time_series.values[valid_mask]
        volts_raw = volts_series.values[valid_mask]

        volts_clean = volts_raw - np.mean(volts_raw)

        if len(time) == 0:
            print("No valid data found")
            return np.array([]), np.array([])

        # Determine beat frequency via FFT
        dt = time[1] - time[0]
        N = len(volts_clean)
        fft_data = np.fft.fft(volts_clean)
        freqs = np.fft.fftfreq(N, dt)
        mask = freqs > 0
        freqs_mhz = freqs[mask] * 1e-6
        spectrum = np.abs(fft_data[mask])
        spectrum /= np.max(spectrum)
        peak_freq = freqs_mhz[np.argmax(spectrum)]

        if plot:
            plt.figure(figsize=figsize)
            plt.plot(time*1e6, volts_clean*1e3, 'b-', linewidth=2.5)
            plt.xlabel(r"time, $\mu$s", fontsize=25)
            plt.ylabel("Voltage, mV", fontsize=25)
            plt.grid(True)
            plt.xlim([-1,1])
            plt.show()

        return {"time_data": time, "voltage_data": volts_clean}

    def spectrum_data(self, plot=True, figsize=(12,12), zoom=0.5):
        osci_data = self.oscilloscope_data(plot=False)
        time = osci_data['time_data']
        volts = osci_data['voltage_data']

        dt = time[1] - time[0]
        N = len(volts)
        fft_data = np.fft.fft(volts)
        freqs_data = np.fft.fftfreq(N, dt)

        mask = freqs_data > 0
        freqs_mhz = freqs_data[mask] * 1e-6
        spectrum_data = np.abs(fft_data[mask])
        spectrum_data /= np.max(spectrum_data)
        power_db = 10 * np.log10(spectrum_data**2)

        peak_idx = np.argmax(spectrum_data)
        peak_freq = freqs_mhz[peak_idx]

        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=figsize)
            ax1.plot(freqs_mhz, spectrum_data, 'b', linewidth=3)
            ax1.set_ylabel("Normalized\nAmplitude", fontsize=23)
            ax1.set_xlim([0,50])
            ax1.grid(True)

            zoom_mask = (freqs_mhz > peak_freq - zoom) & (freqs_mhz < peak_freq + zoom)
            ax2.plot(freqs_mhz[zoom_mask], spectrum_data[zoom_mask], 'r', linewidth=2)
            ax2.axvline(peak_freq, color='blue', linestyle='--', alpha=0.7, 
                        label=f'Peak: {peak_freq:.3f} MHz')
            ax2.set_ylabel("Normalized\nAmplitude", fontsize=23)
            ax2.legend(fontsize=16)
            ax2.grid(True, alpha=0.3)

            ax3.plot(freqs_mhz, power_db, 'g', linewidth=2)
            ax3.axvline(peak_freq, color='red', linestyle='--', alpha=0.7,
                        label=f'Peak: {peak_freq:.3f} MHz')
            ax3.set_xlabel("Frequency (MHz)", fontsize=23)
            ax3.set_ylabel("Power (dB)", fontsize=23)
            ax3.legend(fontsize=16)
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return {
            "frequency_data_MHz": freqs_mhz,
            "fft_spectrum_data": spectrum_data,
            "beat_frequency": peak_freq
        }

    def linewidth(self, fit='gaussian', plot=True, figsize=(12,8), freq_window=0.5, sigma_g=0.01, gamma_g=0.01):
        
        def gaussian(x, A, x0, sigma, offset):
            return A * np.exp(-(x-x0)**2/(2*sigma**2)) + offset

        def lorentzian(x, A, x0, gamma, offset):
            return A / (1 + ((x-x0)/(gamma/2))**2) + offset

        def voigt(x, A, x0, sigma, gamma, offset):
            z = ((x-x0)+1j*gamma)/(sigma*np.sqrt(2))
            return A*np.real(sp.wofz(z))/(sigma*np.sqrt(2*np.pi)) + offset

        osci_data = self.oscilloscope_data(plot=False)
        time, volts = osci_data['time_data'], osci_data['voltage_data']

        dt = time[1]-time[0]
        N = len(volts)
        window = np.hanning(N)
        volts_windowed = volts*window
        volts_windowed /= np.sqrt(np.mean(window**2))
        fft_data = np.fft.fft(volts_windowed)
        freqs = np.fft.fftfreq(N, dt)
        mask = freqs>0
        freqs_mhz = freqs[mask]*1e-6
        spectrum = np.abs(fft_data[mask])
        spectrum /= np.max(spectrum)
        peak_freq = freqs_mhz[np.argmax(spectrum)]

        region_mask = (freqs_mhz>peak_freq-freq_window) & (freqs_mhz<peak_freq+freq_window)
        x_fit = freqs_mhz[region_mask]
        y_fit = spectrum[region_mask]/np.max(spectrum[region_mask])
        x_smooth = np.linspace(x_fit[0], x_fit[-1], 5000)

        A_g = np.max(y_fit)
        offset_g = np.min(y_fit)
        x0_g = peak_freq

        lw = 2.5
        if fit=='gaussian':
            try:
                popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[A_g, x0_g, sigma_g, offset_g], maxfev=10000)
            except RuntimeError:
                print("Gaussian fit did not converge.")
                return None

            fwhm = 2*np.sqrt(2*np.log(2))*popt[2]
            perr = np.sqrt(np.diag(pcov))

            if plot:
                plt.figure(figsize=figsize)
                plt.plot(freqs_mhz, spectrum, 'r--', label='Spectrum', linewidth=lw)
                plt.plot(x_smooth, gaussian(x_smooth,*popt), 'b', 
                        label=f'Gaussian Fit\nPeak={popt[1]:.4f} MHz\nFWHM={1000*fwhm:.4f} kHz', linewidth=lw)
                plt.xlabel("Frequency (MHz)", fontsize=25)
                plt.ylabel("Amplitude", fontsize=25)
                plt.grid(True)
                plt.legend(fontsize=16)
                plt.show()

            return {"freqs_mhz": freqs_mhz, "spectrum": spectrum, "fit_curve": gaussian(x_smooth,*popt), 
                    "central_frequency": popt[1], "fwhm": 1000*fwhm, "fit_params": popt, "fit_cov": pcov}

        elif fit=='lorentzian':
            try:
                popt, pcov = curve_fit(lorentzian, x_fit, y_fit, p0=[A_g, x0_g, gamma_g, offset_g], maxfev=10000)
            except RuntimeError:
                print("Lorentzian fit did not converge.")
                return None

            fwhm = popt[2]
            perr = np.sqrt(np.diag(pcov))

            if plot:
                plt.figure(figsize=figsize)
                plt.plot(freqs_mhz, spectrum, 'r--', label='Spectrum', linewidth=lw)
                plt.plot(x_smooth, lorentzian(x_smooth,*popt), 'b', 
                        label=f'Lorentzian Fit\nPeak={popt[1]:.4f} MHz\nFWHM={1000*fwhm:.4f} kHz', linewidth=lw)
                plt.xlabel("Frequency (MHz)", fontsize=25)
                plt.xlim([5.5,6.5])
                plt.ylabel("Amplitude", fontsize=25)
                plt.grid(True)
                plt.legend(fontsize=16)
                plt.show()

            return {"freqs_mhz": freqs_mhz, "spectrum": spectrum, "fit_curve": lorentzian(x_smooth,*popt),
                    "central_frequency": popt[1], "fwhm": 1000*fwhm, "fit_params": popt, "fit_cov": pcov}

        elif fit=='voigt':
            try:
                popt, pcov = curve_fit(voigt, x_fit, y_fit, p0=[A_g, x0_g, sigma_g, gamma_g, offset_g], maxfev=10000)
            except RuntimeError:
                print("Voigt fit did not converge.")
                return None

            sigma_f, gamma_f = popt[2], popt[3]
            fwhm_v = 0.5346*2*gamma_f + np.sqrt(0.2166*(2*gamma_f)**2 + (2*np.sqrt(2*np.log(2))*sigma_f)**2)
            perr_v = np.sqrt(np.diag(pcov))

            if plot:
                plt.figure(figsize=figsize)
                plt.plot(freqs_mhz, spectrum, 'r--', label='Spectrum', linewidth=lw)
                plt.plot(x_smooth, voigt(x_smooth,*popt), 'b',
                        label=f'Voigt Fit\nPeak={popt[1]:.4f} MHz\nFWHM={1000*fwhm_v:.4f} kHz', linewidth=lw)
                plt.xlabel("Frequency (MHz)", fontsize=25)
                plt.ylabel("Amplitude", fontsize=25)
                plt.grid(True)
                plt.legend(fontsize=16)
                plt.show()

            return {"freqs_mhz": freqs_mhz, "spectrum": spectrum, "fit_curve": voigt(x_smooth,*popt),
                    "central_frequency": popt[1], "fwhm": 1000*fwhm_v, "fit_params": popt, "fit_cov": pcov}

        else:
            print("Invalid fit type. Choose 'gaussian', 'lorentzian' or 'voigt'.")
            return None
        

class AtomSpectroscopy:
    
    def __init__(self, dop_file, sas_file,pol_file):
        self.dop_file = dop_file
        self.sas_file = sas_file
        self.pol_file = pol_file
    
    def oscilloscope_data(self, plot=True, figsize=(12, 8), smooth = True, window_size = 51):
        # Read CSV and validate
        df1 = pd.read_csv(self.dop_file, skiprows=11, names=['TIME', 'CH1'])
        df2 = pd.read_csv(self.sas_file, skiprows=11, names=['TIME', 'CH1'])
        df3 = pd.read_csv(self.pol_file, skiprows=11, names=['TIME', 'CH1'])

        if 'TIME' not in df1.columns or 'CH1' not in df1.columns:
            raise ValueError("Invalid CSV structure: missing TIME or CH1 columns")
        if 'TIME' not in df2.columns or 'CH1' not in df2.columns:
            raise ValueError("Invalid CSV structure: missing TIME or CH1 columns")
        if 'TIME' not in df3.columns or 'CH1' not in df3.columns:
            raise ValueError("Invalid CSV structure: missing TIME or CH1 columns")

        # Convert to numeric and clean NaNs
        time = pd.to_numeric(df1['TIME'], errors='coerce').dropna().values
        dop_volts = pd.to_numeric(df1['CH1'], errors='coerce').dropna().values
        sas_volts = pd.to_numeric(df2['CH1'], errors='coerce').dropna().values
        pol_volts = pd.to_numeric(df3['CH1'], errors='coerce').dropna().values

        if smooth:
            pol_volts_smooth = savgol_filter(pol_volts, window_size, 3)
        else:
            pol_volts_smooth = pol_volts

        if len(time) == 0 or len(pol_volts) == 0 or len(sas_volts) == 0 or len(dop_volts) == 0:
            raise ValueError("No valid data found after cleaning")

        
        if plot:

            ticksize = 20
            labsize = 20
            lw = 0.1
            lower = -0.50
            upper = +0.50

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

            ax1.plot(time * 1.00e3 , dop_volts * 1.00e3 - 80 , 'brown', label = "Doppler", linewidth = lw) 
            # ax1.axhline(y=0.5, color='red', linestyle='--', label='Half Maximum')
            ax1.legend(loc='upper right', fontsize=12)
            ax1.set_ylabel("Voltage, mV", size = labsize)
            ax1.tick_params(axis='both', which='major', labelsize=ticksize)
            ax1.set_xlim([lower,upper])
            ax1.legend(fontsize = 20) #, loc='upper right', bbox_to_anchor=(1, 1))
            ax1.grid(True, alpha = 0.15)

            ax2.plot(time * 1.00e3, sas_volts * 1.00e3-80, 'r', label='SAS', linewidth=lw)
            ax2.plot(time * 1.00e3, pol_volts_smooth * 1.00e3, 'b', label='PS', linewidth=lw)
            # ax2.axhline(y=0.5, color='red', linestyle='--', label='Half Maximum')
            ax2.set_ylabel("Voltage, mV", size = labsize)
            ax2.set_xlabel(fr"Time, $\mu$s", size = labsize)
            ax2.tick_params(axis='both', which='major', labelsize=ticksize)
            ax2.set_xlim([lower,upper])
            ax2.legend(fontsize = 20) #, loc='upper right', bbox_to_anchor=(1, 1))
            ax2.grid(True, alpha = 0.15)

            # change the line width for the legend
            for line in ax2.legend().get_lines():
                line.set_linewidth(2.0)



            # if doppler:
            #     plt.figure(figsize=figsize)
            #     plt.plot(time * 1e3, dop_volts * 1e3 - 80, 'r-', label='Doppler', linewidth=lw)
            #     # plt.title('Oscilloscope Data for Polarisation Spectroscopy', fontsize=24)
            #     plt.xlabel('Time (µs)', fontsize=labsize)
            #     plt.ylabel('Voltage (mV)', fontsize=labsize)
            #     plt.xticks(size = ticksize)
            #     plt.yticks(size = ticksize)
            #     plt.grid(True, alpha = 0.2)
            #     plt.legend(fontsize=labsize)
            #     # plt.ylim([-20, 200])
            #     plt.xlim([-0.50, +0.50])
            #     plt.tight_layout()
            #     plt.show()
            # else:
            #     plt.figure(figsize=figsize)
            #     plt.plot(time * 1e3, sas_volts * 1e3-80, 'r', label='SAS', linewidth=lw)
            #     plt.plot(time * 1e3, pol_volts * 1e3, 'b', label='PS', linewidth=lw)
            #     # plt.title('Oscilloscope Data for Polarisation Spectroscopy', fontsize=24)
            #     plt.xlabel('Time (µs)', fontsize=labsize)
            #     plt.ylabel('Voltage (mV)', fontsize=labsize)
            #     plt.xticks(size = ticksize)
            #     plt.yticks(size = ticksize)
            #     plt.grid(True, alpha = 0.2)
            #     plt.legend(fontsize=labsize)
            #     plt.xlim([-0.50, +0.50])
            #     plt.tight_layout()
            #     plt.show()


        return {
                "time_data": time,
                "doppler_data": dop_volts,
                "sas_data": sas_volts,
                "ps_data": pol_volts_smooth
                
        }


# Example usage:
dop = 'data/atom_spectroscopy/dop_000.csv'
sas = 'data/atom_spectroscopy/sas_000.csv'
pol = 'data/atom_spectroscopy/pol_000.csv'
spec = AtomSpectroscopy(dop,sas,pol)
data = spec.oscilloscope_data(plot=True, smooth=True, figsize = (10,8))
