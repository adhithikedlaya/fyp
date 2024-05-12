import numpy as np
import torch
from torch import linalg as la
from torch.fft import fft, rfft, fftfreq, rfftfreq

def sliding_window_view(x, window_shape, axis=-1):
    if axis < 0:
        axis += x.dim()  # Handle negative axis
    step = x.stride(axis)
    new_shape = list(x.shape)
    new_shape[axis] = 1 + (new_shape[axis] - window_shape) // step
    unfolded = x.unfold(axis, window_shape, step)
    result = unfolded.reshape(new_shape + [window_shape])
    # Select every step-th element along the second axis
    result = result[..., 0::step, :]
    return result

def _median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.

    See Appendix B from [1]_ for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.

    References
    ----------
    .. [1] B. Allen, W.G. Anderson, P.R. Brady, D.A. Brown, J.D.E. Creighton.
           "FINDCHIRP: an algorithm for detection of gravitational waves from
           inspiraling compact binaries", Physical Review D 85, 2012,
           :arxiv:`gr-qc/0509116`
    """
    ii_2 = 2 * torch.arange(1., (n-1) // 2 + 1)
    return 1 + torch.sum(1. / (ii_2 + 1) - 1. / ii_2)

def detrend1(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> npoints = 1000
    >>> noise = rng.standard_normal(npoints)
    >>> x = 3 + 2*np.linspace(0, 1, npoints) + noise
    >>> (signal.detrend(x) - noise).max()
    0.06  # random

    """
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = torch.as_tensor(data)
    dtype = data.dtype
    if type in ['constant', 'c']:
        ret = data - torch.mean(data, axis, keepdim=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = torch.sort(torch.unique(torch.cat((torch.tensor(0), bp, torch.tensor(N)))))
        if torch.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")

        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdata = data.permute(axis, *[i for i in range(rnk) if i != axis]).reshape(N, -1)

        if not overwrite_data:
            newdata = newdata.clone()  # make sure we have a copy
        if newdata.dtype not in [torch.float32, torch.float64]:
            newdata = newdata.to(dtype)

        for m in range(len(bp) - 1):
            Npts = bp[m + 1] - bp[m]
            A = torch.ones((Npts, 2), dtype=dtype, device=data.device)
            A[:, 0] = torch.arange(1, Npts + 1, dtype=dtype, device=data.device) / Npts
            sl = slice(bp[m].item(), bp[m + 1].item())
            coef, _ = la.lstsq(A, newdata[sl].unsqueeze(-1))
            newdata[sl] = newdata[sl] - torch.matmul(A, coef).squeeze(-1)

        newdata = newdata.view(dshape)
        ret = newdata.permute(*[i for i in range(rnk) if i != axis], axis)
        return ret


def _lombscargle(x, y, freqs):
    """
    _lombscargle(x, y, freqs)

    Computes the Lomb-Scargle periodogram.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values (must be registered so the mean is zero).
    freqs : array_like
        Angular frequencies for output periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    See also
    --------
    lombscargle

    """

    if x.shape != y.shape:
        raise ValueError("Input arrays do not have the same size.")

    pgram = torch.empty_like(freqs)

    c = torch.empty_like(x)
    s = torch.empty_like(x)

    for i in range(freqs.shape[0]):

        xc = 0.
        xs = 0.
        cc = 0.
        ss = 0.
        cs = 0.

        c[:] = torch.cos(freqs[i] * x)
        s[:] = torch.sin(freqs[i] * x)

        for j in range(x.shape[0]):
            xc += y[j] * c[j]
            xs += y[j] * s[j]
            cc += c[j] * c[j]
            ss += s[j] * s[j]
            cs += c[j] * s[j]

        if freqs[i] == 0:
            raise ZeroDivisionError()

        tau = torch.atan2(2 * cs, cc - ss) / (2 * freqs[i])
        c_tau = torch.cos(freqs[i] * tau)
        s_tau = torch.sin(freqs[i] * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2 * c_tau * s_tau

        pgram[i] = 0.5 * (((c_tau * xc + s_tau * xs)**2 /
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) +
            ((c_tau * xs - s_tau * xc)**2 /
             (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))

    return pgram

def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    """
    Calculate windowed FFT, for internal use by
    `scipy.signal._spectral_helper`.

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # Created sliding window view of array
    if nperseg == 1 and noverlap == 0:
        result = x[..., None]
    else:
        step = nperseg - noverlap
        result = sliding_window_view(
            x, window_shape=nperseg, axis=-1,
        )
        result = result[..., 0::step, :]



    result = detrend_func(result)

    result = win * result

    if sides == 'twosided':
        func = fft
    else:
        result = result.real
        func = rfft
    result = func(result, n=nfft)

    return result

def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.

    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.

    nperseg : int
        Length of each segment

    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.

    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.

    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            print(f'nperseg = {nperseg:d} is greater than input length '
                          f' = {input_length:d}, using nperseg = {input_length:d}')
            nperseg = input_length
        win = torch.hamming_window(nperseg)
    else:
        win = torch.as_tensor(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    return win, nperseg

def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False):
    """Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    if mode not in ['psd', 'stft']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "{'psd', 'stft'}" % mode)

    boundary_funcs = {
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{}', must be one of: {}"
                         .format(boundary, list(boundary_funcs.keys())))

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have torch tensors, get outdtype
    x = torch.as_tensor(x, dtype=torch.cfloat)
    if not same_data:
        y = torch.as_tensor(y,  dtype=torch.cfloat)
        outdtype = torch.result_type(x, y)
    else:
        outdtype = torch.result_type(x, x)
    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = torch.broadcast_tensors(torch.empty(xouter), torch.empty(youter))[0].shape
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e

    if same_data:
        if x.numel() == 0:
            return torch.empty(x.shape), torch.empty(x.shape), torch.empty(x.shape)
    else:
        if x.numel() == 0 or y.numel() == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = torch.empty(outshape).moveaxis(-1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = x.moveaxis(axis, -1)
            if not same_data and y.ndim > 1:
                y = y.moveaxis(axis, -1)

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = torch.cat((x, torch.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = torch.cat((y, torch.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg//2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg//2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = torch.cat((x, torch.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = torch.cat((y, torch.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return detrend1(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = d.moveaxis(-1, axis)
            d = detrend1(d)
            return d.moveaxis(axis, -1)
    else:
        detrend_func = detrend

    # if torch.result_type(win) != outdtype:
    #     win = win.to(outdtype)

    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    if mode == 'stft':
        scale = scale.sqrt()

    if return_onesided:
        if torch.is_complex(x):
            sides = 'twosided'
            print('Input data is complex, switching to return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if torch.is_complex(y):
                    sides = 'twosided'
                    print('Input data is complex, switching to '
                                  'return_onesided=False')
    else:
        sides = 'twosided'

    if sides == 'twosided':
        freqs = fftfreq(nfft, 1/fs)
    elif sides == 'onesided':
        freqs = rfftfreq(nfft, 1/fs)
    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)
    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
                               sides)
        result = torch.conj(result) * result_y
    elif mode == 'psd':
        result = torch.conj(result) * result

    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2
    time = torch.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                        nperseg - noverlap)/float(fs)
    if boundary is not None:
        time -= (nperseg/2) / fs
    result = result.to(outdtype)
    # All imaginary parts are zero anyways
    if same_data and mode != 'stft':
        result = result.real
        result = result.to(outdtype)

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = result.moveaxis(-1, axis)
    return freqs, time, result

def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density',
        axis=-1, average='mean'):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's method.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    axis : int, optional
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. If the spectrum is
        complex, the average is computed separately for the real and
        imaginary parts. Defaults to 'mean'.

        .. versionadded:: 1.2.0

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross spectral density or cross power spectrum of x,y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method. [Equivalent to
           csd(x,x)]
    coherence: Magnitude squared coherence by Welch's method.

    Notes
    -----
    By convention, Pxy is computed with the conjugate FFT of X
    multiplied by the FFT of Y.

    If the input series differ in length, the shorter series will be
    zero-padded to match.

    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    Consult the :ref:`tutorial_SpectralAnalysis` section of the :ref:`user_guide`
    for a discussion of the scalings of a spectral density and an (amplitude) spectrum.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Generate two test signals with some common features.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> b, a = signal.butter(2, 0.25, 'low')
    >>> x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> y = signal.lfilter(b, a, x)
    >>> x += amp*np.sin(2*np.pi*freq*time)
    >>> y += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)

    Compute and plot the magnitude of the cross spectral density.

    >>> f, Pxy = signal.csd(x, y, fs, nperseg=1024)
    >>> plt.semilogy(f, np.abs(Pxy))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('CSD [V**2/Hz]')
    >>> plt.show()

    """
    
    freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap,
                                     nfft, detrend, return_onesided, scaling,
                                     axis, mode='psd')
    
        # Average over windows.
    if len(Pxy.shape) >= 2 and Pxy.numel() > 0:
        if Pxy.shape[-1] > 1:
            if average == 'median':
                # torch.median must be passed real tensors for the desired result
                bias = _median_bias(Pxy.shape[-1])
                if torch.is_complex(Pxy):
                    Pxy = (torch.median(Pxy.real, dim=-1).values
                           + 1j * torch.median(Pxy.imag, dim=-1).values)
                else:
                    Pxy = torch.median(Pxy, dim=-1).values
                Pxy /= bias
            elif average == 'mean':
                Pxy = Pxy.mean(dim=-1)
            else:
                raise ValueError(f'average must be "median" or "mean", got {average}')
        else:
            Pxy = Pxy.view(*Pxy.shape[:-1])

    return freqs, Pxy


