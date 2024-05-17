import torch

class StimulusPDCMBOLD:
    """Generate the stimulus and the P-DCM equations and ODEs to calculate the
       BOLD response

    Args:
    - All constants involved in the equations and ODEs (details and default
      values see below)
    - ignore_range -- Ignore plausible range checks
    - cross_valid -- Set to True when running CrossValidation.py because mu
                     was outside the plausible range for the plots generated
                     in Havlicek et. al.'s paper
    """

    def __init__(self, w=1.0,
                 sigma=0.5, mu=0.4, lamb=0.2, c=0.25,
                 psi=0.6, phi=1.5, chi=0.6,
                 tMTT=2.0, tau=4.0, alpha=0.32, E0=0.4,
                 V0=4,
                 epsilon=0.3, theta0=80.6, r0=108.0, TE=35.0,
                 ignore_range=False,
                 cross_valid=False
                 ):

        self._w = w  # stimulus duration time, sec

        if not ignore_range:
            if sigma < 0.1 or sigma > 1.5:
                raise ValueError('sigma out of plausible range (0.1-1.5)')
            if mu < 0 or mu > 1.5:  # do not check if running cross validation
                if cross_valid is False:
                    raise ValueError('mu out of plausible range (0-1.5)')
                else:
                    pass
            if lamb < 0 or lamb > 0.3:
                raise ValueError('lamb out of plausible range (0-0.3)')
            if tMTT < 1 or tMTT > 5:
                raise ValueError('tMTT out of plausible range (1-5)')
            if tau < 0 or tau > 30:
                raise ValueError('tau out of plausible range (0-30)')
            if epsilon < 0.1291 or epsilon > 0.5648:
                raise ValueError('epsilon out of plausible range (0.1291-0.5648)')
            if TE < 30 or TE > 45:
                raise ValueError('TE out of plausible range (30-45)')

        # Constants (name, unit, plausible range if applicable)
        self._sigma = sigma  # excitatory self-connection, Hz, 0.1-1.5
        self._mu = mu  # inhibitory-excitatory connection, Hz, 0-1.5
        self._lamb = lamb  # inhibitory fain factor, Hz, 0-0.3
        self._c = c  # scaling factor, AU

        self._psi = psi  # decay of vasoactive signal, Hz
        self._phi = phi  # gain of vasoactive signal, Hz
        self._chi = chi  # decay of blood inflow signal, Hz

        self._tMTT = tMTT  # mean transit time, sec, 1-5
        self._tau = tau  # viscoelastic time, sec, 0-30
        self._alpha = alpha  # Grubb's exponent, AU
        self._E0 = E0  # O2 extraction at rest, AU

        self._V0 = V0  # venous blood volume fraction, %

        self._epsilon = epsilon  # ratio of in- to extra-vascular contribution,
                                 # AU, 0.1291-0.5648
        self._theta0 = theta0  # field-dependent frequency offset
        self._r0 = r0  # sensitivity
        self._TE = TE/1000  # echo time, ms, 30-45


    # Stimulus
    def sti_u(self, t):
        """Function of the input stimulus, u(t), which
        - varies with time, t (unit: second)
        - is a rectangular function starting at t = 1s with width, w, by
          default
        """
        return torch.where((t >= 1) & (t <= (self._w + 1)), torch.tensor(1.0), torch.tensor(0.0))

    # Equations and ODEs in P-DCM
    def ode_xE(self, u, xE, xI):
        """The d x_E(t)/dt ODE from Neuronal model

        Args:
        - u -- u(t), input stimulus
        - xE -- x_E(t), excitatory neuronal states
        - xI -- x_I(t), inhibitory neuronal states
        """
        return -self._sigma*xE - self._mu*xI + self._c*u

    def ode_xI(self, xE, xI):
        """The d x_I(t)/dt ODE from Neuronal model

        Args:
        - xE -- x_E(t), excitatory neuronal states
        - xI -- x_I(t), inhibitory neuronal states
        """
        return self._lamb*(xE - xI)

    def ode_a(self, a, xE):
        """The d a(t)/dt ODE from Feedforward neurovascular coupling (P-DCM)

        Args:
        - a -- vasoactive signal
        - xE -- x_E(t), excitatory neuronal states
        """
        return -self._psi*a + xE

    def ode_f(self, a, f):
        """The d f(t)/dt ODE from Feedforward neurovascular coupling (P-DCM)

        Args:
        - a -- vasoactive signal
        - f -- blood inflow response
        """
        return self._phi*a - self._chi*(f - 1)

    def ode_v(self, f, fout):
        """The d v(t)/dt ODE from Hemodynamic model

        Args:
        - f -- blood inflow response
        - fout -- blood outflow response
        """
        return (f - fout)/self._tMTT

    def ode_q(self, f, E, fout, q, v):
        """The d q(t)/dt ODE from Hemodynamic model

        Args:
        - f -- blood inflow response
        - E -- O2 extraction fraction
        - fout -- blood outflow response
        - q -- deoxyhemoglobin content
        - v -- blod volume
        """
        # print(self._tMTT)
        return (f*(E/self._E0) - fout*(q/v))/self._tMTT

    def equ_E(self, f):
        """The E(f) equation from Hemodynamic model

        Args:
        - f -- blood inflow response
        """
        return 1 - (1 - self._E0)**(1/f)

    def equ_fout(self, v, f=1.0, couple=True):
        """The f_out(v, t) equation from Balloon model

        Args:
        - v -- blod volume
        - f -- blood inflow response
        - couple -- is True when using CBF-CBV coupled model (i.e. tau = 0)
        """
        # print(self._tMTT)
        if couple is True:
            return v**(1/self._alpha)
        return (1/(self._tau + self._tMTT)) * (
            self._tMTT * (v**(1/self._alpha)) + self._tau*f)

    # BOLD response
    def equ_y(self, q, v):
        """The y equation from Physical BOLD signal model

        Args:
        - q -- deoxyhemoglobin content
        - v -- blod volume
        """
        # Dimensionless constants showing baseline physical properties
        k1 = 4.3*self._theta0 * self._E0 * self._TE
        k2 = self._epsilon * self._r0 * self._E0 * self._TE
        k3 = 1 - self._epsilon

        return self._V0 * (k1*(1 - q) + k2*(1 - q/v) + k3*(1 - v))


    def getMTT(self):
        return self._tMTT

    def setParams(self, **kwargs):
        for param, value in kwargs.items():
            setattr(self, "_"+param, value)
        

def time_ref(stimulus, analysis=False):
    """Produce a reference list of time values for iterations in numerical
       methods and error analysis (i.e. RMSE and L2 norm)

    Args:
    - stimulus -- time duration of the stimulus (unit: sec)
    - analysis -- True means produce time list for analysis, which requires a
                  larger final t. Default is False
    """
    h = 0.1  # step size for t, unit: sec
    if analysis is True:
        ti = 0  # initial t
    else:
        ti = -1
    if stimulus == 1:
        if analysis is True:
            tf = 60  # final t
        else:
            tf = 35
    elif stimulus == 30:
        if analysis is True:
            tf = 120
        else:
            tf = 60
    else:
        tf = 150

    return torch.arange(ti, tf, h)  # all t values


def time_select(t_ref, t_all, y_all):
    """Select the values from y_all at the time values of reference

    Args:
    - t_ref -- list of reference values of time
    - t_all -- list of all values of time
    - y_all -- list of all values of BOLD response, y, corresponding to t_all
    """
    y_select = []  # selected y values
    k = 0  # index for selecting y values

    for i in range(len(t_all)):
        if k < len(t_ref):
            # append when the item in t = the item in t_ref
            if torch.abs(t_ref[k] - t_all[i]) <= 1e-15:
                y_select.append(y_all[i])
                k += 1

    return y_select


# Initial values for variables
xE_init = 0.0
xI_init = 0.0
a_init = 0.0
f_init = 1.0
v_init = 1.0
q_init = 1.0
E_init = 0.4  # = E0
fout_init = 1.0
y_init = 0.0
