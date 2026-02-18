from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.bessel import kv
from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import (
    ensure_hashable,
    has_tracers,
    implements,
)
from jax_galsim.gsobject import GSObject
from jax_galsim.position import PositionD
from jax_galsim.random import UniformDeviate


@jax.jit
def _Knu(nu, x):
    """Modified Bessel 2nd kind for Untruncated Moffat"""
    return kv(nu, x)


@implements(
    _galsim.Moffat,
    lax_description="""\
The LAX version of the Moffat profile

- does not support truncation or beta < 1.1
- does not support gsparams.maxk_thresholds > 0.1
""",
)
@register_pytree_node_class
class Moffat(GSObject):
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True
    _has_hard_edges = False

    def __init__(
        self,
        beta,
        scale_radius=None,
        half_light_radius=None,
        fwhm=None,
        trunc=0.0,
        flux=1.0,
        gsparams=None,
    ):
        # notice that trunc==0. means no truncated Moffat.
        # let define beta_thr a threshold to trigger the truncature
        self._beta_thr = 1.1

        if has_tracers(trunc) or (
            isinstance(trunc, (np.ndarray, float, jnp.ndarray, int))
            and np.any(trunc != 0)
        ):
            raise ValueError(
                "JAX-GalSim does not support truncated Moffat profiles "
                f"(got trunc={repr(trunc)}, always pass the constant 0.0)!"
            )

        # Parse the radius options
        if half_light_radius is not None:
            if scale_radius is not None or fwhm is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius, or fwhm may be specified",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                    fwhm=fwhm,
                )
            else:
                super().__init__(
                    beta=beta,
                    scale_radius=(
                        half_light_radius
                        / jnp.sqrt(jnp.power(0.5, 1.0 / (1.0 - beta)) - 1.0)
                    ),
                    flux=flux,
                    gsparams=gsparams,
                )
        elif fwhm is not None:
            if scale_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius, or fwhm may be specified",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                    fwhm=fwhm,
                )
            else:
                super().__init__(
                    beta=beta,
                    scale_radius=fwhm / (2.0 * jnp.sqrt(2.0 ** (1.0 / beta) - 1.0)),
                    flux=flux,
                    gsparams=gsparams,
                )
        elif scale_radius is None:
            raise _galsim.GalSimIncompatibleValuesError(
                "One of scale_radius, half_light_radius, or fwhm must be specified",
                half_light_radius=half_light_radius,
                scale_radius=scale_radius,
                fwhm=fwhm,
            )
        else:
            super().__init__(
                beta=beta,
                scale_radius=scale_radius,
                flux=flux,
                gsparams=gsparams,
            )

        if self.gsparams.maxk_threshold > 0.1:
            raise ValueError(
                "JAX-GalSim Moffat profiles do not support gsparams.maxk_threshold values greater than 0.1!"
            )

    @property
    @implements(_galsim.moffat.Moffat.beta)
    def beta(self):
        return self._params["beta"]

    @property
    @implements(_galsim.moffat.Moffat.trunc)
    def trunc(self):
        return 0.0

    @property
    @implements(_galsim.moffat.Moffat.scale_radius)
    def scale_radius(self):
        return self.params["scale_radius"]

    @property
    def _r0(self):
        return self.scale_radius

    @property
    def _inv_r0(self):
        return 1.0 / self._r0

    @property
    def _r0_sq(self):
        return self._r0 * self._r0

    @property
    def _inv_r0_sq(self):
        return self._inv_r0 * self._inv_r0

    @property
    def _maxRrD(self):
        """maxR/rd ; fluxFactor Integral of total flux in terms of 'rD' units."""
        return jnp.sqrt(
            jnp.power(self.gsparams.xvalue_accuracy, 1.0 / (1.0 - self.beta)) - 1.0
        )

    @property
    def _maxR(self):
        """maximum r"""
        return self._maxRrD * self._r0

    @property
    def _maxRrD_sq(self):
        return self._maxRrD * self._maxRrD

    @property
    def _fluxFactor(self):
        return 1.0

    @property
    @implements(_galsim.moffat.Moffat.half_light_radius)
    def half_light_radius(self):
        return self._r0 * jnp.sqrt(
            jnp.power(1.0 - 0.5 * self._fluxFactor, 1.0 / (1.0 - self.beta)) - 1.0
        )

    @property
    @implements(_galsim.moffat.Moffat.fwhm)
    def fwhm(self):
        return self._r0 * (2.0 * jnp.sqrt(2.0 ** (1.0 / self.beta) - 1.0))

    @property
    def _norm(self):
        """Normalisation f(x) (trunc=0)"""
        return self.flux * (self.beta - 1) / (jnp.pi * self._fluxFactor * self._r0**2)

    @property
    def _knorm(self):
        """Normalisation f(k) (trunc = 0, k=0)"""
        return self.flux

    @property
    def _knorm_bis(self):
        """Normalisation f(k) (trunc = 0; k=/= 0)"""
        x1 = self.flux * 4
        x2 = jnp.power(2.0, self.beta)
        x3 = jnp.exp(jax.lax.lgamma(self.beta - 1.0))
        return x1 / (x2 * x3)

    def __hash__(self):
        return hash(
            (
                "galsim.Moffat",
                ensure_hashable(self.beta),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.trunc),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __repr__(self):
        return (
            "galsim.Moffat(beta=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)"
            % (
                ensure_hashable(self.beta),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.trunc),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __str__(self):
        s = "galsim.Moffat(beta=%s, scale_radius=%s" % (
            ensure_hashable(self.beta),
            ensure_hashable(self.scale_radius),
        )
        if self.trunc != 0.0:
            s += ", trunc=%s" % ensure_hashable(self.trunc)
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    @property
    def _prefactor(self):
        return 2.0 * (self.beta - 1.0) / (self._fluxFactor)

    @jax.jit
    def _maxk_func(self, k):
        return (
            jnp.abs(self._kValue(PositionD(x=k, y=0)).real / self.flux)
            - self.gsparams.maxk_threshold
        )

    @property
    @jax.jit
    def _maxk(self):
        return (
            jnp.exp(
                _logmaxk_psuedo_pade_approx(
                    jnp.atleast_1d(jnp.log(self.beta)),
                    jnp.atleast_1d(jnp.log(self.gsparams.maxk_threshold)),
                    RATIONAL_POLY_VALS,
                )
            )[0]
            / self._r0
        )

    @property
    def _stepk_lowbeta(self):
        # implicit trunc>0 => _maxR= trunc
        #    then flux never converges (or nearly so),
        #   => so just use truncation radius
        return jnp.pi / self._maxR

    @property
    def _stepk_highbeta(self):
        # ignore the 1 in (1+R^2), so approximately
        R = (
            jnp.power(self.gsparams.folding_threshold, 0.5 / (1.0 - self.beta))
            * self._r0
        )
        R = jnp.minimum(R, self._maxR)
        # at least R should be 5 HLR
        R5hlr = self.gsparams.stepk_minimum_hlr * self.half_light_radius
        R = jnp.maximum(R, R5hlr)
        return jnp.pi / R

    @property
    def _stepk(self):
        """The fractional flux out to radius R is (if not truncated)
        1 - (1+(R/rd)^2)^(1-beta)
        So solve (1+(R/rd)^2)^(1-beta) = folding_threshold
        """
        return jax.lax.select(
            self.beta <= self._beta_thr, self._stepk_lowbeta, self._stepk_highbeta
        )

    @property
    def _max_sb(self):
        return self._norm

    @jax.jit
    def _xValue(self, pos):
        rsq = (pos.x**2 + pos.y**2) * self._inv_r0_sq
        # trunc if r>maxR with r0 scaled version
        return jnp.where(
            rsq > self._maxRrD_sq, 0.0, self._norm * jnp.power(1.0 + rsq, -self.beta)
        )

    def _kValue_untrunc(self, k):
        """Non truncated version of _kValue"""
        k_msk = jnp.where(k > 0, k, 1.0)
        return jnp.where(
            k > 0,
            self._knorm_bis
            * jnp.power(k_msk, self.beta - 1.0)
            * _Knu(self.beta - 1.0, k_msk),
            self._knorm,
        )

    @jax.jit
    def _kValue(self, kpos):
        """computation of the Moffat response in k-space with switch of truncated/untracated case
        kpos can be a scalar or a vector (typically, scalar for debug and 2D considering an image)
        """
        k = safe_sqrt((kpos.x**2 + kpos.y**2) * self._r0_sq)
        out_shape = jnp.shape(k)
        k = jnp.atleast_1d(k)
        res = self._kValue_untrunc(k)
        return res.reshape(out_shape)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @implements(_galsim.Moffat.withFlux)
    def withFlux(self, flux):
        return Moffat(
            beta=self.beta,
            scale_radius=self.scale_radius,
            trunc=self.trunc,
            flux=flux,
            gsparams=self.gsparams,
        )

    @implements(_galsim.Moffat.shoot)
    def _shoot(self, photons, rng):
        # from the galsim C++ in SBMoffat.cpp
        ud = UniformDeviate(rng)

        # First get a point uniformly distributed on unit circle
        theta = ud.generate(photons.x) * 2.0 * jnp.pi
        rsq = ud.generate(
            photons.x
        )  # cumulative dist function P(<r) = r^2 for unit circle
        sint = jnp.sin(theta)
        cost = jnp.cos(theta)

        # Then map radius to the Moffat flux distribution
        newRsq = jnp.power(1.0 - rsq * self._fluxFactor, 1.0 / (1.0 - self.beta)) - 1.0
        r = self.scale_radius * jnp.sqrt(newRsq)
        photons.x = r * cost
        photons.y = r * sint
        photons.flux = self.flux / photons.size()


# this fitting function and code to build it is defined in the
# dev notebook dev/notebooks/moffat_maxk_interp.ipynb
# !!! DO NOT CHANGE ANY OF THE VARIABLES BELOW !!!

# order of rational function in log(maxk_threshold), log(beta)
PADE_ORDERS = [9, 11]

N_PARAMS_MKTS = PADE_ORDERS[0] * 2 - 1
N_PARAMS_BETA = PADE_ORDERS[1] * 2 - 1
N_PARAMS = N_PARAMS_MKTS * N_PARAMS_BETA

LOG_BETA_MIN = np.log(1.1 + 1e-6)
LOG_BETA_MAX = np.log(100)
LOG_MKTS_MIN = np.log(1e-12)
LOG_MKTS_MAX = np.log(0.1)


def _pade_func(coeffs, x):
    order = (coeffs.shape[0] - 1) // 2
    p = jnp.polyval(coeffs[:order], x)
    q = jnp.polyval(
        jnp.concatenate([coeffs[order:], jnp.ones(1)], axis=0),
        x,
    )
    return p / q


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None))
def _logmaxk_psuedo_pade_approx(log_beta, log_mkts, coeffs):
    log_beta = (log_beta - LOG_BETA_MIN) / (LOG_BETA_MAX - LOG_BETA_MIN)
    log_mkts = (log_mkts - LOG_MKTS_MIN) / (LOG_MKTS_MAX - LOG_MKTS_MIN)
    coeffs = coeffs.reshape(N_PARAMS_MKTS, N_PARAMS_BETA)
    pqvals = jax.vmap(_pade_func, in_axes=(0, None))(coeffs, log_beta)
    return _pade_func(pqvals, log_mkts)


# START OF GENERATED CODE
# RATIONAL_POLY_VALS is the array of rational function
# polynomial coefficients that define the approximation
# fmt: off
RATIONAL_POLY_VALS = np.array(
    [+4.0377541235164999e-01, +9.8573979309710097e-02, -8.8368998636191423e-02, -1.4404058874465467e-01,
     -1.8722517103965541e-01, -2.3941575929900452e-01, +1.9477051520522798e-01, +2.5174893659382911e+00,
     +6.9802569884628065e+00, +2.9528987005934546e+00, -9.1832169346703629e-01, +4.9286238397646115e-01,
     +1.0005636301164393e+00, +7.0392335018807339e-01, -1.4054536940247431e-01, -8.5218622931551169e-01,
     -6.7621128905401928e-01, -2.9537613003541291e-01, -1.2854667245219107e+00, +4.0189909948806379e+00,
     +2.1850570724764290e-01, -4.2274342642823717e-02, -2.2450115304011090e-01, -3.6887180044787632e-01,
     -4.3603364254842064e-01, -4.9256905759091729e-01, -6.6398873219847576e-01, -5.9558712629992638e-01,
     +1.1837909921308221e+00, -4.6138529248538136e+00, +1.3450469324602885e+00, +4.9458187528754460e-01,
     +6.0273293491308400e-01, +6.3962989463396580e-01, +6.1582284694766809e-01, +5.6781212563865269e-01,
     +5.5125443702360621e-01, +5.9619266285882933e-01, +5.4745878470377551e-01, -1.2351160388207373e-01,
     -7.0107993183023398e-01, +9.5935634414374444e+00, -8.7283833589376003e-01, -1.0255510475210847e+00,
     -1.0929211542319643e+00, -9.1020529616651413e-01, -6.0023870397444312e-01, -6.5507195560618903e-01,
     -1.0722148851554705e+00, +7.3885075419617319e-01, -4.0294754110685673e+00, -7.8297431020829418e+00,
     -6.6474640833734255e-01, +7.7380626162435606e-01, +6.7528838101327693e-01, -1.7804564435440101e-01,
     -6.2398848498466120e-01, +5.7643808537703685e-02, +1.1923835092489283e+00, +4.9233103375211917e-01,
     -2.9426949991492894e+00, +2.6998628292637314e+00, -1.4093483711909682e-01, +4.0372810590505115e-01,
     +2.4647318964152784e-01, +1.1417136722445211e-02, -2.9679657567820844e-01, -6.8704346690711138e-01,
     -1.2937569996243186e+00, -1.7597197158870368e+00, +1.1143935878967266e+00, -3.5782107847819544e+00,
     +4.1199620228132250e-01, +1.1970601681985499e-01, +2.0114099603243733e-02, -6.5434136381943390e-03,
     +8.5736126942115937e-02, +2.7261328153414083e-01, +4.8960392670307473e-01, +9.7834457064666291e-01,
     +2.7168180915113544e+00, +5.8280454184534474e+00, +4.3537429070024833e+00, -3.6175915101171152e-01,
     +3.4258789295460745e-01, +1.5673518908599102e-01, +1.0666667233357530e-01, +2.7402242443574487e-01,
     +5.7450989722739154e-01, +6.5865111299117973e-01, +1.0619471254256168e+00, +5.0281107124390561e+00,
     +4.3769392931878642e-01, +1.3797757705398774e+00, -4.0642060782490819e-01, -4.8107106379014103e-01,
     -4.6545384216554275e-01, -3.2668532926704019e-01, +1.8685658280818033e-02, +7.3138884860456699e-01,
     +2.0590374557083453e+00, +4.2319105687511795e+00, +6.3443456809823573e+00, +3.1050925854801519e+00,
     +2.4566683840045755e-01, -1.4272027049584994e+00, -1.2769555499840839e+00, -1.1702505921157993e+00,
     -1.0897538422282065e+00, -9.9570783824994358e-01, -7.6663044360462829e-01, +2.4280904621074063e-02,
     -6.2825977906654862e-01, +4.7748459315512886e+00, -3.4479672270932647e-02, +1.5810745647692885e+00,
     +1.0592324253022773e+00, +6.5192719848377234e-01, +3.4733038965947277e-01, +1.3710269537809816e-01,
     +1.7457176684719813e-02, -1.0579281455920138e-02, +1.1577313256892235e-01, +6.3498619537927703e-01,
     -3.5636713339300936e+00, +2.7473701713859548e+01, -8.1471628900480619e-01, -6.2163518452759314e-01,
     -6.3079895204862313e-01, -8.7260751681315663e-01, -7.9737042513719381e-01, +4.1369737806800105e-01,
     +2.2584145837329177e+00, +6.9141061101150347e-01, +1.2238567163847529e+01, -1.4881240397019797e+00,
     +2.5346008210149407e+00, +3.4144215256062256e+00, +3.9404077941580811e+00, +4.0697631657277480e+00,
     +4.0020162458839401e+00, +3.6956516567833777e+00, +1.5471753010948357e+00, -4.5096733243409268e+00,
     +6.2197356650587894e+00, +1.3720454142006250e+01, +7.1789808873106802e+00, +1.2878469385318689e+00,
     +2.5159257373932378e+00, +1.9825772948334588e+00, -2.0124795293398445e-01, +7.3072182794828144e-01,
     +4.3788775079638773e+00, +2.4851710519494072e+00, +3.9571444680295214e+00, +7.9890557503685585e+01,
     +3.2163012590954909e+00, -1.3055299196086150e+00, +2.4047152551806925e+00, +1.1869982622608657e+00,
     -2.6782591519389305e+00, -2.1351447178930658e+00, +4.7064345711296918e+00, +3.7186007845229394e+00,
     -6.2948721775023833e+00, +1.9222396464692157e+00, -5.2929135394148226e+00, +2.4064161945473813e+01,
     +4.6699620000159325e-01, -2.8824483958610392e-01, -5.2987053311717192e-01, -2.5266220838257281e-01,
     +3.3359802997166604e-01, +3.4368754731009948e-01, -1.7259824215429349e+00, +1.1875712562662768e+00,
     -6.4125482087187963e-03, -3.0986991304825956e-01, +1.7012128546438319e+00, +1.4041294008838343e+00,
     +1.1944956562005928e+00, +1.0382744098886449e+00, +9.6541559432919521e-01, +1.0484758969468237e+00,
     +1.1394063481569940e+00, +3.4269611648067827e-01, -1.9133248804297314e+00, +1.0420905503308806e+01,
     +2.1179961925447843e+00, -1.1710420652576292e-01, +1.4435488943498453e-01, +3.1217816268483334e-01,
     +4.9427150925041208e-01, +9.8917710459391761e-01, +1.6007479958753712e+00, +1.7741348266216928e-01,
     -2.3968752594096783e+00, +2.7544604324111326e+00, +1.3892657174473839e+00, +8.7578407666511837e-01,
     -3.5073692239590564e-03, -5.2224301780733773e-01, -3.6262978387803235e-01, +4.3774916805850367e-01,
     +1.2253499947430535e+00, +1.1500998128760624e+00, +1.1268239524120995e+00, +5.8797742250985641e+00,
     -6.6184466647212081e-01, +5.6721441744825265e+00, -5.6157472743129566e-01, -4.3196098885679984e-01,
     -3.7715669435929966e-01, -2.5883322619683380e-01, +4.1549169648999079e-01, +1.9625009395677049e+00,
     +2.4486960792180841e+00, -1.7975615666661942e+00, +1.9835147875960215e+00, -1.2784851851614869e+00,
     +5.8984320559814285e-01, +7.4229958232406057e-01, +7.4162021876167439e-01, +4.0929957187021387e-01,
     -2.0185674746649859e-01, -5.9468903144747898e-01, +2.5498400735845517e-01, +3.1515519885019567e+00,
     +4.4864718840647067e+00, -7.0240004332775219e-01, +4.6010907702840802e+00, -7.1516760160570758e-01,
     -6.0415971544731641e-02, +2.8908143240634510e-01, +2.9629615720773861e-01, +1.4493278876410129e-01,
     +1.4899706167753710e-01, -2.5624565717364778e-01, -5.4356503353058161e-01, +2.1871925270524719e+00,
     +1.2543898090398127e+00, -1.0185718997820286e+00, -6.7596194102120388e-01, -3.8041180369333510e-01,
     -1.0978898835205055e-01, +1.5099465724664984e-01, +4.2016694973194896e-01, +7.3351664383189963e-01,
     +1.0551205409739042e+00, +9.1024490373168820e-01, -6.3763153414131846e-01, +6.8893802543718774e+00,
     -4.5971718133357287e-01, -6.9284977440148543e-01, -8.6796268450746850e-01, -1.0950384320348023e+00,
     -1.1966360910997955e+00, -6.7707548888538704e-01, -1.3280469312410206e-01, -3.2278394553456100e+00,
     -1.4343263133851598e+00, -2.6555777667727885e+00, +8.7311541655505295e-02, +2.4177522001854126e-01,
     +4.5981899324422781e-01, +5.3346538819879574e-01, +5.2250263059044200e-01, +6.7501024530644282e-01,
     +7.0657219206455624e-01, -3.9412109509589072e-01, -1.0587888360932021e+00, +4.2457441911909166e+00,
     -2.3140489272869411e-01, -2.2861844257468347e+00, -1.3329768983397348e+00, -5.3454494370389308e-02,
     +1.0612438826511756e+00, +1.6464548555599556e+00, +2.1981962147620164e+00, +2.1218914660603314e+00,
     -3.7578534604190112e+00, +8.2081595576296251e-01, +4.3296094785753092e-01, -1.0816112906194235e+00,
     -3.6484327546078127e-01, +6.0909104075857867e-01, +1.6358031456615689e+00, +2.3229646816108964e+00,
     +2.2118727909599141e+00, +1.3469163868086866e+00, +1.2994214055102531e+00, +4.3813763657608504e+00,
     +4.9975532252885566e+00, +1.3903679245689864e+00, +4.8414327285872227e-01, +7.0224530774924843e-01,
     +1.0803339536568390e+00, +1.3373166223466377e+00, +9.5760508245182152e-01, -6.1863962466179623e-01,
     -3.3019470393120369e+00, -5.9049244231580120e+00, +9.2207101115694030e+00, +5.3444749163645511e-01,
     +1.6108800350605517e+00, +1.3165642014926429e+00, +9.0587909506896747e-01, +4.3594752341224680e-01,
     +3.6916356990756842e-02, -8.2038474944938655e-02, +3.2166115760007336e-01, +1.4564827647673204e+00,
     +2.8544422054339025e+00, +2.6552473716396197e+00, +1.4919279500221457e+01, -2.5271129085382765e-01,
     -8.8992556214129459e-01, -6.9246476700165105e-01, +1.8480538936798771e-01, +1.2640503721685272e+00,
     +1.4458856404463873e+00, -3.1286650751120088e-01, -3.3813513244865669e+00, +3.7081909895009160e+00,
     +9.5461952229348529e-02, +1.8316337726387808e+00, +1.9081909769641892e+00, +1.7882437073633883e+00,
     +1.4886753215997492e+00, +1.0530609759478995e+00, +6.0216177299874951e-01, +4.6111555601369725e-01,
     +1.2362626622408777e+00, +2.6589804765640928e+00, +3.0218939157874125e+00, +9.6956758605654425e+00,
     +3.0084649428700154e+00, -1.6285708118911306e+00, -3.3636750849697350e+00, -1.3583162670326305e+00,
     +2.5494015893551629e+00, +3.0980878905749085e+00, -3.0085452437932623e+00, +1.5124259940551708e+00,
     +5.5515261212099150e+00, -1.5205550733351489e-01, -5.5411686182748421e-01, +4.9613391570372412e-01,
     +1.5851717222454447e+00, +2.4378677019392678e+00, +2.4074271019318774e+00, +1.3514419949232819e+00,
     +1.7063949677974886e+00, +7.4087472372617151e+00, +1.7585074429003971e+00, +1.3341690752552770e+01,
     +7.0750414351312099e+00],
    dtype=np.float64,
)
# fmt: on
# END OF GENERATED CODE
