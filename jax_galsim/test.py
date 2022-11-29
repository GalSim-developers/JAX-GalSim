import jax
from jax_galsim.noise import sample, GaussianNoise
from jax_galsim.helpers import seed


def model(sig=1, data2={'mean':-1}):
  n1 = sample('n1',GaussianNoise(loc=0,scale=sig), sample_shape=(10,10))
  n2 = sample("n2",GaussianNoise(loc=data2['mean']))
  return n1,n2


print('test1:',seed(model, rng_seed=1)(sig=10))
print('test2:',seed(model, rng_seed=jax.random.PRNGKey(10))(data2={'mean':10}))
