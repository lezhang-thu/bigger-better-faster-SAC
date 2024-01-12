import jax.numpy as jnp
import torch
import math

num_actions = 16
frac = 0.2
x = torch.full((num_actions,),
               fill_value=frac / num_actions,
               dtype=torch.float64)
x[0] += 1 - frac
print(x.sum().item())
assert math.isclose(x.sum().item(), 1)
ent_targ = torch.distributions.categorical.Categorical(
    probs=x.float()).entropy()
ent_targ = jnp.asarray(ent_targ)
print(ent_targ.shape)
print(type(ent_targ))
