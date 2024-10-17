import jax
import numpy as np

# Register the "proxy" platform.
from jaxlib.xla_extension import ifrt_proxy
jax._src.xla_bridge.register_backend_factory(  # pylint: disable=protected-access
    "proxy",
    lambda: ifrt_proxy.get_client(
        jax.config.read("jax_backend_target"),
        ifrt_proxy.ClientConnectionOptions(),
    ),
    priority=-1,
)
# Make sure we only use the "proxy" platform.
jax.config.update("jax_platforms", "proxy")

# Use insecure credentials
# https://github.com/openxla/xla/blob/eef7ee50d0980848436f0b4f402cec8c5bf86f21/xla/python/ifrt_proxy/common/grpc_credentials.cc#L32
import os
os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = "dummy"

# Set the "remote" proxy address.
jax.config.update("jax_backend_target", "grpc://localhost:4321")

print("Proxy devices: ", jax.devices("proxy"))

def f(x):
    return x + 1

s = jax.sharding.SingleDeviceSharding(jax.devices("proxy")[0])
a = jax.device_put(np.array([1,2,3,4]), device=s)
a = jax.jit(f, in_shardings=s, out_shardings=s)(a)

print("Result: ", a)
print("Device: ", a.sharding._device)