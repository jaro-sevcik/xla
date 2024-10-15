import jax

from jaxlib.xla_extension import ifrt_proxy
jax._src.xla_bridge.register_backend_factory(  # pylint: disable=protected-access
    "proxy",
    lambda: ifrt_proxy.get_client(
        jax.config.read("jax_backend_target"),
        ifrt_proxy.ClientConnectionOptions(),
    ),
    priority=-1,
)

# Use insecure credentials
# https://github.com/openxla/xla/blob/eef7ee50d0980848436f0b4f402cec8c5bf86f21/xla/python/ifrt_proxy/common/grpc_credentials.cc#L32
import os
os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = "dummy"

jax.config.update("jax_backend_target", "grpc://localhost:4321")

print("Creating proxy client")
print(jax.devices("proxy"))
print("Created proxy client")
