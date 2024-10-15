import jax
import jaxlib

import os
os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = "dummy"

local_client = jax._src.xla_bridge.get_backend()
jaxlib.xla_extension.ifrt_proxy.get_server("localhost:4321", local_client)