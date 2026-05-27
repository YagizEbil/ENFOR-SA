import sys
import os

# add the root path to to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.gemmini.gemmini_extension_definitions as ext
import src.definitions as defs


# Available configs
#
# OS configs
#
#CONFIG_KEY = "OSDIM4"
#CONFIG_KEY = "OSDIM8"
#CONFIG_KEY = "OSDIM16"
#CONFIG_KEY = "OSDIM32"
#CONFIG_KEY = "OSDIM64"

#
# WS configs
#
#CONFIG_KEY = "WSDIM4"
CONFIG_KEY = "WSDIM8"
#CONFIG_KEY = "WSDIM64"


defs.ENABLE_GL_FAULT_MODEL = False

#
# Loads the Gemmini module - the ahead-of-time extension to interface with the verilated Gemmini module (this lib is designed in /rtl/lib/Gemmini)
#
gemmini = ext.load_extension(CONFIG_KEY)
gemmini.init()
gemmini.print_info()
