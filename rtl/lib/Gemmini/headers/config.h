#ifndef CONFIG__H
#define CONFIG__H

#define EVAL_GEMMINI 1

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define CLK clock // the clock name used in context->step() (must be the same as in the verilated DUT's top module)
#define VCD 0  // enables VCD tracing (must use verilator flag --trace)

#ifndef CLK
    #warning "No clock signal defined"
#endif

#if EVAL_GEMMINI
#   define CONFIG_NAME "VMesh"
#   define VL_DUT VMesh
#   if !(defined (GEMM_WS) || defined (GEMM_OS))
#       error "Error: You must define either GEMM_OS or GEMM_WS"
#   endif


/* 
    Includes the configuration file in /configs/
        1 - there must be one file for each config
        2 - the file names must match the <config_name> (i.e., passed in g++ -DCONFIG=<config_name>): name it like <config_name>.cc
*/
#   ifdef GEMM_OS
#       include STR(configs/os/CONFIG.h)
#   else
#       include STR(configs/ws/CONFIG.h)
#   endif
#endif  // #if EVAL_GEMMINI


// If we enable the integration of Gemmini with a single gate-level MAC uint
#ifdef USE_GL_INJECTION
    // Code removed for open source
//#   error "Galte-level FI not available as open source"
#   include "../../../verilated/MacUnit_netlist_hdfit/VMacUnit_netlist_hdfit.h"
#   include "../../../verilated/MacUnit_netlist_hdfit/VMacUnit_netlist_hdfit___024root.h"
#   define MAC_UNIT VMacUnit_netlist_hdfit
#endif


#endif // CONFIG__H