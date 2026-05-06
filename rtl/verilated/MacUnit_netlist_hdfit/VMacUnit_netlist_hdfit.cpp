// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "VMacUnit_netlist_hdfit.h"
#include "VMacUnit_netlist_hdfit__Syms.h"

//============================================================
// Constructors

VMacUnit_netlist_hdfit::VMacUnit_netlist_hdfit(VerilatedContext* _vcontextp__, const char* _vcname__)
    : vlSymsp{new VMacUnit_netlist_hdfit__Syms(_vcontextp__, _vcname__, this)}
    , io_in_a{vlSymsp->TOP.io_in_a}
    , io_in_b{vlSymsp->TOP.io_in_b}
    , GlobalFiSignal{vlSymsp->TOP.GlobalFiSignal}
    , GlobalFiNumber{vlSymsp->TOP.GlobalFiNumber}
    , io_in_c{vlSymsp->TOP.io_in_c}
    , io_out_d{vlSymsp->TOP.io_out_d}
    , GlobalFiModInstNr{vlSymsp->TOP.GlobalFiModInstNr}
    , rootp{&(vlSymsp->TOP)}
{
}

VMacUnit_netlist_hdfit::VMacUnit_netlist_hdfit(const char* _vcname__)
    : VMacUnit_netlist_hdfit(nullptr, _vcname__)
{
}

//============================================================
// Destructor

VMacUnit_netlist_hdfit::~VMacUnit_netlist_hdfit() {
    delete vlSymsp;
}

//============================================================
// Evaluation loop

void VMacUnit_netlist_hdfit___024root___eval_initial(VMacUnit_netlist_hdfit___024root* vlSelf);
void VMacUnit_netlist_hdfit___024root___eval_settle(VMacUnit_netlist_hdfit___024root* vlSelf);
void VMacUnit_netlist_hdfit___024root___eval(VMacUnit_netlist_hdfit___024root* vlSelf);
QData VMacUnit_netlist_hdfit___024root___change_request(VMacUnit_netlist_hdfit___024root* vlSelf);
#ifdef VL_DEBUG
void VMacUnit_netlist_hdfit___024root___eval_debug_assertions(VMacUnit_netlist_hdfit___024root* vlSelf);
#endif  // VL_DEBUG
void VMacUnit_netlist_hdfit___024root___final(VMacUnit_netlist_hdfit___024root* vlSelf);

static void _eval_initial_loop(VMacUnit_netlist_hdfit__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    VMacUnit_netlist_hdfit___024root___eval_initial(&(vlSymsp->TOP));
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial loop\n"););
        VMacUnit_netlist_hdfit___024root___eval_settle(&(vlSymsp->TOP));
        VMacUnit_netlist_hdfit___024root___eval(&(vlSymsp->TOP));
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = VMacUnit_netlist_hdfit___024root___change_request(&(vlSymsp->TOP));
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("designs/MacUnit_netlist_hdfit/MacUnit_netlist_hdfit.v", 7, "",
                "Verilated model didn't DC converge\n"
                "- See https://verilator.org/warn/DIDNOTCONVERGE");
        } else {
            __Vchange = VMacUnit_netlist_hdfit___024root___change_request(&(vlSymsp->TOP));
        }
    } while (VL_UNLIKELY(__Vchange));
}

void VMacUnit_netlist_hdfit::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VMacUnit_netlist_hdfit::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    VMacUnit_netlist_hdfit___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Clock loop\n"););
        VMacUnit_netlist_hdfit___024root___eval(&(vlSymsp->TOP));
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = VMacUnit_netlist_hdfit___024root___change_request(&(vlSymsp->TOP));
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("designs/MacUnit_netlist_hdfit/MacUnit_netlist_hdfit.v", 7, "",
                "Verilated model didn't converge\n"
                "- See https://verilator.org/warn/DIDNOTCONVERGE");
        } else {
            __Vchange = VMacUnit_netlist_hdfit___024root___change_request(&(vlSymsp->TOP));
        }
    } while (VL_UNLIKELY(__Vchange));
    // Evaluate cleanup
}

//============================================================
// Utilities

VerilatedContext* VMacUnit_netlist_hdfit::contextp() const {
    return vlSymsp->_vm_contextp__;
}

const char* VMacUnit_netlist_hdfit::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

VL_ATTR_COLD void VMacUnit_netlist_hdfit::final() {
    VMacUnit_netlist_hdfit___024root___final(&(vlSymsp->TOP));
}
