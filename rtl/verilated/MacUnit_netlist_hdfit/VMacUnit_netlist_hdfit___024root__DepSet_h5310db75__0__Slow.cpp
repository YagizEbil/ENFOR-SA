// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VMacUnit_netlist_hdfit.h for the primary calling header

#include "verilated.h"

#include "VMacUnit_netlist_hdfit___024root.h"

VL_ATTR_COLD void VMacUnit_netlist_hdfit___024root___eval_initial(VMacUnit_netlist_hdfit___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VMacUnit_netlist_hdfit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMacUnit_netlist_hdfit___024root___eval_initial\n"); );
}

void VMacUnit_netlist_hdfit___024root___combo__TOP__0(VMacUnit_netlist_hdfit___024root* vlSelf);

VL_ATTR_COLD void VMacUnit_netlist_hdfit___024root___eval_settle(VMacUnit_netlist_hdfit___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VMacUnit_netlist_hdfit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMacUnit_netlist_hdfit___024root___eval_settle\n"); );
    // Body
    VMacUnit_netlist_hdfit___024root___combo__TOP__0(vlSelf);
}

VL_ATTR_COLD void VMacUnit_netlist_hdfit___024root___final(VMacUnit_netlist_hdfit___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VMacUnit_netlist_hdfit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMacUnit_netlist_hdfit___024root___final\n"); );
}

VL_ATTR_COLD void VMacUnit_netlist_hdfit___024root___ctor_var_reset(VMacUnit_netlist_hdfit___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VMacUnit_netlist_hdfit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMacUnit_netlist_hdfit___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->GlobalFiSignal = VL_RAND_RESET_I(31);
    vlSelf->GlobalFiNumber = VL_RAND_RESET_I(32);
    for (int __Vi0=0; __Vi0<1; ++__Vi0) {
        vlSelf->GlobalFiModInstNr[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->io_in_a = VL_RAND_RESET_I(8);
    vlSelf->io_in_b = VL_RAND_RESET_I(8);
    vlSelf->io_in_c = VL_RAND_RESET_I(32);
    vlSelf->io_out_d = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___000_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___001_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___002_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___003_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___004_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___005_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___006_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___007_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___008_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___009_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___010_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___011_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___012_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___013_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___014_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___015_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___016_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___017_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___018_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___019_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___020_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___021_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___022_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___023_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___024_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___025_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___026_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___027_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___028_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___029_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___030_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___031_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___032_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___033_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___034_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___035_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___036_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___037_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___038_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___039_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___040_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___041_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___042_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___043_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___044_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___045_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___046_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___047_ = VL_RAND_RESET_I(25);
    vlSelf->MacUnit__DOT___048_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___049_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT___050_ = VL_RAND_RESET_I(32);
    vlSelf->MacUnit__DOT____Vconcswap_1_h188b270b__0 = VL_RAND_RESET_I(17);
    vlSelf->MacUnit__DOT____Vconcswap_1_h16bc72b2__0 = VL_RAND_RESET_I(20);
    vlSelf->MacUnit__DOT____Vconcswap_1_hca2eac98__0 = VL_RAND_RESET_I(16);
    vlSelf->MacUnit__DOT____Vconcswap_1_ha4f6844f__0 = VL_RAND_RESET_I(19);
    vlSelf->MacUnit__DOT____Vconcswap_1_h768c2d94__0 = VL_RAND_RESET_I(15);
    vlSelf->MacUnit__DOT____Vconcswap_1_h840d9808__0 = VL_RAND_RESET_I(15);
    vlSelf->MacUnit__DOT____Vconcswap_1_hfc8e40d8__0 = VL_RAND_RESET_I(14);
    vlSelf->MacUnit__DOT____Vconcswap_1_h27e0a201__0 = VL_RAND_RESET_I(14);
    vlSelf->MacUnit__DOT____Vconcswap_1_hd0f65f2a__0 = VL_RAND_RESET_I(13);
    vlSelf->MacUnit__DOT____Vconcswap_1_hebdcd2fa__0 = VL_RAND_RESET_I(13);
    vlSelf->__Vchglast__TOP__io_out_d = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___000_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___001_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___002_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___003_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___004_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___005_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___006_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___007_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___008_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___009_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___010_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___011_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___012_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___013_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___014_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___015_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___016_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___017_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___018_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___019_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___023_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___024_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___025_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___026_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___028_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___029_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___030_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___031_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___033_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___034_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___036_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___038_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___041_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___042_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___043_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___047_ = VL_RAND_RESET_I(25);
    vlSelf->__Vchglast__TOP__MacUnit__DOT___048_ = VL_RAND_RESET_I(32);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_h188b270b__0 = VL_RAND_RESET_I(17);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_h16bc72b2__0 = VL_RAND_RESET_I(20);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_hca2eac98__0 = VL_RAND_RESET_I(16);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_ha4f6844f__0 = VL_RAND_RESET_I(19);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_h768c2d94__0 = VL_RAND_RESET_I(15);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_h840d9808__0 = VL_RAND_RESET_I(15);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_hfc8e40d8__0 = VL_RAND_RESET_I(14);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_h27e0a201__0 = VL_RAND_RESET_I(14);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_hd0f65f2a__0 = VL_RAND_RESET_I(13);
    vlSelf->__Vchglast__TOP__MacUnit__DOT____Vconcswap_1_hebdcd2fa__0 = VL_RAND_RESET_I(13);
}
