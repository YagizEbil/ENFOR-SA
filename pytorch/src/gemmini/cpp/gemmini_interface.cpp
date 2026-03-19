#include <torch/torch.h>
#include <iostream>
#include "../../../../rtl/lib/Gemmini/headers/MxM.h"


/* 
    This is the code to interface PyTorch with Gemmini. This is used on top of the MxM Gemmini interface. 
    Gemmini is accessed through the MxM *mxm object below.
    We need this interface on top of MxM mostly to convert PyTorch tensors to native matrix representation (e.g., int mat[DIM][DIM]) 
    so that it can be sent to Gemmini through the MxM interface.

    The full call stack is as follows:
        PyTorch app
        ↓
        [Optional] gemmini_interface.py
        ↓
        This interface: converts PyTorch tensors to native types. Transfers data between the app and Gemmini
        ↓
        MxM.h wrapper (main API definition to stream data to Gemmini)
        ↓
        Gemmini.h (interface adapters to Mesh)
        ↓
        Mesh.h (encapsulates the verilated signals) + PeState.h (PE abstracton + PE signals)
        ↓
        VMesh.cpp (verilated design, RTL models)

    VERY IMPORTANT: 
        1. All input tensors passed as parameters to this layer (preloads or matmuls) MUST be *** contiguous *** in memory.
        2. In the quantized models, the inputs and weights are assured to be contigous with the following flattener.py function:
            input_unfolded = input_unfolded.contiguous().view(batch_size, in_channels * kernel_height * kernel_width, -1)
            conv_weight_flat = conv_weight.contiguous().view(out_channels, -1) 
            the extract_tile assure contiguous tensors: torch::Tensor tile = torch::zeros({DIM, DIM}, get_dtype_from_code(dtype_code))

        3. Many PyTorch functions produce non-contigous tensors:
            torch.transpose(tensor)
            tensor.t()
            tensor.permute(*dims)   Reorders dimensions → non-contiguous
            tensor.view()   Returns view only if contiguous; may raise error if not
            tensor.reshape()

            copies like D_tile = D_p[i:i+DIM, j:j+DIM] are not assured to be contigous

            If a tensor uses any of these function calls, the tensor must be copied to a contigous space. Example:
                tensor = torch.randint(...)        → contiguous. safe to be used here directly
                tensor_t = tensor.t()              → maybe non-contiguous. tensor_t cannot be used in any function here
                tensor_t = tensor.t().contiguous() → contiguous. safe to be used here
*/


/* The main matmul solver on top of Gemmini */
MxM *mxm = nullptr;


/* Creates an instance of the main interface to Gemmini */
int init()
{
    /* This is the core verilated module (defined in /rtl/lib/Gemmini/headers/config.h) */
    VL_DUT *dut;
    
    /* An instance of the Gemmini systolic array (emulates OS and WS input flow to RTL verilated modules) */
    Gemmini *gemmini;  
   
    if ((dut = new VL_DUT) == nullptr)
    {
        printf("[C++ extension Failed] init() err: DUT\n");
        return 0;
    }

    if ((gemmini = new Gemmini(dut, GEMM_CONFIG_ALIAS)) == nullptr)
    {
        printf("[C++ extension Failed] init() err: gemmini\n");
        return 0;
    }

    if ((mxm = new MxM(gemmini)) == nullptr)
    //if ((mxm = MxM::getInstance(gemmini)) == nullptr)
    {
        printf("[C++ extension Failed] init() err: mxm\n");
        return 0;
    }
    //std::cout << RED << "[C++ extension]: Gemmini config: " << GEMM_CONFIG_ALIAS << RESET <<  std::endl;
    return 1;
}


void finish()
{
    delete mxm;
}


/* Preloads Gemmini with the tensor. Assuming C=A.B + D, OS must preload D. WS must preload B.*/
uint32_t preload(const torch::Tensor& tensor)
{
    /* this trick forces the tensor to be contiguous in memory, however, if the tensor is not already contiguous, this 
    will result in a memory copy, which is bad for perf. i'd rather only pass contiguous tensors to all functions in this layer */
    //if (!tensor.is_contiguous()) 
        //tensor = tensor.contiguous();
    
    assert(tensor.dim() == 2 && tensor.size(0) == DIM && tensor.size(1) == DIM);
    // Get pointers to the data
#ifdef GEMM_OS
    Input_t* data_ptr = tensor.data_ptr<Input_t>();  // cannot cast this to char
#else
    Input_t* data_ptr = tensor.data_ptr<Output_t>();
#endif
    return mxm->preload(data_ptr);
}

 
#ifdef GEMM_OS
/* Flushes the Gemmini PE state to the tensor (OS mode only) */
uint32_t flush_gemm(torch::Tensor& tensor, bool transpose_output)
{
    assert(tensor.dim() == 2 && tensor.size(0) == DIM && tensor.size(1) == DIM);
    // Get pointers to the data
    Output_t* data_ptr = tensor.data_ptr<Output_t>();
    uint32_t r =  mxm->flush(data_ptr, transpose_output);
    tensor = torch::from_blob(data_ptr, {tensor.size(0), tensor.size(0)}, torch::kInt32); 
    return r;
}


/* OS: Streams tA and tB in the array, and keeps the data in the internal PE accumulators */
uint32_t stream(const torch::Tensor& tA, const torch::Tensor& tB) 
{
#if 0 // no need. ViTs run on GPUs, but the tensor was already transfered to the CPU in PyTorch before this call
    auto A_cpu = A.to(torch::kCPU, torch::kInt).contiguous();
    auto B_cpu = B.to(torch::kCPU, torch::kInt).contiguous();
#endif

    assert(tA.dim() == 2 && tA.size(0) == DIM);
    assert(tB.dim() == 2 && tB.size(1) == DIM);
    assert(tA.size(1) == tB.size(0));

    // Get pointers to the data
    Input_t* data_ptr_A = tA.data_ptr<Input_t>();
    Input_t* data_ptr_B = tB.data_ptr<Input_t>();

    return mxm->stream(data_ptr_A, data_ptr_B, tA.size(1));
}
#endif


#ifdef GEMM_WS
/* WS: streams tA through the array. this assumes a bias matrix zero. the output is stored in tC*/
uint32_t stream(const torch::Tensor& tA, torch::Tensor& tC) 
{
    assert(tA.dim() == 2 && tA.size(0) == DIM);
    assert(tC.dim() == 2 && tC.size(0) == DIM && tC.size(1) == DIM);

    // Get pointers to the data
    Input_t*  data_ptr_A = tA.data_ptr<Input_t>();
    Output_t* data_ptr_C = tC.data_ptr<Output_t>();
   
    uint32_t r = mxm->stream(data_ptr_A, data_ptr_C,  tA.size(1));
    tC = torch::from_blob(data_ptr_C, {tC.size(0), tC.size(0)}, torch::kInt32); 
    return r;
}


/* WS: streams tA and tD (bias) matrix through the array. the output is stored in tC */
uint32_t stream_bias(const torch::Tensor& tA, const torch::Tensor& tD, torch::Tensor& tC) 
{
    assert(tA.dim() == 2 && tA.size(0) == DIM);
    assert(tD.dim() == 2 && tD.size(1) == DIM);
    assert(tA.size(1) == tD.size(0));

    // Get pointers to the data
    Input_t*  data_ptr_A = tA.data_ptr<Input_t>();
    Output_t* data_ptr_D = tD.data_ptr<Output_t>();
    Output_t* data_ptr_C = tC.data_ptr<Output_t>();

    uint32_t r =  mxm->stream_bias(data_ptr_A, data_ptr_D, data_ptr_C, tA.size(1));
    tC = torch::from_blob(data_ptr_C, {tC.size(0), tC.size(0)}, torch::kInt32);
    return r;
}
#endif


/* Adds faults to the Gemmini fault list. See /rtl/lib/Gemmini/headers/FaultList.h 
   The "cluster" group specifies which input is the inject target according to the values defined in the "ClusterLoader.h" enum values 
   (Mesh.h refers to such values to create the fault injection callbacks)
*/
void add_transient_fault(
    uint8_t cluster, 
    uint8_t row, 
    uint8_t col, 
    uint8_t bit,
    uint8_t ficycle,
    uint32_t cell, 
    bool silent)
{
    mxm->gemmini->addTransientFault(
        FaultModel::FM_TRANSIENT, 
        cluster, 
        row, 
        col, 
        bit,
        ficycle,
        cell, 
        silent);
}


void add_permanent_fault(
    uint8_t cluster, 
    uint8_t row, 
    uint8_t col, 
    uint8_t bit, 
    int8_t pol,
    uint32_t cell, 
    bool silent)
{
    mxm->gemmini->addPermanentFault(
        FaultModel::FM_PERMANENT, 
        cluster, 
        row, 
        col, 
        bit, 
        pol, 
        cell, 
        silent);
}


/* Clears the fault list. For transient faults, this must be called before each new Gemmini matmul */
void clear_fault_list()
{
    mxm->gemmini->clearFaultList();
}


const char* get_config_name()
{
    return (const char*)GEMM_CONFIG_ALIAS;
}


uint32_t reset()
{
    return mxm->reset();
}


void tick_test_perf(size_t trials)
{
    mxm->tick_test_perf(trials);
}


void print_info()
{
    std::cout << "- Gemmini config  : " << GEMM_CONFIG_ALIAS << std::endl;
    
    #ifdef GEMM_OS
        std::cout << "- Gemmini mode    : OS\n";
    #else
        std::cout << "- Gemmini mode    : WS\n";
    #endif

    std::cout << "- Gemmini DIM     : " << DIM << std::endl;
    std::cout << "- Verilated module: " << CONFIG_NAME << std::endl;

    #ifdef USE_GL_INJECTION
        std::cout << "- Fault level     : Gates\n";
    #else
        std::cout << "- Fault level     : RTL\n";
    #endif

#if 0
    #ifdef ENABLE_PERMANENT_FAULTS
        std::cout << "- ENABLE_PERMANENT_FAULTS: ON\n";
    #else
        std::cout << "- ENABLE_PERMANENT_FAULTS: OFF\n";
    #endif
#endif
    std::cout << RESET << std::endl;
}


#if 0
/* This sets a value 'val' to a chosen Gemmini signal in a chosen row of a cluster */
int set_signal (int cluster, int row, int val)
{
    mxm->gemmini->setSignalValue(cluster, row, val);
    return 0;
}


/* Reads the value*/
int get_signal (int cluster, int row)
{
    return mxm->gemmini->getSignalValue(cluster, row);
}
#endif


// Register the function with PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("init", &init, "Custom C++ function");
    m.def("reset", &reset, "Custom C++ function");
    m.def("stream", &stream, "Custom C++ function");
    m.def("finish", &finish, "Custom C++ function");
    m.def("preload", &preload, "Custom C++ function");
    m.def("print_info", &print_info, "Custom C++ function");
    m.def("tick_test_perf", &tick_test_perf, "Custom C++ function");
    m.def("get_config_name", &get_config_name, "Custom C++ function");
    m.def("clear_fault_list", &clear_fault_list, "Custom C++ function");
    m.def("add_transient_fault", &add_transient_fault, "Custom C++ function");
    m.def("add_permanent_fault", &add_permanent_fault, "Custom C++ function");

#ifdef GEMM_OS
    m.def("flush_gemm", &flush_gemm, "Custom C++ function");
#else // GEMM_WS
    m.def("stream_bias", &stream_bias, "Custom C++ function");
#endif

    //m.def("set_signal", &set_signal, "Custom C++ function");
    //m.def("get_signal", &get_signal, "Custom C++ function");

}
