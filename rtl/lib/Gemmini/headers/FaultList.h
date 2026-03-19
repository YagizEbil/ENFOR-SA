#ifndef FAULT_LIST__H
#define FAULT_LIST__H
#include <map>


/* used to insert/remove code that check for permanent faults for improved fi perf. */
#define ENABLE_PERMANENT_FAULTS
//#undef ENABLE_PERMANENT_FAULTS

#define NUMBER_OF_REGISTER_CLUSTERS 10

enum class FaultModel : int 
{
    FM_NO_FAULT   = 0,
    FM_TRANSIENT  = 1,
    FM_PERMANENT  = 2,
    N_FAULT_TYPES = 3
};

const char *fault_models_str[] = 
{
    "FM_NO_FAULT",
    "FM_TRANSIENT",
    "FM_PERMANENT"
};

enum
{
    IDX_io_in_a,    // 0
    IDX_io_in_b,    // 1
    IDX_io_in_d,    // 2
    IDX_io_out_b,   // 3
    IDX_io_out_c,   // 4
    IDX_propagate,  // 5
    IDX_valid,      // 6
    IDX_io_out_a,   // 7
    IDX_c1,         // 8 
    IDX_c2          // 9
};

const char *CLUSTER_NAMES[NUMBER_OF_REGISTER_CLUSTERS] = 
{
    "IN_A",
    "IN_B",
    "IN_D",
    "OUT_B",
    "OUT_C",
    "PROPAG",
    "VALID",
    "OUT_A",
    "C1",
    "C2"
};

typedef struct _Fault // the fault models and attributes
{
    FaultModel faultModel; // the current fault model (no fault, transient of permanent)
    uint8_t group;  // which reg is the target. e.g., A, B, C, propag, ....
    uint8_t row;    // PE row
    uint8_t col;    // PE col
    uint8_t bit;    // the target bit for permanent faults
    uint8_t ficycle; // the fault injection cycle
    int8_t pol;     // the pol for permanent faults
    uint32_t cell;  // for gate-level FI. specifies which cell to target (HDFIT's GlobalFiNumber = cell)
    bool silent;    // calls print() in the moment the fault is injected
    bool performed; // tells if a transient fault was already performed

    _Fault()
    {}

    _Fault (FaultModel fm,   
        int group_, 
        int row_, 
        int col_, 
        int bit_,
        int ficycle_,
        int cell_, 
        bool silent_) // transient faults
    {
        faultModel = fm;
        group = group_;
        row = row_;
        col = col_;
        bit = bit_;
        ficycle = ficycle_;
        pol = -1; // transients have no polarity
        cell = cell_;
        silent = silent_;
        performed = false;

    #if USE_GL_INJECTION
        sprintf (toString, "Model: %s - Tgt: %s (%d) - PE(%d,%d) - Bit: %d - Cell: %d - Cycle: %d", 
            fault_models_str[static_cast<int>(faultModel)], CLUSTER_NAMES[group], group, row, col, bit, cell, ficycle);
    #else
        sprintf (toString, "Model: %s - Tgt: %s (%d) - PE(%d,%d) - Bit: %d - Cycle: %d", 
            fault_models_str[static_cast<int>(faultModel)], CLUSTER_NAMES[group], group, row, col, bit, ficycle);
    #endif
    }

    _Fault (
        FaultModel fm, 
        int group_, 
        int row_, 
        int col_, 
        int bit_,
        int ficycle_, // invalid
        int pol_, 
        int cell_, 
        bool silent_) // permanent faults
    {
        faultModel = fm;
        group = group_;
        row = row_;
        col = col_;
        bit = bit_;
        silent = silent_;
        pol = pol_;
        cell = cell_;

        sprintf (toString, "Model: %s - Tgt: %s (%d) - PE(%d,%d) - Bit: %d - Pol: %d - Cell: %d", 
            fault_models_str[static_cast<int>(faultModel)], CLUSTER_NAMES[group], group, row, col, bit, pol, cell);
    }

    void print()
    {
        if (!silent)
            printf ("%s\n", toString);
    }

    char toString[200];

} Fault;


/* helper structure when receiving a fault from a C-based client */

typedef enum _ClientFaultModel 
{
    FM_NO_FAULT   = 0,
    FM_TRANSIENT  = 1,
    FM_PERMANENT  = 2,
    N_FAULT_TYPES = 3
} ClientFaultModel;


#pragma pack(1) // prevents the compiler from padding this struct 
typedef struct _ClientFault // the fault models and attributes (this is used for the server-based gemmini. a client will send this model through a pipe to the server)
{
    ClientFaultModel faultModel;
    uint8_t target;      // which reg is the target. e.g., A, B, C, propag, ....
    uint8_t row, col;   // the target PE position
    uint8_t bit;        // the target bit 
    uint8_t ficycle;
    uint8_t pol;
    int cell; // TODO: this is not included in the client side yet (21/12/2025)
    bool silent;  
} ClientFault;
#pragma pack()


#endif