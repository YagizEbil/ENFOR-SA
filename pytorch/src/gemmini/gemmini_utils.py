import random
from src.gemmini import gemmini_config as conf
from src.flist import fault_list as flist


# #define WILL_PE_OUTPUT_BE_ASSIGNED(row,col,iteration) (iteration == row + col + 1)

def pe_first_cycle(row, col):
    return row + col


def pe_last_cycle(row, col, stream_size):
    return row + col + stream_size - 1


def pe_is_active(row, col, stream_size, it):
    return it >= pe_first_cycle(row, col) and it <= pe_last_cycle(row, col, stream_size)


def get_pe_active_rand_cycle(row, col, stream_size):
    first = pe_first_cycle(row, col)
    last  = pe_last_cycle(row, col, stream_size)
    return first + random.randint(0, last)%(last - first + 1)


# generates a random Gemmini fault position at runtime
def gen_random_fault():
    new_fault = flist.GemminiPos()
    idx = random.randint(0, 1) # 0-1 is either IN_A or IN_B
    
    new_fault.target = list(conf.SIGNAL.values())[idx][0]
    bits = list(conf.SIGNAL.values())[idx][1]

    new_fault.pe_row = random.randint(0, conf.DIM-1)
    new_fault.pe_col = random.randint(0, conf.DIM-1)
    new_fault.bit = random.randint(0, bits-1)
    new_fault.ficycle = get_pe_active_rand_cycle(new_fault.pe_row, new_fault.pe_col, conf.DIM)
    return new_fault