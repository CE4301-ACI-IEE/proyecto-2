
# import libraries
import sys
import matplotlib.pyplot as plt

# START 
print (' -> Starting...') 

# Read files and extract information
# Class BenchmarkParameters that store the information from
# the result of the simulated benchmarks
class BenchmarkParameters:
    def __init__( self, bpred, repl, 
    il1_size, il2_size, dl1_size, dl2_size,
    assoc, imult, ialu, fpmult, fpalu, 
    mem_width, in_order ):
        self.bpred = bpred
        self.repl = repl
        self.il1_size = il1_size
        self.il2_size = il2_size
        self.dl1_size = dl1_size
        self.dl2_size = dl2_size
        self.assoc = assoc
        self.imult = imult
        self.ialu = ialu
        self.fpmult = fpmult
        self.fpalu = fpalu
        self.mem_width = mem_width
        self.in_order = in_order
    
    def print_BenchmarkParameters( self ):
        print " * OBJECT BenchmarkParameters atributes * "
        print ("   - bpred : " + self.bpred)
        print ("   - repl : " + self.repl)
        print ("   - il1_size : " + self.il1_size)
        print ("   - il2_size : " + self.il2_size)
        print ("   - dl1_size : " + self.dl1_size)
        print ("   - dl2_size : " + self.dl2_size)
        print ("   - assoc : " + self.assoc)
        print ("   - imult : " + self.imult)
        print ("   - ialu : " + self.ialu)
        print ("   - fpmult : " + self.fpmult)
        print ("   - fpalu : " + self.fpalu)
        print ("   - mem_width : " + self.mem_width)
        print ("   - in_order : " + self.in_order)
    
class BenchmarkResults:
    def __init__( self, il1_misses, il2_misses,
    dl1_misses, dl2_misses, sim_cycles,
    sim_IPC, sim_CPI, sim_elapsed_time ):
        self.il1_misses = il1_misses
        self.il2_misses = il2_misses
        self.dl1_misses = dl1_misses
        self.dl2_misses = dl2_misses
        self.sim_cycles = sim_cycles
        self.sim_IPC = sim_IPC
        self.sim_CPI = sim_CPI
        self.sim_elapsed_time = sim_elapsed_time

    def print_BenchmarkResults( self ):
        print " * OBJECT BenchmarkResults atributes * "
        print ("   - il1_misses : " + self.il1_misses)
        print ("   - il2_misses : " + self.il2_misses)
        print ("   - dl1_misses : " + self.dl1_misses)
        print ("   - dl2_misses : " + self.dl2_misses)
        print ("   - sim_cycles : " + self.sim_cycles)
        print ("   - sim_IPC : " + self.sim_IPC)
        print ("   - sim_CPI : " + self.sim_CPI)
        print ("   - sim_elapsed_time (seconds) : " + self.sim_elapsed_time)

# Read file
def read_information_from_file( path ):

    file_object = open( path, "r" )
    params_list = file_object.readline().split(",")
    
    aux_benchmark = BenchmarkParameters(
        params_list[1].split()[1],
        params_list[2].split()[1],
        params_list[4].split()[1],
        params_list[4].split()[1],
        params_list[4].split()[1],
        params_list[4].split()[1],
        params_list[5].split()[1],
        params_list[6].split()[1],
        params_list[7].split()[1],
        params_list[8].split()[1],
        params_list[9].split()[1],
        params_list[10].split()[1],
        params_list[11].split()[1]
    )

    result_lines = file_object.readlines()
    aux_result = BenchmarkResults(
        result_lines[212].split()[1],
        result_lines[222].split()[1],
        result_lines[232].split()[1],
        result_lines[242].split()[1],
        result_lines[167].split()[1],
        result_lines[168].split()[1],
        result_lines[169].split()[1],
        result_lines[160].split()[1]
    )

    file_object.close()

    return aux_benchmark, aux_result

# Reading files to extract information
def get_information( path, amount ):
    result_matrix = [[],[]]
    initial_path = path
    for i in range( 1, amount+1 ):
        res_tuple = read_information_from_file( initial_path + str(i) + ".txt" )
        result_matrix[0] += [res_tuple[0]]
        result_matrix[1] += [res_tuple[1]]

    return result_matrix


# Call functions
b2lev_matrix = get_information("results/b2lev_", 24)
b2lev_types_matrix = get_information("results/b2lev_types_", 3)
bimod_matrix = get_information("results/bimod_", 24)
bimod_types_matrix = get_information("results/bimod_types_", 3)
comb_matrix = get_information("results/comb_", 24)
comb_types_matrix = get_information("results/comb_types_", 3)
nottaken_matrix = get_information("results/nottaken_", 24)
nottaken_types_matrix = get_information("results/nottaken_types_", 3)
taken_matrix = get_information("results/taken_", 24)
taken_types_matrix = get_information("results/taken_types_", 3)
perfect_matrix = get_information("results/perfect_", 24)
perfect_types_matrix = get_information("results/perfect_types_", 3)

"""
# Asociatividad multiple LRU
# bimod cache size cache misses l1
# 2lev cache size cache misses l1
# perfect cache size cache misses l1 """

def get_misses_l1( pred_mode, assoc ):
    aux_list = []
    for i in range(0,len(pred_mode[0])):
        if pred_mode[0][i].assoc == assoc:
            aux_list += [int(pred_mode[1][i].il1_misses)]
    
    return aux_list

print get_misses_l1(bimod_matrix, '1')

#####################################################################################
# BIMODAL PREDICTION
#####################################################################################
# plot assoc 1 
caches_sizes = [128,256,512,1024,2048,4096]
plt.plot( caches_sizes, get_misses_l1(bimod_matrix, '1'), label='Asociatividad 1' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 2
plt.plot( caches_sizes, get_misses_l1(bimod_matrix, '2'), label='Asociatividad 2' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 4
plt.plot( caches_sizes, get_misses_l1(bimod_matrix, '4'), label='Asociatividad 4' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 8
plt.plot( caches_sizes, get_misses_l1(bimod_matrix, '8'), label='Asociatividad 8' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")
plt.legend(loc='best')
plt.suptitle("Cache size vs Cache Misses para iL1 (Predictor bimodal)")
plt.show()

######################################################################################
# TWO LEVEL PREDICTOR
#####################################################################################
# plot assoc 1 
caches_sizes = [128,256,512,1024,2048,4096]
plt.plot( caches_sizes, get_misses_l1(b2lev_matrix, '1'), label='Asociatividad 1' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 2
plt.plot( caches_sizes, get_misses_l1(b2lev_matrix, '2'), label='Asociatividad 2' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 4
plt.plot( caches_sizes, get_misses_l1(b2lev_matrix, '4'), label='Asociatividad 4' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 8
plt.plot( caches_sizes, get_misses_l1(b2lev_matrix, '8'), label='Asociatividad 8' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")
plt.legend(loc='best')
plt.suptitle("Cache size vs Cache Misses para iL1 (Predictor de 2 niveles)")
plt.show()

######################################################################################
# PERFECT PREDICTION
######################################################################################
# plot assoc 1 
caches_sizes = [128,256,512,1024,2048,4096]
plt.plot( caches_sizes, get_misses_l1(perfect_matrix, '1'), label='Asociatividad 1' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 2
plt.plot( caches_sizes, get_misses_l1(perfect_matrix, '2'), label='Asociatividad 2' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 4
plt.plot( caches_sizes, get_misses_l1(perfect_matrix, '4'), label='Asociatividad 4' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")

# plot assoc 8
plt.plot( caches_sizes, get_misses_l1(perfect_matrix, '8'), label='Asociatividad 8' )
plt.xlabel("Cache size")
plt.ylabel("Cache misses iL1")
plt.legend(loc='best')
plt.suptitle("Cache size vs Cache Misses para iL1 (Predictor perfecto)")
plt.show()

##########################################################################################

"""
# Asociatividad multiple LRU
# bimod cache size cache misses l2
# 2lev cache size cache misses l2
# perfect cache size cache misses l2 """

"""
# Asociatividad multiple LRU
# bimod cache size cache misses d2
# 2lev cache size cache misses d2
# perfect cache size cache misses d2 """

"""
# Asociatividad multiple LRU
# bimod cache size cache misses d1
# 2lev cache size cache misses d1
# perfect cache size cache misses d1 """

""" 
# Asociatividad multiple LRU
# bimod cache size CPI
# 2lev cache size CPI
# perfect cache size CPI """



print (' -> END.')
