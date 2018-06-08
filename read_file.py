
# import libraries
import sys

# START 
print (' -> Starting...') 

# Global variables
terminal_parameters = sys.argv # terminal parameters string list
terminal_parameters_lenght = len( terminal_parameters ) # amount of terminal parameters
posible_parameters = {
    "-d" : "       - Debuging activated!",
    "-r" : "       - Reading files!",
}
debug_state = False # print debugs flag
read_state = False # read files flag

# Checking parameters
def check_parameters( p_terminal_paratemers, p_size_terminal_parameters ):
    
    global debug_state # Write global debug_state
    global read_state # Write global files_state

    if( p_size_terminal_parameters > 1 and p_size_terminal_parameters < 4 ): 
        print("       PARAMETERS DETECTED:")
        
        if parse_parameter( "-d", p_terminal_paratemers ):
            debug_state = True
        
        if parse_parameter( "-r", p_terminal_paratemers ):
            read_state = True
    else:
        sys.exit( "       - FAILED. Please check parameters!" )
    
    if( debug_state == False and read_state == False ):
        sys.exit( "       - FAILED. Please check parameters!" )
    
    if( p_size_terminal_parameters == 2 ):
        None

def parse_parameter( p_parameter, p_compare_list ):
    global posible_parameters   # Reads posible parameter dictionary

    if p_parameter in p_compare_list:
        print( posible_parameters[p_parameter] )
        return True
    else:
        return False

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
        self.dl1_misses = il1_misses
        self.dl2_misses = il2_misses
        self.sim_cycles = sim_cycles
        self.sim_IPC = sim_IPC
        self.sim_CPI = sim_CPI
        self.sim_elapsed_time = sim_elapsed_time

    def print_BenchmarkResults( self ):
        print " * OBJECT BenchmarkResults atributes * "
        print ("   - il1_misses : " + self.il1_misses)
        print ("   - il2_misses : " + self.il2_misses)
        print ("   - dl1_misses : " + self.il1_misses)
        print ("   - dl2_misses : " + self.dl2_misses)
        print ("   - sim_cycles : " + self.sim_cycles)
        print ("   - sim_IPC : " + self.sim_IPC)
        print ("   - sim_CPI : " + self.sim_CPI)
        print ("   - sim_elapsed_time (seconds) : " + self.sim_elapsed_time)

# Developer tools
def print_debug( msg ):
    global debug_state # Reads global debug_state
    if debug_state: print( msg )

benchmarks_list = []
results_list = []

# Read file
def read_information_from_file( path ):
    
    global benchmarks_list
    global results_list
    
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

    benchmarks_list += [aux_benchmark]
    results_list += [aux_result]

# Call functions
check_parameters( terminal_parameters, terminal_parameters_lenght )

# If i need to read all the different files, i need a list
# for bpred-2lev- this are the params
#first_test_params_0 = ['128','256','']

read_information_from_file( "results/bpred-2lev-128-1-4.txt" )

print (' -> END.')
