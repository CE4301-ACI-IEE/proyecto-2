
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
# Class ResultBenchmark that store the information from
# the result of the simulated benchmarks
class ResultBenchmark:
    

# Developer tools
def print_debug( msg ):
    global debug_state # Reads global debug_state
    if debug_state: print( msg )

# Call functions
check_parameters( terminal_parameters, terminal_parameters_lenght )
print_debug( "impresion" )

print (' -> END.')
