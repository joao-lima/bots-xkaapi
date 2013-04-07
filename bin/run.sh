#!/bin/bash

export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

function run_test {
    export KAAPI_CPUSET="0"
    export KAAPI_GPUSET="0~1"

#    export KAAPI_CPUSET="0,5,6,11"
#    export KAAPI_GPUSET="0~1,1~2,2~3,3~4,4~7,5~8,6~9,7~10"

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"

#    export KAAPI_DUMP_GRAPH=1
#    export KAAPI_DOT_NOVERSION_LINK=1
#    export KAAPI_DOT_NODATA_LINK=1
#    export KAAPI_DOT_NOLABEL_INFO=1
#    export KAAPI_DOT_NOACTIVATION_LINK=1
#    export KAAPI_DOT_NOLABEL=1

#    export KAAPI_DISPLAY_PERF=1
#
#    export KAAPI_PUSH_AFFINITY="writer"
#    export KAAPI_PUSH_AFFINITY="heft"
#    export KAAPI_HEFT_CALIBRATE=1
#    export KAAPI_STEAL_AFFINITY="writer"
#    export KAAPI_PUSH_AFFINITY="locality"
#    export KAAPI_STEAL_AFFINITY="locality"


#    execfile="./strassen.gcc.kaapi"
    execfile="./sparselu.gcc.kaapi"
    msizes="2048"
    bsizes="512"
    niter=1
#    verif=1
    export KAAPI_CUDA_WINDOW_SIZE=2
#   KAAPI_STACKSIZE_MASTER=536870912 gdb $execfile
    KAAPI_STACKSIZE_MASTER=536870912 $execfile -n 128 -m 64 
}

run_test
exit 0
