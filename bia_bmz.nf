#!/usr/bin/env nextflow

process run_analysis {
    
    input:
    val line

    output:
    val("analysis complete"), emit: analysis_complete

    script:
    """
    echo ${line}
    cd /
    cd /bia-bmz-integration
    set +u
    source ./bin/activate
    set -u
    cd ${workflow.launchDir}
    bia_bmz_benchmark ${line}
    """
}

process amalgamate_jsons {
    
    input:
    val(analysis_complete) 

    script:
    """
    cd /
    cd /bia-bmz-integration
    set +u
    source ./bin/activate
    set -u
    cd ${workflow.launchDir}
    amalgamate_jsons
    """
}

file_lines = Channel.fromPath(params.input_file).splitText() { it.trim() }

workflow {
    run_analysis(file_lines)
    amalgamate_jsons(run_analysis.out.analysis_complete.collect())
}
