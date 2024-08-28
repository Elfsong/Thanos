import optuna
from optuna.trial import TrialState

import measure
import statistics
from tqdm import tqdm

def decode(lehmer_code: list[int]) -> list[int]:
    """
        Decode Lehmer code to permutation.
        This function decodes Lehmer code represented as a list of integers to a permutation.
    """
    all_indices = list(range(n))
    output = []
    for k in lehmer_code:
        value = all_indices[k]
        output.append(value)
        all_indices.remove(value)
    return output

code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100  // You can adjust this value to make the computation more intensive

void multiply_matrices(int **a, int **b, int **c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = 0;
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    // Allocate and initialize matrices
    int **a = malloc(N * sizeof(int*));
    int **b = malloc(N * sizeof(int*));
    int **c = malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        a[i] = malloc(N * sizeof(int));
        b[i] = malloc(N * sizeof(int));
        c[i] = malloc(N * sizeof(int));
    }

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = rand() % 10;
            b[i][j] = rand() % 10;
        }
    }

    // Measure start time
    clock_t start = clock();

    // Perform matrix multiplication
    multiply_matrices(a, b, c, N);

    // Measure end time
    clock_t end = clock();

    // Calculate and print the elapsed time in seconds
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", elapsed_time);

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    return 0;
}
"""

llvm_transform_passes = ['-simple-loop-unswitch',
 '-rewrite-symbols',
 '-attributor',
 '-break-crit-edges',
 '-bdce',
 '-dse',
 '-pa-eval',
 '-lower-global-dtors',
 '-O3',
 '-make-guards-explicit',
 '-openmp-opt-cgscc',
 '-del-rof',
 '-forceattrs',
 '-embed-bitcode',
 '-pseudo-probe-update',
 '-recompute-globalsaa',
 '-globaldce',
 '-loop-flatten',
 '-lower-expect',
 '-callsite-splitting',
 '-wholeprogramdevirt',
 '-jump-threading',
 '-newgvn',
 '-inliner-wrapper',
 '-rpo-function-attrs',
 '-globalsplit',
 '-late-dse-hop',
 '-rewrite-statepoints-for-gc',
 '-loop-rotate',
 '-loop-simplify',
 '-loop-versioning-licm',
 '-loop-inst-merge',
 '-aa-eval',
 '-extract-blocks',
 '-lower-ifunc',
 '-partially-inline-libcalls',
 '-inliner-wrapper-no-mandatory-first',
 '-instnamer',
 '-gvn-sink',
 '-early-globals-aa-globals',
 '-load-store-map',
 '-lowertypetests',
 '-O2',
 '-strip-nondebug',
 '-hotcoldsplit',
 '-scalarizer',
 '-lower-widenable-condition',
 '-name-anon-globals',
 '-synthetic-counts-propagation',
 '-coro-split',
 '-separate-const-offset-from-gep',
 '-deadargelim',
 '-aggressive-instcombine',
 '-internalize',
 '-scf-simplify-loop-latch',
 '-lowerswitch',
 '-div-rem-pairs',
 '-float2int',
 '-scc-oz-module-inliner',
 '-early-cse',
 '-indvars',
 '-Os',
 '-O0',
 '-lcssa',
 '-loop-partitioning',
 '-strip-nonlinetable-debuginfo',
 '-strip-dead-debug-info',
 '-canonicalize-aliases',
 '-partial-inliner',
 '-cross-dso-cfi',
 '-flattencfg',
 '-loop-fusion',
 '-irce',
 '-fade',
 '-strip-dead-prototypes',
 '-loop-data-prefetch',
 '-sink',
 '-inline',
 '-assume-simplify',
 '-tailcall',
 '-declare-to-assign',
 '-simplifycfg',
 '-artitioned-loop-numbering',
 '-loop-deletion',
 '-assume-builder',
 '-slp-vectorizer',
 '-adce',
 '-vector-combine',
 '-instsimplify',
 '-count-visits',
 '-openmp-opt-postlink',
 '-ipsccp',
 '-constmerge',
 '-loop-unroll',
 '-loop-bound-split',
 '-loop-bound-checking',
 '-mergefunc',
 '-no-op-cgscc',
 '-lower-matrix-intrinsics',
 '-loop-sink',
 '-loop-unroll-and-jam',
 '-elim-avail-extern',
 '-add-discriminators',
 '-lower-constant-intrinsics',
 '-implicitly-used-globals',
 '-likely-hoist-cond-loop-header-dominating-inst',
 '-dce',
 '-strip',
 '-sccp',
 '-openmp-opt',
 '-module-inline',
 '-fix-irreducible',
 '-loop-versioning',
 '-coro-cleanup',
 '-function-attrs',
 '-canon-freeze',
 '-Oz',
 '-inferattrs',
 '-dfa-jump-threading',
 '-slsr',
 '-rel-lookup-table-converter',
 '-gvn-hoist',
 '-loop-inline',
 '-called-value-propagation',
 '-consthoist',
 '-instcombine',
 '-loop-idiom',
 '-lower-guard-intrinsic',
 '-globalopt',
 '-licm',
 '-always-inline',
 '-annotation2metadata',
 '-loop-simplifycfg',
 '-loop-reduce',
 '-flatten-multicfg',
 '-nee-weave',
 '-iroutliner',
 '-loop-interchange',
 '-coro-early',
 '-argpromotion',
 '-loop-predication',
 '-ee-instrument',
 '-hardware-loops',
 '-alignment-from-assumptions',
 '-loop-block',
 '-loop-load-elim',
 '-attributor-cgscc',
 '-speculative-execution',
 '-loop-unroll-full',
 '-O1',
 '-strip-gc-relocates',
 '-strip-debug-declare']
n = len(llvm_transform_passes)
top_k = 9

def objective(trial: optuna.Trial) -> float:
    lehmer_code = [trial.suggest_int(f"p{i}", 0, n - i - 1) for i in range(top_k)]
    permutation = decode(lehmer_code)
    
    states_to_consider = (TrialState.COMPLETE,)
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)

    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value
        
    pass_sequence = [llvm_transform_passes[p] for p in permutation]
    cid = measure.PerfMeasure().create(code=code, pass_sequence=pass_sequence)

    runtimes = list()
    for _ in range(16):
        runtimes += [measure.PerfMeasure().measure(cid)]
        
    rt = statistics.fmean(runtimes)
    return rt

sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_jobs=1, n_trials=1000)

lehmer_code = study.best_params.values()
pass_index = decode(lehmer_code)
pass_sequence = [llvm_transform_passes[p] for p in pass_index]

print(pass_index)
print(pass_sequence)

cid = measure.PerfMeasure().create(code=code, pass_sequence=[])
r0 = measure.PerfMeasure().measure(cid)

cid = measure.PerfMeasure().create(code=code, pass_sequence=pass_sequence)
r1 = measure.PerfMeasure().measure(cid)

print(r0, r1, r1/r0)