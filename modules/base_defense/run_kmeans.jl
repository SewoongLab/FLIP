using Printf
using NPZ
include("spectre-defense/util.jl")
include("spectre-defense/kmeans_filters.jl")

input_folder = ARGS[1]
output_path = ARGS[2]
target_label = ARGS[3]
eps = parse(Int16,ARGS[4])

reps = npzread(input_folder * "$(target_label).npy")'
n = size(reps)[2]
removed = round(Int, 1.5*eps)

println("Running kmeans filter")
kmeans_poison_ind = .! kmeans_filter2(reps, eps)
poison_removed = sum(kmeans_poison_ind[end-eps+1:end])
clean_removed = removed - poison_removed
@printf("%d poisons removed, %d cleans removed\n", poison_removed, clean_removed)
npzwrite(output_path, kmeans_poison_ind)
