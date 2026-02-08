using CUDA: CUDA

include("cases.jl")
include("util.jl")

# Generate benchmarks
function run_benchmarks(cases, log2p, max_steps, ftype, backend, bstr; data_dir="./")
    for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
        println("Benchmarking: $(case)")
        suite = BenchmarkGroup()
        results = BenchmarkGroup([case, "sim_step!", p, s, ft, bstr, git_hash, string(VERSION)])
        add_to_suite!(suite, getf(case); p=p, s=s, ft=ft, backend=backend, bstr=bstr,
            remeasure = any(x -> x==case, ["cylinder", "jelly"])
        ) # create benchmark
        dev = first(CUDA.NVML.devices())
        energy_start = CUDA.NVML.energy_consumption(dev)
        total_time = @elapsed results[bstr] = run(suite[bstr], samples=1, evals=1, seconds=1e6, verbose=true) # run!
        energy_end = CUDA.NVML.energy_consumption(dev)
        fname = "$(case)_$(p...)_$(s)_$(ft)_$(bstr)_$(git_hash)_$VERSION.json"
        BenchmarkTools.save(joinpath(data_dir,fname), results)

        # Convert energy usage from Joule to Watt-Hour
        energy_wh = (energy_end - energy_start) / 3600
        json = """
        {
          "energy": $(energy_wh),
          "total_time": $(total_time)
        }
        """
        # Write an additional JSON file with energy usage and total run time
        # (not the average time step) of the benchmark.
        write(joinpath(data_dir, "energy-time.json"), json)
    end
end

cases, log2p, max_steps, ftype, backend, data_dir = parse_cla(ARGS;
    cases=["tgv", "jelly"], log2p=[(6,7), (5,6)], max_steps=[100, 100], ftype=[Float32, Float32], backend=Array, data_dir="data/"
)

# Generate benchmark data
data_dir = joinpath(data_dir, hostname * "_" * git_hash)
mkpath(data_dir)
run_benchmarks(cases, log2p, max_steps, ftype, backend, backend_str[backend]; data_dir)
