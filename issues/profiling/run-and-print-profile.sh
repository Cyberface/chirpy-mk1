
# run this with
# bash run-and-print-profile.sh
python -m cProfile -o profiling_results wf-gen-profile.py
python -c "import pstats; stats = pstats.Stats('profiling_results'); stats.sort_stats('tottime'); stats.print_stats(20)"
