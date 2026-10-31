Matplotlib offline package
Target: Python 3.10, Linux x86_64 (matches repository environment.yml).
Contains matplotlib 3.9.4 and its complete binary dependency closure.
NumPy 1.26.4 is provided as a fallback; existing compatible NumPy is retained by normal pip install.
No torch/CUDA packages are included or requested.

Upload matplotlib_offline_py310_linux.zip to the server project directory.
Activate the das-resshift environment, then run from that project directory:

python -m zipfile -e matplotlib_offline_py310_linux.zip .
python -m pip install --no-index --find-links matplotlib_offline_py310_linux/wheels -r matplotlib_offline_py310_linux/requirements-eval.txt
python -c "import matplotlib; print(matplotlib.__version__)"
python visualize_das.py --results outputs/das_4gpu_test/evaluation_results.json --output-dir outputs/das_4gpu_test/figures

If Python or CPU architecture differs from this target, do not use these binary wheels.
This archive has been validated for dependency completeness and ZIP integrity, but has not been installed on the server.
