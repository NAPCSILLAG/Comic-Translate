# go.fish tartalma:
source .venv/bin/activate.fish
set -gx LD_LIBRARY_PATH /opt/cudnn8/lib64 $LD_LIBRARY_PATH
echo "Sikeresen beléptél a Comic Translate környezetbe!"
