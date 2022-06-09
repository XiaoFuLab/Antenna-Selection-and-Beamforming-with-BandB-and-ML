ROOT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

export PATH=$ROOT_DIRECTORY:$PATH
export PYTHONPATH=$ROOT_DIRECTORY:$PYTHONPATH

source $ROOT_DIRECTORY/venv/bin/activate
