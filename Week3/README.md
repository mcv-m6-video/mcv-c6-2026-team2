To install pyflow:

```bash
git submodule init # If no .gitmodules file is present
git submodule update # This will clone the 'pyflow' repo as a submodule inside the submodules folder

pip install Cython numpy # Requirements for pyflow

cd submodule/pyflow
pip install . # If using conda or python's virtual environments
#uv pip install . # If using uv as virtual environment manager
```

Y asi sinmas funsiona :) (seguir los pasos del repo oficial hace que se buildee en local, no en el virtual environment)