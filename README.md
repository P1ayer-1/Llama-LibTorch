# Llama-LibTorch
Llama causal LM fully recreated in LibTorch. Designed to be used in Unreal Engine 5 

## Usage
Navigate to the root of this repo.

Then run these commands:
1. `mkdir build && cd build`
2. `cmake -DCMAKE_PREFIX_PATH=$PATH_TO_LIBTORCH ..` # ".." tells cmake what you are trying to compile. In this case, ".." points to the root of the Llama-LibTorch project. ".."= path to the parent of current directory
3. `cmake --build . --config Release ` # this actually builds the project and creates the .exe file
