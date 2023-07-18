# code function
1. Construction a dynamic shape model by trt api
2. Build model to trt engine
3. Serialize engine and save as file
4. Load file and deserialize to engine
5. inference
# compile and run
```
cd build
cmake ..
make
./bin/exec
```
# Attention
* The Engine saved path in cpp file should be changed by yourself
* Compile this code should link cuda and trt libraries, check these lib path and change the lib path in CMakeLists.txt 
