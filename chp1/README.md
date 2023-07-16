# code function
1. Construct a model by trt api
2. Build model to trt engine
3. Serialize engine and save as file
# compile and run
```
cd build
camke ..
make
./bin/exec
```
# attention
* The Engine saved path in cpp file should be changed by yourself
* Compile this code should link cuda and trt libraries, check these lib path and change the lib path in CMakeLists.txt 
