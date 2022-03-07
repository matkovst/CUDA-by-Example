# CUDA-by-Example

Implemented for self-education working examples from "[CUDA by Example](https://developer.nvidia.com/cuda-example)" of Jason Sanders & Edward Kandrot. A little more personal samples also added here.</br>
Some examples use *OpenCV* for image loading and saving.

<details>
<summary>Whole list of examples</summary>
<br>
    <ul>
        <li>Dot product</li>
        <li>Matmul</li>
        <li>Julia set</li>
        <li>Mandelbrot set</li>
        <li>Ray tracing</li>
        <li>Ripple</li>
        <li>Heat transfer</li>
        <li>Histogram</li>
        <li>Grayscale</li>
    </ul>
<br><br>
</details>

## Requirements
- **Windows** / **Linux**
- **C++11** ( or higher ) compiler
- **CMake** >= 3.15
- **OpenCV** >= 3.0

## Build & Install

### CMake

The whole project can be built with CMake

```bash
export OpenCV_DIR=path/to/opencv

mkdir build && cd build
cmake ..
make
make install
```

### VS Code

Add to *.vscode/settings.json*

```bash
"cmake.configureSettings": 
{
    "OpenCV_DIR": "path/to/opencv"
}
```

and build project.

## Run
Choose desired task in *bin/* directory (like *dot*) and just execute it.
```bash
cd bin
./dot
```