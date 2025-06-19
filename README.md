# eozinpy-wsi-viewer

A simple whole slide image (WSI) viewer written in Python. 
This project serves as a practical example demonstrating the usage of eozin, yet another Python library for working with whole slide images.

![demo](docs/demo.gif)

## ✨ Features

- Just 350 lines of code. 
- Simple WSI Visualization: Easily view whole slide images. Use mouse drag to pan and mouse scroll to zoom.
- Backend Flexibility: Supports eozinpy as the primary backend and openslide-python as an alternative.

## 🚀 Getting Started

### Prerequisites

Before running the viewer, you need to have one of the following Python libraries installed:

- [eozinpy](https://github.com/yujota/eozin)
- [openslide-python](https://openslide.org/api/python/) (if you plan to use the --openslide flag)

Please refer to each web page for details

### Excecution

- Using `eozinpy` (default)

~~~sh
$ git clone https://github.com/yujota/eozinpy-wsi-viewer.git
$ cd eozinpy-wsi-viewer
$ python3 eozinpy_viewer/app.py path-to-wsi-file
~~~

- Using `openslide-python` as a backend

~~~ sh
$ python3 eozinpy_viewer/app.py path-to-wsi-file --openslide
~~~


## 📄 License

This project is licensed under the MIT License.
