echo $(pwd)
export build_path="$(pwd)/thirdparty/librealsense/build"
export python_path="$(pwd)/src/python_path"

cd $build_path && cmake ../ -DBUILD_PYTHON_BINDINGS=bool:true
cd $build_path && make -j4
cd $build_path && make install
mkdir -p $python_path
ls $build_path/wrappers/python/  
cp -r $build_path/Release/py* $python_path/
#cd $python_path && python3 setup.py build
echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/lib"
