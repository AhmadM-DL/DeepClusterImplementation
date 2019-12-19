pip install cudf-cuda100==0.6
apt install libopenblas-base libomp-dev
pip install cuml-cuda100
find / -name librmm.so* 
cp /usr/local/lib/python3.6/dist-packages/librmm.so . 
cp /usr/local/lib/python3.6/dist-packages/libcuml.so /usr/lib64-nvidia/
