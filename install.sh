cd TyXe
python3 setup.py install
cd ..

cd pytorchviz
python3 setup.py install
cd ..

# install rest of requirements: torchvision etc for notebook
python3 -m pip install -r requirements.txt
