DEST=/home/fl237079/workspace/tao_built/
yes | cp ${DEST}/tao_compiler_main disc_dcu/
yes | cp ${DEST}/libtao_ops.so disc_dcu/
rm -rf dist 
python setup.py bdist_wheel
yes | pip uninstall  disc_dcu
pip install dist/*whl
rm -rf build
