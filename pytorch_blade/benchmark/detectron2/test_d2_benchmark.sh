cwd=$(cd $(dirname "$0"); pwd)
cd $cwd
echo DIR: $(pwd)

python3 run_blade.py --model testCascadeRCNN
python3 run_blade.py --model testMaskRCNNC4
python3 run_blade.py --model testMaskRCNNFPN
python3 run_blade.py --model testMaskRCNNFPN_b2
python3 run_blade.py --model testMaskRCNNFPN_pproc
python3 run_blade.py --model testRetinaNet
python3 run_blade.py --model testRetinaNet_scripted

cd ..
