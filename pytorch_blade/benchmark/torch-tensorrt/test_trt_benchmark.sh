
cwd=$(cd $(dirname "$0"); pwd)
cd $cwd
echo DIR: $(pwd)

wget -cnv https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com/download/torch-blade/benchmark/torch-tensorrt/models.tar.gz -O models.tar.gz
tar xfz models.tar.gz

python3 perf_run.py --config=config/vgg16.yml
python3 perf_run.py --config=config/yolov5.yml
python3 perf_run.py --config=config/crnn.yml

cd ..
