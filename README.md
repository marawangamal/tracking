# tracking

### Usage

1. Install Deformable Convolution Networks package (dcn_v2)
```commandline
git clone https://github.com/CharlesShang/DCNv2.git
```
```commandline
cd DCNv2 && ./make.sh  # this adds it to PYTHONPATH
```

2. Download hockey pre-trained model and put it "src/models"
    - Go to https://www.dropbox.com/s/r8v8lvt87fm8cjm/TRACING-and-TEAMid.zip?dl=0
    - Download model found at tracking/models/hockey_20.pth

3. Run tracking algo
```commandline
python tracking/src/run_tracking.py mot --videos_dir /home/marawan/data/hockey_videos \
--load_model tracking/models/hockey_20.pth --conf_thres 0.5 --output-format video \
--output-root /home/marawan/data/hockey_videos_tracked
```
